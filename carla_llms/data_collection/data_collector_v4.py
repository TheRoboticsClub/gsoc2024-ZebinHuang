import argparse
import logging
import random

import carla
import h5py
import numpy as np

from agent import NoisyTrafficManagerAgent
from sensors import RGBCamera, SegmentationCamera, setup_collision_sensor
from utils.preprocess import (carla_rgb_to_array, carla_seg_to_array, read_routes,
                              road_option_to_int, traffic_light_to_int)
from utils.traffic import (cleanup, get_traffic_light_status, spawn_pedestrians,
                           spawn_vehicles)

has_collision = False


def collision_callback(data):
    global has_collision
    has_collision = True


def setup_carla_world(params):
    client = carla.Client(params.ip, params.port)
    client.set_timeout(params.timeout)

    world = client.get_world()
    if world.get_map().name.split('/')[-1] != params.map:
        world = client.load_world(params.map)
    world.set_weather(carla.WeatherParameters.ClearNoon)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    return world, client


def setup_traffic_manager(client, params):
    traffic_manager = client.get_trafficmanager(params.tm_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_hybrid_physics_mode(True)
    traffic_manager.set_hybrid_physics_radius(70.0)

    return traffic_manager


def main_loop(world, client, traffic_manager, params):
    episode_configs = read_routes(params.episode_file)
    episode_cnt = 0

    while episode_cnt < params.n_episodes:
        logging.info(f'episode {episode_cnt + 1}')
        handle_episode(world, client, traffic_manager, episode_configs, params, episode_cnt)
        episode_cnt += 1


def handle_episode(world, client, traffic_manager, episode_configs, params, episode_cnt):
    global has_collision
    has_collision = False

    episode_config, vehicle = setup_episode(world, traffic_manager, episode_configs, params)
    agent, spectator = setup_agent_and_spectator(vehicle, traffic_manager, episode_config, world)

    all_id, all_actors, pedestrians_list, vehicles_list = spawn_dynamic_agents(world, client, traffic_manager, params)

    rgb_cam, seg_cam, sensors = setup_sensors(world, vehicle)
    for _ in range(10):
        world.tick()

    collect_data_for_episode(vehicle, agent, spectator, world, params, episode_cnt, rgb_cam, seg_cam)

    cleanup(vehicles_list, pedestrians_list, all_id, all_actors, vehicle, sensors, client)


def setup_episode(world, traffic_manager, episode_configs, params):
    spawn_points = world.get_map().get_spawn_points()
    episode_config = random.choice(episode_configs)
    blueprint_library = world.get_blueprint_library()
    blueprint = blueprint_library.filter('model3')[0]
    blueprint.set_attribute('role_name', 'hero')
    start_point = spawn_points[episode_config[0][0]]
    vehicle = world.spawn_actor(blueprint, start_point)

    configure_vehicle_for_traffic_manager(vehicle, traffic_manager, params)

    return episode_config, vehicle


def configure_vehicle_for_traffic_manager(vehicle, traffic_manager, params):
    vehicle.set_autopilot(True)
    if params.ignore_traffic_light:
        traffic_manager.ignore_lights_percentage(vehicle, 100)
    traffic_manager.ignore_signs_percentage(vehicle, 100)
    traffic_manager.distance_to_leading_vehicle(vehicle, 4.0)
    traffic_manager.set_desired_speed(vehicle, params.target_speed)


def setup_agent_and_spectator(vehicle, traffic_manager, episode_config, world):
    agent = NoisyTrafficManagerAgent(vehicle, traffic_manager)
    route = episode_config[2]
    end_point = world.get_map().get_spawn_points()[episode_config[0][1]]
    agent.set_route(route, end_point)

    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

    return agent, spectator


def spawn_dynamic_agents(world, client, traffic_manager, params):
    all_id, all_actors, pedestrians_list, vehicles_list = [], [], [], []

    if params.n_vehicles > 0:
        vehicles_list = spawn_vehicles(world, client, params.n_vehicles, traffic_manager)

    if params.n_pedestrians > 0:
        all_id, all_actors, pedestrians_list = spawn_pedestrians(world, client, params.n_pedestrians)

    logging.info('spawned %d vehicles and %d walkers.' % (len(vehicles_list), len(pedestrians_list)))

    return all_id, all_actors, pedestrians_list, vehicles_list


def setup_sensors(world, vehicle):
    rgb_cam = RGBCamera(world, vehicle)
    seg_cam = SegmentationCamera(world, vehicle)
    collision_sensor = setup_collision_sensor(world, vehicle)
    collision_sensor.listen(collision_callback)
    sensors = [rgb_cam.get_sensor(), seg_cam.get_sensor(), collision_sensor]
    return rgb_cam, seg_cam, sensors


def collect_data_for_episode(vehicle, agent, spectator, world, params, episode_cnt, rgb_cam, seg_cam):
    episode_data = {
        'frame': [],
        'hlc': [],
        'light': [],
        'controls': [],
        'measurements': [],
        'rgb': [],
        'segmentation': [],
        'distance_to_next_wp': [],
        'distance_traveled': [],
        'distance_to_stop_line': []
    }

    frame = 0
    prev_location = vehicle.get_transform().location
    accumulated_distance = 0
    while True:
        if should_end_episode(agent, frame, params):
            break

        update_spectator(vehicle, spectator)
        world.tick()
        accumulated_distance = process_frame(vehicle, agent, episode_data, frame, rgb_cam, seg_cam, prev_location, accumulated_distance, world)

        prev_location = vehicle.get_transform().location
        frame += 1

    useful_data = False
    for value in episode_data['hlc']:
        if value == 4:
            useful_data = True
            # logging.info("This episode contains hlc[4]. Saving now.")

    # if not has_collision and frame < params.max_frames_per_episode:
    if not has_collision and useful_data:
        save_episode_data(episode_data, params.dataset_path, episode_cnt)


def should_end_episode(agent, frame, params):
    done = False
    if frame >= params.max_frames_per_episode:
        logging.info("Maximum frames reached, episode ending")
        done = True
    elif has_collision:
        logging.info("Collision detected! episode ending")
        done = True
    return done


def calculate_distances(vehicle, prev_location, world):
    # Distance to next waypoint
    next_wp = get_next_waypoint(vehicle, world)
    if next_wp:
        next_wp_location = next_wp.transform.location
        distance_to_next_wp = vehicle.get_transform().location.distance(next_wp_location)
    else:
        distance_to_next_wp = float('inf')  # If no waypoint found

    # Distance traveled
    distance_traveled = vehicle.get_transform().location.distance(prev_location)

    # Distance to stop line (use traffic light sensor or map data)
    traffic_light_status = get_traffic_light_status(vehicle)
    if traffic_light_status == "RED":
        distance_to_stop_line = get_distance_to_stop_line(vehicle, world)
    else:
        distance_to_stop_line = float('inf')  # No stop required

    return {
        'distance_to_next_wp': distance_to_next_wp,
        'distance_traveled': distance_traveled,
        'distance_to_stop_line': distance_to_stop_line
    }


def process_frame(vehicle, agent, episode_data, frame, rgb_cam, seg_cam, prev_location, accumulated_distance, world):
    control, noisy_control = agent.run_step()
    if agent.done():
        if not hasattr(agent, 'target_logged'):
            # logging.info("The target has been reached, vehicle stopping")
            agent.target_logged = True

        vehicle.set_autopilot(False)
        control.throttle = 0
        control.steer = 0
        control.brake = 1
        vehicle.apply_control(control)
    elif noisy_control:
        vehicle.apply_control(noisy_control)
    velocity = vehicle.get_velocity()
    speed_km_h = (3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))  # m/s to km/h

    frame_data = {}

    # New added distances
    distances = calculate_distances(vehicle, prev_location, world)
    if not agent.noise:
        if agent.reaching_countdown_distance():
            accumulated_distance += distances['distance_traveled']
            # logging.info("vehicle reaching countdown distance")
            frame_data = {
                'frame': np.array([frame]),
                'hlc': np.array([4]),
                'light': np.array([traffic_light_to_int(get_traffic_light_status(vehicle))]),
                'controls': np.array([control.throttle, control.steer, control.brake]),
                'measurements': np.array([speed_km_h]),
                'distance_to_next_wp': np.array([distances['distance_to_next_wp']]),
                'distance_traveled': np.array([accumulated_distance]),
                'distance_to_stop_line': np.array([distances['distance_to_stop_line']]),
                'rgb': np.copy(carla_rgb_to_array(rgb_cam.get_sensor_data())),
                'segmentation': np.copy(carla_seg_to_array(seg_cam.get_sensor_data())),
            }
        else:
            frame_data = {
                'frame': np.array([frame]),
                'hlc': np.array([road_option_to_int(agent.get_next_action())]),
                'light': np.array([traffic_light_to_int(get_traffic_light_status(vehicle))]),
                'controls': np.array([control.throttle, control.steer, control.brake]),
                'measurements': np.array([speed_km_h]),
                'distance_to_next_wp': np.array([distances['distance_to_next_wp']]),
                'distance_traveled': np.array([0]),
                'distance_to_stop_line': np.array([distances['distance_to_stop_line']]),
                'rgb': np.copy(carla_rgb_to_array(rgb_cam.get_sensor_data())),
                'segmentation': np.copy(carla_seg_to_array(seg_cam.get_sensor_data())),
            }

        for key, value in frame_data.items():
            episode_data[key].append(value)
    return accumulated_distance


def get_distance_to_stop_line(vehicle, world):
    # Placeholder for now
    stop_line_distance = random.uniform(0, 30)
    return stop_line_distance


def get_next_waypoint(vehicle, world, distance=1.0):
    carla_map = world.get_map()
    current_location = vehicle.get_transform().location
    current_waypoint = carla_map.get_waypoint(current_location)

    next_waypoints = current_waypoint.next(distance)

    if next_waypoints:
        return next_waypoints[0]  # return the first next waypoint
    else:
        return None


def update_spectator(vehicle, spectator):
    vehicle_location = vehicle.get_transform().location
    spectator.set_transform(carla.Transform(vehicle_location + carla.Location(z=50), carla.Rotation(pitch=-90)))


def save_episode_data(episode_data, dataset_path, episode_cnt):
    with h5py.File(f'{dataset_path}/episode_{episode_cnt + 1}.hdf5', 'w') as file:
        for key, data_list in episode_data.items():
            data_array = np.array(data_list)
            file.create_dataset(key, data=data_array, maxshape=(None,)+data_array.shape[1:])


def main(params):
    world, client = setup_carla_world(params)
    traffic_manager = setup_traffic_manager(client, params)
    world.tick()
    main_loop(world, client, traffic_manager, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--map', default='Town01')
    parser.add_argument('--tm_port', type=int, default=8000)
    parser.add_argument('--timeout', type=int, default=100)
    parser.add_argument('--episode_file', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--n_vehicles', type=int, default=10)
    parser.add_argument('--n_pedestrians', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=4)
    parser.add_argument('--max_frames_per_episode', type=int, default=3000)
    parser.add_argument('--target_speed', type=int, default=40)
    parser.add_argument('--ignore_traffic_light', action="store_true")

    params = parser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    main(params)
