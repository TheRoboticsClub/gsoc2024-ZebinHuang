import os
import sys
import math
import time
import carla
import glob


def update_bird_view(vehicle, spectator):
    transform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(
        transform.location + carla.Location(z=50),
        carla.Rotation(pitch=-90)
    ))


def update_relative_view(vehicle, spectator, offset):
    if offset:
        transform = vehicle.get_transform()
        spectator.set_transform(carla.Transform(
            transform.location + offset.location,
            carla.Rotation(pitch=offset.rotation.pitch, yaw=offset.rotation.yaw)
        ))


def calculate_distance(loc1, loc2):
    return math.sqrt((loc2.x - loc1.x) ** 2 + (loc2.y - loc1.y) ** 2)


def drive_forward(vehicle, distance, speed, world, spectator, update_view_fn):
    start_location = vehicle.get_location()
    while True:
        if calculate_distance(start_location, vehicle.get_location()) >= distance:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            break
        vehicle.apply_control(carla.VehicleControl(throttle=speed))
        update_view_fn(vehicle, spectator)
        world.tick()


def perform_u_turn(vehicle, world, reverse=False):
    target_yaw = -173
    global update_spectator_view
    update_spectator_view = False

    while abs(vehicle.get_transform().rotation.yaw - target_yaw) > 2.0:
        control = carla.VehicleControl(
            throttle=-0.3 if reverse else 0.3,
            steer=1.0 if reverse else -1.0
        )
        vehicle.apply_control(control)
        world.tick()

    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
    update_spectator_view = True


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    world = client.load_world('Town01_Opt')
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_point = carla.Transform(carla.Location(x=14.5, y=2.3, z=1.5))
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    spectator = world.get_spectator()

    try:
        drive_forward(vehicle, 100, 0.5, world, spectator, update_bird_view)
        time.sleep(3)
        perform_u_turn(vehicle, world)
        time.sleep(3)

        vehicle_transform = vehicle.get_transform()
        spectator_transform = spectator.get_transform()
        relative_offset = carla.Transform(
            spectator_transform.location - vehicle_transform.location,
            spectator_transform.rotation
        )

        drive_forward(vehicle, 100, 0.5, world, spectator, lambda v, s: update_relative_view(v, s, relative_offset))
    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Simulation completed.")


if __name__ == "__main__":
    main()
