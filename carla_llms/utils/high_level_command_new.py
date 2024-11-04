import numpy as np
import carla
import math
import os

class HighLevelCommandLoader:
    def __init__(self, vehicle, carla_map,route=None, start_point_id=None, end_point_id=None):
        self.vehicle = vehicle
        self.map = carla_map
        self.prev_hlc = 0
        self.route = route
        self.start_point_id = start_point_id

        self.end_point_id = end_point_id

        print(start_point_id, end_point_id)
        
        self.start_location = None
        self.dist_travelled = 0.0  # Ensure distance is reset at initialization
        self.command_sequence = []  # Placeholder for the current episode's command sequence

        # Load commands based on start and end points
        self.load_commands()

    def _command_to_int(self, command):
        """Convert string command to integer."""
        commands = {
            'Left': 1,
            'Right': 2,
            'Straight': 3
        }
        return commands[command]

    def load_commands(self):
        """Load commands based on predefined start and end points."""
        if (self.start_point_id, self.end_point_id) == (59, 39):  # Episode from 59 to 39
            distance_1 = 4    # Right turn at 4 meters
            distance_2 = 48   # Straight drive at 48 meters
            command_1 = 'Right'
            command_2 = 'Straight'
        elif (self.start_point_id, self.end_point_id) == (32, 75):  # Episode from 32 to 75
            distance_1 = 82   # Right turn at 82 meters
            distance_2 = 132  # Straight drive at 132 meters
            command_1 = 'Right'
            command_2 = 'Straight'
        else:
            print(f"Unknown spawn points: {self.start_point_id}, {self.end_point_id}")
            return

        # Convert commands to integers
        command_1_int = self._command_to_int(command_1)
        command_2_int = self._command_to_int(command_2)

        # Set the command sequence for this episode
        self.command_sequence = [
            (distance_1, command_1_int),
            (distance_2, command_2_int)
        ]

        # Print the episode details for debugging (optional)
        print(f"Loaded commands: Start ({self.start_point_id}, {self.end_point_id}), "
              f"Commands: {command_1} at {distance_1} meters, "
              f"{command_2} at {distance_2} meters")

    def get_next_hlc(self):
        """Fetch the next high-level command (HLC) based on the distance traveled."""
        return self.load_next_hlc()

    def find_dist(self, loc1, loc2):
        """Calculate the distance between two locations."""
        return math.sqrt((loc2.x - loc1.x)**2 + (loc2.y - loc1.y)**2 + (loc2.z - loc1.z)**2)

    def update_distance_travelled(self):
        """Update the cumulative distance travelled by the vehicle."""
        current_location = self.vehicle.get_transform().location

        # If it's the first calculation, initialize start_location
        if self.start_location is None:
            self.start_location = current_location
            self.dist_travelled = 0.0
        else:
            # Calculate the incremental distance from the previous location to the current location
            incremental_distance = self.find_dist(self.start_location, current_location)

            # Add the incremental distance to the total distance
            self.dist_travelled += incremental_distance

            # Update start_location to the current location for the next iteration
            self.start_location = current_location

    def load_next_hlc(self):
        """Load the next high-level command based on predefined distances."""
        if self.prev_hlc is None:
            return None

        # Update the distance traveled by the vehicle
        self.update_distance_travelled()

        print("DISTANCE_TRAVELED:", self.dist_travelled)
        print("current_location:", self.vehicle.get_transform().location)

        # Check if the vehicle has traveled enough distance to trigger the first command
        if self.command_sequence and self.command_sequence[0][0] <= self.dist_travelled < self.command_sequence[1][0]:
            hlc = self.command_sequence[0][1]  # Right turn (2)
            print(f"Triggering Right Turn at {self.command_sequence[0][0]} meters")

        # Check if the vehicle has traveled enough distance for the second command
        elif self.command_sequence and self.dist_travelled >= self.command_sequence[1][0]:
            hlc = self.command_sequence[1][1]  # Drive straight (3)
            print(f"Triggering Straight drive at {self.command_sequence[1][0]} meters")

        else:
            # If distance traveled is less than the first command, keep the previous HLC
            hlc = self.prev_hlc

        self.prev_hlc = hlc
        return hlc

