import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import numpy as np
import carla
import logging


class HighLevelCommandLoader:
    def __init__(self, vehicle, map, route=None):
        self.vehicle = vehicle
        self.map = map
        self.prev_hlc = 0
        self.route = route

    def initialize_model(self, model_path, tokenizer_name, label_mapping_path):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        with open(label_mapping_path, 'r') as f:
            label_mapping = json.load(f)
        self.label_mapping = {int(v): k for k, v in label_mapping.items()}

        num_labels = len(self.label_mapping)

        # Load the pre-trained model with the appropriate number of labels
        self.model = BertForSequenceClassification.from_pretrained(
            tokenizer_name,
            num_labels=num_labels
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _get_junction(self):
        """Determine whether vehicle is at junction"""
        junction = None
        vehicle_location = self.vehicle.get_transform().location
        vehicle_waypoint = self.map.get_waypoint(vehicle_location)

        # Check whether vehicle is at junction
        for j in range(1, 11):
            next_waypoint = vehicle_waypoint.next(j * 1.0)[0]
            if next_waypoint.is_junction:
                junction = next_waypoint.get_junction()
                break
        if vehicle_waypoint.is_junction:
            junction = vehicle_waypoint.get_junction()
        return junction

    def _command_to_int(self, command):
        commands = {
            'Left': 1,
            'Right': 2,
            'Straight': 3
        }
        return commands[command]

    def _predict_instruction(self, instruction):
        encodings = self.tokenizer(instruction, truncation=True, padding=True, max_length=128, return_tensors='pt')
        encodings = {key: val.to(self.device) for key, val in encodings.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encodings)
            pred = torch.argmax(outputs.logits, axis=1).item()

        return self.label_mapping[pred]

    def get_random_hlc(self):
        """Select a random turn at junction"""
        junction = self._get_junction()
        vehicle_location = self.vehicle.get_transform().location
        vehicle_waypoint = self.map.get_waypoint(vehicle_location)

        # Randomly select a turning direction
        if junction is not None:
            if self.prev_hlc == 0:
                valid_turns = []
                waypoints = junction.get_waypoints(carla.LaneType.Driving)
                for next_wp, _ in waypoints:
                    yaw_diff = next_wp.transform.rotation.yaw - vehicle_waypoint.transform.rotation.yaw
                    yaw_diff = (yaw_diff + 180) % 360 - 180 # Convert to [-180, 180]
                    if -15 < yaw_diff < 15:
                        valid_turns.append(3)  # Go Straight
                    elif 15 < yaw_diff < 165:
                        valid_turns.append(1)  # Turn Left
                    elif -165 < yaw_diff < -15:
                        valid_turns.append(2)  # Turn Right
                hlc = np.random.choice(valid_turns)
            else:
                hlc = self.prev_hlc
        else:
            hlc = 0

        self.prev_hlc = hlc

        return hlc

    def get_next_hlc(self):
        if self.route is not None and len(self.route) > 0:
            return self.load_next_hlc()
        return self.get_random_hlc()

    def load_next_hlc(self):
        """Load the next high level command from pre-defined route"""
        if self.prev_hlc is None:
            return None

        junction = self._get_junction()

        if junction is not None:
            if self.prev_hlc == 0:
                if len(self.route) == 0:
                    return None
                instruction = self.route.pop(0)
                logging.info('User instructions: %s', instruction)
                logging.info('Predicted action: %s', self._predict_instruction(instruction))
                logging.info('Action index: %s', self._command_to_int(self._predict_instruction(instruction)))
                hlc = self._command_to_int(self._predict_instruction(instruction))
            else:
                hlc = self.prev_hlc
        else:
            hlc = 0

        self.prev_hlc = hlc

        return hlc
