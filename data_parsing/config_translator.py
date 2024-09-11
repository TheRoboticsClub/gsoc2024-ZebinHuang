import os
import argparse
from tqdm import tqdm
from utils import generate_instruction


def translate_actions(configs):
    """
    Translate the actions data to driving instructions.
    """
    translated_configs = []
    for config in tqdm(configs, desc="Translating actions"):
        try:
            parts = config.split()
            # Format validation
            if len(parts) < 4:
                raise ValueError(f"Invalid format: {config}")
            start_idx, end_idx, length = parts[:3]
            actions = parts[3:]
            translated_actions = [generate_instruction(action) for action in actions]
            translated_config = f"{start_idx} {end_idx} {length} {' '.join(translated_actions)}"
            translated_configs.append(translated_config)
        except Exception as e:
            print(f"Error processing config '{config}': {e}")
    return translated_configs


def load_configs_from_file(file_path):
    """
    Load configuration data from a specified file.
    """
    with open(file_path, 'r') as file:
        configs = file.readlines()
    return [config.strip() for config in configs]


def save_translated_configs(translated_configs, output_file):
    """
    Save the translated data to a specified output file.
    """
    with open(output_file, 'w') as file:
        for config in translated_configs:
            file.write(config + '\n')


def process_file(file_path, output_dir):
    """
    Process a single configuration file and save the translated data.
    """
    configs = load_configs_from_file(file_path)
    translated_configs = translate_actions(configs)
    output_file = os.path.join(output_dir, os.path.basename(file_path))
    save_translated_configs(translated_configs, output_file)


def process_configs(input_path, output_dir):
    """
    Process the configuration files from a directory or a single file and save the translated data.
    """
    if os.path.isdir(input_path):
        # Traverse the directory and process each file
        for root, dirs, files in os.walk(input_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                print(f"Processing file: {file_path}")
                process_file(file_path, output_dir)
    else:
        print(f"Processing file: {input_path}")
        process_file(input_path, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Convert configuration data to driving instructions.")
    parser.add_argument('input_path', type=str, help="Path to the input file or directory.")
    parser.add_argument('output_dir', type=str, help="Path to the output file to save translated configurations.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Output directory '{args.output_dir}' does not exist. Created new directory.")
    process_configs(args.input_path, args.output_dir)


if __name__ == "__main__":
    main()
