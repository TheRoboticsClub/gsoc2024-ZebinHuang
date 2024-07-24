import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import time
import argparse
from utils import generate_instruction
from settings import VALID_ACTIONS


def generate_dataset(actions, num_samples_per_action=25):
    """
    Generate a dataset of driving instructions for the given actions.
    """
    data = []
    generated_instructions = set()  # To keep track of unique instructions
    discarded_count = 0  # Counter for discarded (duplicate) instructions
    for action in actions:
        count = 0
        while count < num_samples_per_action:
            instruction = generate_instruction(action)
            if instruction not in generated_instructions:
                data.append((instruction, [action]))
                generated_instructions.add(instruction)
                count += 1
                print(f"Action '{action}': Generated {count} / {num_samples_per_action} unique instructions")
            else:
                discarded_count += 1
    print(f"Total discarded (duplicate) instructions: {discarded_count}")
    return pd.DataFrame(data, columns=['instruction', 'action'])


def main():
    parser = argparse.ArgumentParser(description="Generate a dataset of driving instructions.")
    parser.add_argument('--actions', nargs='+', default=VALID_ACTIONS, help="List of actions to generate instructions for.")
    parser.add_argument('--num_samples', type=int, default=25, help="Number of unique samples per action.")
    parser.add_argument('--output_file', type=str, default='dataset.csv', help="Output CSV file to save the dataset.")
    parser.add_argument('--max_batch_size', type=int, default=50, help="Maximum batch size for generating instructions.")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        start_time = time.time()
        dataset = generate_dataset(args.actions, args.num_samples, args.max_batch_size)
        dataset.to_csv(args.output_file, index=False)
        print(f"Dataset generated and saved to '{args.output_file}'")
        duration = time.time() - start_time
        print(f"Generated {len(dataset)} instructions in {duration:.2f} seconds.")
    else:
        print("API key not found. Please check your .env file.")


if __name__ == "__main__":
    main()
