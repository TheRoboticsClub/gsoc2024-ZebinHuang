import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import time
import argparse
from utils import generate_instructions_batch
from settings import VALID_ACTIONS


def generate_dataset(actions, num_samples_per_action=25, max_batch_size=50):
    """
    Generate a dataset of driving instructions for the given actions.
    """
    data = []
    total_discarded = 0  # Counter for discarded (duplicate) instructions
    for action in actions:
        unique_instructions = set()
        while len(unique_instructions) < num_samples_per_action:
            # Ensure at least 5 samples per batch
            remaining_samples = max(num_samples_per_action - len(unique_instructions), 5)
            # Dynamically set batch size, capped at max_batch_size
            batch_size = min(remaining_samples * 2, max_batch_size)
            batch_instructions = generate_instructions_batch(action, batch_size)
            initial_size = len(unique_instructions)
            unique_instructions.update(batch_instructions)
            new_size = len(unique_instructions)
            # Calculate discarded instructions
            total_discarded += (batch_size - (new_size - initial_size))
            if len(unique_instructions) > num_samples_per_action:
                unique_instructions = set(list(unique_instructions)[:num_samples_per_action])
            print(f"Action '{action}': Generated {len(unique_instructions)} / {num_samples_per_action} unique instructions")
        for instruction in unique_instructions:
            data.append((instruction, [action]))
    print(f"Total discarded (duplicate) instructions: {total_discarded}")
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
