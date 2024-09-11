import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import time
import argparse
from utils import generate_instructions_batch
from settings import VALID_ACTIONS
from concurrent.futures import ThreadPoolExecutor, as_completed


def clean_instruction(instruction):
    """
    Clean the instruction string by ensuring consistent quoting.
    """
    # Remove leading and trailing quotes
    instruction = instruction.strip().strip('"').strip("'")
    instruction = instruction.replace('"', '""')
    return f'"{instruction}"'


def generate_dataset(actions, num_samples_per_action=25, max_batch_size=50, num_threads=20):
    """
    Generate a dataset of driving instructions for the given actions.
    """
    data = []
    total_discarded = 0  # Counter for discarded (duplicate) instructions
    unique_instructions_per_action = {action: set() for action in actions}

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_action = {}
        while any(len(unique_instructions_per_action[action]) < num_samples_per_action for action in actions):
            # Count the number of currently running tasks
            running_tasks = len(future_to_action)

            # If running tasks are less than the max number of threads, submit new tasks
            if running_tasks < num_threads:
                for action in actions:
                    if len(unique_instructions_per_action[action]) < num_samples_per_action:
                        remaining_samples = max(num_samples_per_action - len(unique_instructions_per_action[action]), 5)
                        batch_size = min(remaining_samples * 2, max_batch_size)
                        print(f"Submitting batch generation for action '{action}' with batch size {batch_size}")
                        future = executor.submit(generate_instructions_batch, action, batch_size)
                        future_to_action[future] = action

            # Collect results as tasks complete
            for future in as_completed(future_to_action):
                action = future_to_action[future]
                try:
                    batch_instructions = future.result()
                    initial_size = len(unique_instructions_per_action[action])
                    unique_instructions_per_action[action].update(batch_instructions)
                    new_size = len(unique_instructions_per_action[action])
                    discarded = batch_size - (new_size - initial_size)
                    total_discarded += discarded
                    print(f"Action '{action}': Generated {new_size} / {num_samples_per_action} unique instructions (Discarded {discarded} duplicates)")
                    if len(unique_instructions_per_action[action]) > num_samples_per_action:
                        unique_instructions_per_action[action] = set(list(unique_instructions_per_action[action])[:num_samples_per_action])
                except Exception as exc:
                    print(f"An error occurred for action '{action}': {exc}")
                del future_to_action[future]
                # Break early if we've already met the required number of samples for the current action
                if len(unique_instructions_per_action[action]) >= num_samples_per_action:
                    break

    for action in actions:
        for instruction in unique_instructions_per_action[action]:
            cleaned_instruction = clean_instruction(instruction)
            data.append((cleaned_instruction, [action]))

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
        print(dataset)
        dataset.to_csv(args.output_file, index=False)
        print(f"Dataset generated and saved to '{args.output_file}'")
        duration = time.time() - start_time
        print(f"Generated {len(dataset)} instructions in {duration:.2f} seconds.")
    else:
        print("API key not found. Please check your .env file.")


if __name__ == "__main__":
    main()
