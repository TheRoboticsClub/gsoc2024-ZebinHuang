import pandas as pd
from utils import generate_instructions_batch
from concurrent.futures import ThreadPoolExecutor, as_completed


def clean_instruction(instruction):
    """
    Clean the instruction string by ensuring consistent quoting.
    """
    # Remove leading and trailing quotes
    instruction = instruction.strip().strip('"').strip("'")
    instruction = instruction.replace('"', '""')
    return f'"{instruction}"'


def generate_dataset(actions, num_samples_per_action=25, max_batch_size=50, num_threads=20, progress_bar=None, **custom_params):
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
                        remaining_samples = max(
                            num_samples_per_action - len(unique_instructions_per_action[action]), 5
                        )
                        batch_size = min(remaining_samples * 2, max_batch_size)
                        print(
                            f"Submitting batch generation for action '{action}' with batch size {batch_size}"
                        )
                        future = executor.submit(
                            generate_instructions_batch, action, batch_size, **custom_params
                        )
                        future_to_action[future] = action

            # Collect results as tasks complete
            for future in as_completed(future_to_action):
                action = future_to_action.pop(future)
                try:
                    batch_instructions = future.result()
                    initial_size = len(unique_instructions_per_action[action])
                    unique_instructions_per_action[action].update(batch_instructions)
                    new_size = len(unique_instructions_per_action[action])
                    discarded = batch_size - (new_size - initial_size)
                    total_discarded += discarded
                    print(
                        f"Action '{action}': Generated {new_size} / {num_samples_per_action} "
                        f"unique instructions (Discarded {discarded} duplicates)"
                    )
                    if len(unique_instructions_per_action[action]) >= num_samples_per_action:
                        unique_instructions_per_action[action] = set(
                            list(unique_instructions_per_action[action])[:num_samples_per_action]
                        )
                except Exception as exc:
                    print(f"An error occurred for action '{action}': {exc}")

                # Update progress bar
                total_instructions = sum(len(unique_instructions_per_action[action]) for action in actions)
                total_required_samples = num_samples_per_action * len(actions)
                total_progress = total_instructions / total_required_samples if total_required_samples > 0 else 0
                if progress_bar:
                    progress_bar.progress(total_progress)

                # Break early if we've already met the required number of samples for the current action
                if len(unique_instructions_per_action[action]) >= num_samples_per_action:
                    break

    for action in actions:
        for instruction in unique_instructions_per_action[action]:
            cleaned_instruction = clean_instruction(instruction)
            data.append((cleaned_instruction, [action]))

    print(f"Total discarded (duplicate) instructions: {total_discarded}")
    return pd.DataFrame(data, columns=['instruction', 'action'])
