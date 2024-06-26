from email import message
from pyexpat.errors import messages
import openai
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os
import time

# Load environment variables from .env file
load_dotenv()


def generate_instruction_prompt():
    return """
    Generate a random driving instruction that a human might give to an autonomous vehicle. Include details such as directions, actions, and landmarks. Here are three examples:
    1. "At the next intersection, turn right and then immediately merge onto the freeway."
    2. "Keep going straight until you see the big mall on your left, then slow down and prepare to stop."
    3. "Accelerate to match the highway speed limit as you pass the gas station."
    Generate another instruction:
    """


def generate_instruction(api_key):
    openai.api_key = api_key
    prompt = [
        {"role": "system", "content": "You are a highly knowledgeable virtual assistant."},
        {"role": "user", "content": generate_instruction_prompt()}
    ]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # TODO: test other models
        messages=prompt,
        max_tokens=100
    )
    instruction = response.choices[0].message.content
    return instruction


def generate_action_label(instruction, api_key):
    openai.api_key = api_key
    prompt_init = f"Based on the instruction: '{instruction}', what is the most appropriate driving action label? The options are Turn Left, Turn Right, Take Exit, Go Straight, Accelerate, Slow Down."
    prompt = [
        {"role": "system", "content": "I will give you instruction. Based on the instruction, you should tell me that what is the most appropriate driving action label? The options are Turn Left, Turn Right, Take Exit, Go Straight, Accelerate, Slow Down. Just reply the options."},
        {"role": "user", "content": prompt_init}
    ]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        max_tokens=20,
        temperature=0.0,  # For more consistent outputs
        top_p=1.0
    )
    action_label = response.choices[0].message.content
    # print("Action: " + action_label)

    # Validate and map to predefined labels
    valid_actions = {
        'Turn Left': ["turn left"],
        'Turn Right': ["turn right"],
        'Take Exit': ["exit"],
        'Go Straight': ["continue straight", "go straight"],
        'Accelerate': ["accelerate"],
        'Slow Down': ["slow down", "prepare to stop", "stop"]
    }
    detected_actions = []
    # Check each valid action and see if it's mentioned in the response
    for key, values in valid_actions.items():
        if any(val in action_label.lower() for val in values):
            detected_actions.append(key)

    return detected_actions if detected_actions else ['Unknown']


def generate_dataset(api_key, num_samples=1000):
    data = []
    for _ in tqdm(range(num_samples), desc="Generating dataset"):
        instruction = generate_instruction(api_key)
        action = generate_action_label(instruction, api_key)
        data.append((instruction, action))
    return pd.DataFrame(data, columns=['instruction', 'action'])


api_key = os.getenv("OPENAI_API_KEY")
start_time = time.time()
if api_key:
    dataset = generate_dataset(api_key)
    print(dataset)
    dataset.to_csv('dataset_1000.csv', index=False)
    print("Dataset generated and saved to 'dataset.csv'")
else:
    print("API key not found. Please check your .env file.")

duration = time.time() - start_time
print(f"Generated {len(dataset)} instructions in {duration:.2f} seconds.")
