import openai
from dotenv import load_dotenv
import os
import re
from settings import VALID_ACTIONS, OPENAI_PARAMS

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


def validate_api_key(api_key):
    """
    Validate the OpenAI API key by request.
    """
    openai.api_key = api_key
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a highly knowledgeable virtual assistant."},
                {"role": "user", "content": "Say hello in one sentence."}
            ],
            max_tokens=10
        )
        return True
    except openai.APIError as e:
        print(f"OpenAI API returned an API Error: {e}")


# Validate the API key
if not validate_api_key(openai.api_key):
    print("Invalid API key. Please check your .env file.")
    exit(1)


# def generate_instruction_prompt(action):
#     """
#     Generate a prompt for the OpenAI API to create a driving instruction for a given action.
#     """
#     return f"""
#     Generate a short driving instruction that includes the action '{action}'. Here are three examples:
#     1. "Turn right at the next intersection."
#     2. "Go straight past the traffic light."
#     3. "Merge into the left lane."
#     Generate an instruction that includes '{action}':
#     """


def generate_instruction_prompt(action, batch_size):
    """
    Generate a prompt for the OpenAI API to create a driving instruction for a given action.
    """
    return f"""
    Generate a short driving instruction that includes the action '{action}'.
    Make sure each instruction is distinct and uses different wording or context. Here are some examples:
    "Turn right at the next intersection."
    "Go straight past the traffic light."
    "Merge into the left lane."
    "At the roundabout, take the second exit."
    "Keep left to stay on the main road."
    There does not need to be any numerical numbering or any prefixes.
    Generate {batch_size} unique instruction.
    Use vocabulary with a meaning similar to '{action}', but not the same word:
    """


def fetch_instruction(action, batch_size=1):
    """
    Fetch a single driving instruction for the given action.
    """
    if action not in VALID_ACTIONS:
        print(f"Warning: The action '{action}' is not in the list of valid actions.")

    prompt = [
        {"role": "system", "content": "You are a highly knowledgeable virtual assistant."},
        {"role": "user", "content": generate_instruction_prompt(action, batch_size)}
    ]
    response = openai.chat.completions.create(
        model=OPENAI_PARAMS["model"],
        messages=prompt,
        max_tokens=OPENAI_PARAMS["max_tokens"],
        temperature=OPENAI_PARAMS["temperature"],
        top_p=OPENAI_PARAMS["top_p"],
        # n=n
    )
    instruction = response.choices[0].message.content
    return instruction


def generate_instruction(action):
    """
    Generate a driving instruction for the given action.
    """
    return fetch_instruction(action)


def generate_instructions_batch(action, batch_size):
    """
    Generate a batch of driving instructions for a given action using the OpenAI API.
    """
    raw_instruction = fetch_instruction(action, batch_size)
    # Split multiple instructions from a single response
    instructions = raw_instruction.split('\n')
    # Remove leading numbers and punctuation from each instruction
    instructions_list = [
        re.sub(r'^[\d\W]+', '', instruction).strip() for instruction in instructions
    ]
    return instructions_list
