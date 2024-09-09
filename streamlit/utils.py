import os
import re
import openai
from dotenv import load_dotenv
from xhtml2pdf import pisa
from settings import VALID_ACTIONS, OPENAI_PARAMS
import streamlit as st

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def validate_api_key(api_key):
    """
    Validate the OpenAI API key by request.
    """
    openai.api_key = api_key
    try:
        openai.chat.completions.create(
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


def fetch_instruction(action, batch_size=1, **custom_params):
    """
    Fetch a single driving instruction for the given action.
    """
    if action not in VALID_ACTIONS:
        print(f"Warning: The action '{action}' is not in the list of valid actions.")

    prompt = [
        {"role": "system", "content": "You are a highly knowledgeable virtual assistant."},
        {"role": "user", "content": generate_instruction_prompt(action, batch_size)}
    ]

    # Use OPENAI_PARAMS as the base parameters, and override with any custom parameters provided
    params = {**OPENAI_PARAMS, **custom_params}

    response = openai.chat.completions.create(
        model=params.get("model"),
        messages=prompt,
        max_tokens=params.get("max_tokens"),
        temperature=params.get("temperature"),
        top_p=params.get("top_p"),
        # n=params.get("n", 1)  # Use 1 as default if n is not provided
    )

    instruction = response.choices[0].message.content
    return instruction


def generate_instruction(action, **custom_params):
    """
    Generate a driving instruction for the given action.
    """
    return fetch_instruction(action, **custom_params)


def generate_instructions_batch(action, batch_size, **custom_params):
    """
    Generate a batch of driving instructions for a given action using the OpenAI API.
    """
    raw_instruction = fetch_instruction(action, batch_size, **custom_params)
    # Split multiple instructions from a single response
    instructions = raw_instruction.split('\n')
    # Remove leading numbers and punctuation from each instruction
    instructions_list = [
        re.sub(r'^[\d\W]+', '', instruction).strip() for instruction in instructions
    ]
    return instructions_list


def convert_html_to_pdf(source_html, output_filename):
    """
    Convert an HTML document to a PDF file using the Pisa library.
    """
    with open(output_filename, "w+b") as result_file:
        pisa_status = pisa.CreatePDF(source_html, dest=result_file)
        return pisa_status.err
