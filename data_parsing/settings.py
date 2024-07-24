# Define valid actions
VALID_ACTIONS = [
    "Right",
    "Left",
    "Straight",
    "LaneFollow"
]

# Define OpenAI API parameters
OPENAI_PARAMS = {
    # You can use gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, and gpt-3.5-turbo
    "model": "gpt-3.5-turbo",
    "max_tokens": 1000,
    "temperature": 1.0,
    "top_p": 0.95
}
