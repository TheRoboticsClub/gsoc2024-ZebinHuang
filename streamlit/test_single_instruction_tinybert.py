import json

import torch
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
import streamlit as st


def test_instruction(instruction, tokenizer_name, model_path, label_mapping_path):

    instruction = instruction
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    # Tokenize the instruction
    encodings = tokenizer(instruction, truncation=True, padding=True, max_length=128, return_tensors='pt')

    # Load label mapping from the provided path
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    label_mapping = {int(v): k for k, v in label_mapping.items()}

    # Ensure the model and inputs are on the same device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    encodings = {key: val.to(device) for key, val in encodings.items()}

    # Dynamically determine the number of labels based on the label mapping
    num_labels = len(label_mapping)
    model = BertForSequenceClassification.from_pretrained(tokenizer_name, num_labels=num_labels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Load label mapping
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    label_mapping = {int(v): k for k, v in label_mapping.items()}

    model.eval()

    with torch.no_grad():
        outputs = model(**encodings)
        pred = torch.argmax(outputs.logits, axis=1).item()

    predicted_action = label_mapping[pred]
    st.json({
        "Instruction": instruction,
        "Predicted Action": predicted_action
    })
