import torch
from transformers import BertTokenizer, BertForSequenceClassification
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Test a trained TinyBERT model for a single instruction.")
    parser.add_argument('--instruction', type=str, required=True, help="The instruction to classify.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model directory.")
    parser.add_argument('--tokenizer_name', type=str, default='huawei-noah/TinyBERT_General_4L_312D', help="Tokenizer name or path.")
    parser.add_argument('--label_mapping_path', type=str, required=True, help="Path to the label mapping JSON file.")
    args = parser.parse_args()

    instruction = args.instruction
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)

    # Tokenize the instruction
    encodings = tokenizer(instruction, truncation=True, padding=True, max_length=128, return_tensors='pt')

    # Load the model
    model = BertForSequenceClassification.from_pretrained(args.model_path)

    # Ensure the model and inputs are on the same device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    encodings = {key: val.to(device) for key, val in encodings.items()}

    # Load label mapping
    with open(args.label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    label_mapping = {int(v): k for k, v in label_mapping.items()}

    model.eval()

    with torch.no_grad():
        outputs = model(**encodings)
        pred = torch.argmax(outputs.logits, axis=1).item()

    predicted_action = label_mapping[pred]
    print(f"Instruction: {instruction}")
    print(f"Predicted Action: {predicted_action}")


if __name__ == '__main__':
    main()
