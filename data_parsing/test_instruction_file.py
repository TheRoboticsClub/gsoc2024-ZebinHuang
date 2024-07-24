import torch
from transformers import BertTokenizer, BertForSequenceClassification
import argparse
import json
import os


def load_instructions(file_path):
    """
    Load instructions from the provided file.
    """
    instructions = []
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('" "')
            if len(parts) == 2:
                instruction1 = parts[0].strip('"')
                instruction2 = parts[1].strip('"')
                instructions.append(instruction1)
                instructions.append(instruction2)
                lines.append(line.strip())
    return instructions, lines


def classify_instructions(instructions, model, tokenizer, label_mapping, device):
    """
    Classify the list of instructions using the provided model and tokenizer.
    """
    results = []
    encodings = tokenizer(instructions, truncation=True, padding=True, max_length=128, return_tensors='pt')
    encodings = {key: val.to(device) for key, val in encodings.items()}

    model.eval()

    with torch.no_grad():
        outputs = model(**encodings)
        preds = torch.argmax(outputs.logits, axis=1).tolist()

    for instruction, pred in zip(instructions, preds):
        predicted_action = label_mapping[pred]
        results.append(predicted_action)

    return results


def save_predictions(lines, predictions, output_file):
    """
    Save the original lines with predictions appended.
    """
    with open(output_file, 'w') as file:
        pred_index = 0
        for line in lines:
            parts = line.strip().split('" "')
            if len(parts) == 2:
                instruction1_pred = predictions[pred_index]
                instruction2_pred = predictions[pred_index + 1]
                pred_index += 2
                file.write(f'{line} "{instruction1_pred}" "{instruction2_pred}"\n')
    print(f"Predictions saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Test a trained TinyBERT model for a list of instructions.")
    parser.add_argument('--file_path', type=str, required=True, help="Path to the instruction file.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model directory.")
    parser.add_argument('--tokenizer_name', type=str, default='huawei-noah/TinyBERT_General_4L_312D', help="Tokenizer name or path.")
    parser.add_argument('--label_mapping_path', type=str, required=True, help="Path to the label mapping JSON file.")
    args = parser.parse_args()

    # Load the instruction file
    instructions, lines = load_instructions(args.file_path)

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)

    # Load the model
    model = BertForSequenceClassification.from_pretrained(args.model_path)

    # Ensure the model and inputs are on the same device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Load label mapping
    with open(args.label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    label_mapping = {int(v): k for k, v in label_mapping.items()}

    # Classify the instructions
    predictions = classify_instructions(instructions, model, tokenizer, label_mapping, device)

    # Save the results to a new text file
    output_file = os.path.splitext(args.file_path)[0] + "_pred.txt"
    save_predictions(lines, predictions, output_file)


if __name__ == '__main__':
    main()
