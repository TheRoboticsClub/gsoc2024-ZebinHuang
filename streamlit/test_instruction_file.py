import json
import io
import torch
from transformers import BertTokenizer, BertForSequenceClassification


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


def load_instructions_stream(file_stream):
    """
    Load instructions from the provided file stream.
    """
    instructions = []
    lines = []

    # Use io.TextIOWrapper to convert the byte stream to a text stream
    with io.TextIOWrapper(file_stream, encoding='utf-8') as text_stream:
        for line in text_stream:
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


def test_file(upload_file, tokenizer_name, model_path, label_mapping_path):
    # Load the instruction file
    instructions, lines = load_instructions_stream(upload_file)

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    # Load label mapping from the provided path
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    label_mapping = {int(v): k for k, v in label_mapping.items()}

    # Dynamically determine the number of labels based on the label mapping
    num_labels = len(label_mapping)
    model = BertForSequenceClassification.from_pretrained(tokenizer_name, num_labels=num_labels)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Classify the instructions
    predictions = classify_instructions(instructions, model, tokenizer, label_mapping, device)

    # Save the results to a new text file
    save_path = './streamlit/datasets/translated_test_suites/result.txt'
    save_predictions(lines, predictions, save_path)
    return save_path
