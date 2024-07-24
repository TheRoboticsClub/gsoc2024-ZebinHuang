import ast
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import argparse
import os
import json


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a TinyBERT model for action classification.")
    parser.add_argument('--file_path', type=str, required=True, help="Path to the CSV dataset file.")
    parser.add_argument('--model_name', type=str, default='huawei-noah/TinyBERT_General_4L_312D', help="Model name or path.")
    parser.add_argument('--output_dir', type=str, default='./models', help="Directory to save the model and results.")
    parser.add_argument('--logging_dir', type=str, default='./logs', help="Directory to save the logs.")
    parser.add_argument('--num_train_epochs', type=int, default=3, help="Number of training epochs.")
    parser.add_argument('--train_batch_size', type=int, default=8, help="Training batch size.")
    parser.add_argument('--eval_batch_size', type=int, default=16, help="Evaluation batch size.")
    return parser.parse_args()


def safe_literal_eval(val):
    """
    Safely evaluate a string literal.
    """
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


class ActionDataset(Dataset):
    """
    PyTorch Dataset for action instructions.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def prepare_data(file_path):
    """
    Prepare the dataset by encoding labels.
    """
    data = pd.read_csv(file_path)
    data['action'] = data['action'].apply(safe_literal_eval)
    data['action_label'] = data['action'].apply(lambda x: x[0])
    labels, uniques = pd.factorize(data['action_label'])
    return data['instruction'], labels, uniques


def main():
    args = parse_args()

    # Prepare data
    instructions, labels, uniques = prepare_data(args.file_path)
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    encodings = tokenizer(list(instructions), truncation=True, padding=True, max_length=128)

    # Save the label mapping
    label_mapping = {label: idx for idx, label in enumerate(uniques)}
    label_mapping_path = os.path.join(args.output_dir, 'label_mapping.json')
    with open(label_mapping_path, 'w') as f:
        json.dump(label_mapping, f)
    print(f"Label mapping saved to {label_mapping_path}")

    # Split data into training and validation sets
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        encodings['input_ids'], labels, test_size=0.1, random_state=42
    )
    train_masks, val_masks = train_test_split(
        encodings['attention_mask'], test_size=0.1, random_state=42
    )

    train_encodings = {'input_ids': train_inputs, 'attention_mask': train_masks}
    val_encodings = {'input_ids': val_inputs, 'attention_mask': val_masks}

    train_dataset = ActionDataset(train_encodings, train_labels)
    val_dataset = ActionDataset(val_encodings, val_labels)

    # Load model
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=len(uniques))

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=args.logging_dir,
        logging_steps=10,
    )

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Predict and report
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    print(classification_report(val_labels, pred_labels, zero_division=0))

    # Save the model
    model_save_path = os.path.join(args.output_dir, 'tinybert_model.pt')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == '__main__':
    main()
