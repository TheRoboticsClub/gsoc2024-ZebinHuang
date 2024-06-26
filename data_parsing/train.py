import ast
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset


class ActionDataset(Dataset):
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
    data = pd.read_csv(file_path)
    data['action'] = data['action'].apply(ast.literal_eval)
    # Simplifying the problem by selecting the first action as the label
    data['action_label'] = data['action'].apply(lambda x: x[0])
    labels = pd.factorize(data['action_label'])[0]  # Convert labels to numerical format
    return data['instruction'], labels


# Load and prepare dataset
instructions, labels = prepare_data('dataset_1000.csv')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encode text
encodings = tokenizer(list(instructions), truncation=True, padding=True, max_length=128)

# Check the length to ensure they match
print(len(encodings['input_ids']), len(labels))
print("Labels length:", len(labels))
print("Input IDs length:", len(encodings['input_ids']))
print("Attention Mask length:", len(encodings['attention_mask']))

train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    encodings['input_ids'], labels, test_size=0.1, random_state=42)
train_masks, val_masks = train_test_split(
    encodings['attention_mask'], test_size=0.1, random_state=42)

train_encodings = {'input_ids': train_inputs, 'attention_mask': train_masks}
val_encodings = {'input_ids': val_inputs, 'attention_mask': val_masks}

train_dataset = ActionDataset(train_encodings, train_labels)
val_dataset = ActionDataset(val_encodings, val_labels)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(np.unique(labels)))

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

trainer.evaluate()

predictions = trainer.predict(val_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)

print(classification_report(val_dataset.labels, pred_labels))
