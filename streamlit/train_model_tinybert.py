import os
import json
import ast
from datetime import datetime
from io import BytesIO

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from fpdf import FPDF
import streamlit as st
import matplotlib.pyplot as plt


COLUMNS_TO_EXTRACT = [
        'loss', 'grad_norm', 'learning_rate', 'train_runtime',
        'train_samples_per_second', 'train_steps_per_second',
        'total_flos', 'train_loss'
    ]


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(labels, pred)
    return {
        'accuracy': accuracy
    }


def safe_literal_eval(val):
    """
    Safely evaluate a string literal.
    """
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


def prepare_data(file_path):
    """
    Prepare the dataset by encoding labels.
    """
    data = pd.read_csv(file_path)
    data['action'] = data['action'].apply(safe_literal_eval)
    data['action_label'] = data['action'].apply(lambda x: x[0])
    labels, uniques = pd.factorize(data['action_label'])
    return data['instruction'], labels, uniques


class ActionDataset(Dataset):
    """
    PyTorch Dataset for action instructions.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx], dtype=torch.long) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


class ProgressBarCallback(TrainerCallback):
    def __init__(self, progress_bar):
        self.progress_bar = progress_bar

    def on_train_begin(self, args, state, control, **kwargs):
        self.total_steps = state.max_steps

    def on_step_end(self, args, state, control, **kwargs):
        self.progress_bar.progress(state.global_step / self.total_steps)


class LoggingCallback(TrainerCallback):
    def __init__(self):
        self.log_data = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            self.log_data.append(logs)


logging_callback = LoggingCallback()
# progress_bar_callback = ProgressBarCallback()


def save_logs(logging_dir, log_df, total_steps, num_train_epochs, cls_report):
    now = datetime.now()
    time_str = now.strftime("%Y%m%d%H%M")
    dir = f'{logging_dir}/log_{time_str}'
    os.makedirs(dir)
    st.info(f'log saved in {dir}')

    train_df = log_df[["loss", "grad_norm", "learning_rate", "epoch"]]
    train_df.dropna(how="any", axis=0, inplace=True)
    eval_df = log_df[
        [
            "eval_loss",
            "eval_accuracy",
            "eval_runtime",
            "eval_samples_per_second",
            "eval_steps_per_second",
        ]
    ]
    eval_df.dropna(how="any", axis=0, inplace=True)
    train_df_ = log_df[
        [
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "total_flos",
            "train_loss",
        ]
    ]
    train_df_ = train_df_.fillna(method="ffill").fillna(method="bfill")

    log_df.to_excel(f'{dir}/raw.xlsx', index=False)
    train_df.to_excel(f'{dir}/train_df.xlsx', index=False)
    eval_df.to_excel(f'{dir}/eval_df.xlsx', index=False)
    train_df_.to_excel(f'{dir}/train_df_.xlsx', index=False)

    train_log = {
        "global_step": total_steps,
        "train_loss": (
            train_df_["train_loss"].iloc[-1] if "train_loss" in train_df_.columns else None
        ),
        "metrics": {
            "train_runtime": (
                train_df_["train_runtime"].iloc[-1]
                if "train_runtime" in train_df_.columns
                else None
            ),
            "train_samples_per_second": (
                train_df_["train_samples_per_second"].iloc[-1]
                if "train_samples_per_second" in train_df_.columns
                else None
            ),
            "train_steps_per_second": (
                train_df_["train_steps_per_second"].iloc[-1]
                if "train_steps_per_second" in train_df_.columns
                else None
            ),
            "total_flos": (
                train_df_["total_flos"].iloc[-1]
                if "total_flos" in train_df_.columns
                else None
            ),
            "train_loss": (
                train_df_["train_loss"].iloc[-1]
                if "train_loss" in train_df_.columns
                else None
            ),
            "epoch": num_train_epochs,
        },
    }

    eval_log = {
        "eval_loss": (
            eval_df["eval_loss"].iloc[-1] if "eval_loss" in eval_df.columns else None
        ),
        "eval_runtime": (
            eval_df["eval_runtime"].iloc[-1] if "eval_runtime" in eval_df.columns else None
        ),
        "eval_samples_per_second": (
            eval_df["eval_samples_per_second"].iloc[-1]
            if "eval_samples_per_second" in eval_df.columns
            else None
        ),
        "eval_steps_per_second": (
            eval_df["eval_steps_per_second"].iloc[-1]
            if "eval_steps_per_second" in eval_df.columns
            else None
        ),
        "epoch": num_train_epochs,
    }
    logs = {"train_log": train_log, "eval_log": eval_log, "cls_report": cls_report}
    with open(f"{dir}/log.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(logs, ensure_ascii=False, indent=4, separators=(",", ":")))
    return train_log, eval_log


def model_train_tinybert(file_path, model_name, output_dir, logging_dir, num_train_epochs,
                         train_batch_size, eval_batch_size, progress_bar=None):
    os.makedirs(logging_dir, exist_ok=True)

    # Prepare data
    instructions, labels, uniques = prepare_data(file_path)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    encodings = tokenizer(list(instructions), truncation=True, padding=True, max_length=128)

    # Save the label mapping
    label_mapping = {label: idx for idx, label in enumerate(uniques)}
    label_mapping_path = os.path.join(output_dir, 'label_mapping.json')
    label_mapping_path = os.path.normpath(label_mapping_path)
    with open(label_mapping_path, 'w') as f:
        json.dump(label_mapping, f)

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
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(uniques))

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=logging_dir,
        logging_steps=1,
        log_level='info',
    )
    total_steps = (
        len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs
    )
    progress_bar_callback = ProgressBarCallback(progress_bar)
    logging_callback = LoggingCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[logging_callback, progress_bar_callback],
        compute_metrics=compute_metrics
    )

    _ = trainer.train()

    # Evaluate the model
    _ = trainer.evaluate()

    # Predict and report
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    cls_report = classification_report(val_labels, pred_labels, zero_division=0)

    # Save the model
    model_save_path = os.path.join(output_dir, 'tinybert_model.pt')
    torch.save(model.state_dict(), model_save_path)

    # Train log
    log_df = pd.DataFrame(logging_callback.log_data)
    train_log, eval_log = save_logs(logging_dir, log_df, total_steps, num_train_epochs, cls_report)

    return train_log, eval_log, cls_report


def plot_combined_train_logs(train_df, train_df_):

    combined_df = pd.concat([train_df, train_df_], axis=0)

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    columns_to_plot = [
        "loss",
        "grad_norm",
        "learning_rate",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
        "total_flos",
        "train_loss",
    ]

    line_color = '#FFD700'
    border_width = 2
    tick_width = 2

    for i, col in enumerate(columns_to_plot):
        row = i // 4
        col_idx = i % 4

        axs[row, col_idx].plot(combined_df.index, combined_df[col], color=line_color)
        axs[row, col_idx].set_title(col, fontsize=16)
        axs[row, col_idx].tick_params(axis='both', which='major', labelsize=12, width=tick_width)

        axs[row, col_idx].grid(True, which='both', linestyle='--', linewidth=0.5)
        axs[row, col_idx].minorticks_on()

        axs[row, col_idx].spines['top'].set_linewidth(border_width)
        axs[row, col_idx].spines['right'].set_linewidth(border_width)
        axs[row, col_idx].spines['bottom'].set_linewidth(border_width)
        axs[row, col_idx].spines['left'].set_linewidth(border_width)

    fig.tight_layout()
    return fig


def plot_eval_logs(eval_df):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    columns_to_plot = ["eval_loss", "eval_accuracy", "eval_runtime", "eval_samples_per_second"]

    for i, col in enumerate(columns_to_plot):
        axs[i].plot(eval_df.index, eval_df[col], color='#FFD700')
        axs[i].set_title(col, fontsize=16)
        axs[i].tick_params(axis='both', which='major', labelsize=12, width=2)
        axs[i].spines['top'].set_linewidth(2)
        axs[i].spines['right'].set_linewidth(2)
        axs[i].spines['bottom'].set_linewidth(2)
        axs[i].spines['left'].set_linewidth(2)
        axs[i].grid(True, which='both', linestyle='--', linewidth=0.5)
        axs[i].minorticks_on()

    fig.tight_layout()
    return fig


def save_plots_and_logs_to_pdf(train_log_md, eval_log_md, cls_report_md, train_fig, eval_fig, pdf_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Convert markdown to plain text for PDF
    def add_markdown_to_pdf(pdf, markdown_text):
        for line in markdown_text.split('\n'):
            pdf.multi_cell(0, 10, line)

    pdf.add_page()
    pdf.set_font("Arial", size=12)
    add_markdown_to_pdf(pdf, train_log_md)

    pdf.add_page()
    add_markdown_to_pdf(pdf, eval_log_md)

    pdf.add_page()
    add_markdown_to_pdf(pdf, cls_report_md)

    # Save the figures to a buffer
    with BytesIO() as train_buf, BytesIO() as eval_buf:
        train_fig.savefig(train_buf, format='png')
        eval_fig.savefig(eval_buf, format='png')
        train_buf.seek(0)
        eval_buf.seek(0)

        pdf.add_page()
        pdf.image(train_buf, x=10, y=20, w=180)

        pdf.add_page()
        pdf.image(eval_buf, x=10, y=20, w=180)

    pdf.output(pdf_path)


def generate_markdown(train_log, eval_log, cls_report):
    train_log_md = f"""
    ## Train Log

    - Global Step: {train_log["global_step"]}
    - Train Loss: {train_log["train_loss"]}
    - Train Runtime: {train_log["metrics"]["train_runtime"]}
    - Train Samples per Second: {train_log["metrics"]["train_samples_per_second"]}
    - Train Steps per Second: {train_log["metrics"]["train_steps_per_second"]}
    - Total FLOPs: {train_log["metrics"]["total_flos"]}
    - Epoch: {train_log["metrics"]["epoch"]}
    """

    eval_log_md = f"""
    ## Eval Log

    - Eval Loss: {eval_log["eval_loss"]}
    - Eval Runtime: {eval_log["eval_runtime"]}
    - Eval Samples per Second: {eval_log["eval_samples_per_second"]}
    - Eval Steps per Second: {eval_log["eval_steps_per_second"]}
    - Epoch: {eval_log["epoch"]}
    """

    cls_report_md = f"""
    ## Classification Report

        \n{cls_report}\n
    """
    return train_log_md, eval_log_md, cls_report_md


if __name__ == '__main__':
    import sys
    sys.path.append('data_parsing')
    default_params = {
        "file_path": "datasets/user_instructions/dataset1234.csv",
        "model_name": "huawei-noah/TinyBERT_General_4L_312D",
        "output_dir": "./models",
        "logging_dir": "./logs",
        "num_train_epochs": 2,
        "train_batch_size": 1,
        "eval_batch_size": 1
    }
    model_train_tinybert(**default_params)
