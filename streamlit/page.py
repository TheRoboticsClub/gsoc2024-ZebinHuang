import base64
import io
import os
import time
from datetime import datetime
import json

import pandas as pd
from dataset_generator_batch_optimized import generate_dataset
from dotenv import load_dotenv, set_key
from instruction_analysis import analysis
from markdown import markdown
from settings import OPENAI_MODELS, VALID_ACTIONS
from test_instruction_file import test_file
from test_single_instruction_tinybert import test_instruction
from train_model_tinybert import (
    model_train_tinybert,
    plot_combined_train_logs,
    plot_eval_logs,
    generate_markdown,
)
from utils import convert_html_to_pdf, validate_api_key
import streamlit as st


def page1():
    """Page 1: Data generation"""

    # Using demo from https://github.com/streamlit/llm-examples/blob/main/Chatbot.py
    with st.sidebar:
        api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

        if st.button("Validate API Key"):
            if validate_api_key(api_key):
                st.success("API Key is valid!")
                set_key(".env", "OPENAI_API_KEY", api_key)  # Save the API key to .env file
            else:
                st.error("Invalid API Key. Please try again.")

        st.markdown(
            "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)",
            unsafe_allow_html=True,
        )
        st.markdown(
            "[View the source code](https://github.com/TheRoboticsClub/gsoc2024-ZebinHuang)",
            unsafe_allow_html=True,
        )
        st.markdown(
            "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/TheRoboticsClub/gsoc2024-ZebinHuang?quickstart=1)",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<h1 style='text-align: center;'>Data Generation</h1>", unsafe_allow_html=True
    )
    st.write("Working dir:", os.getcwd())

    container = st.container(border=True)

    col1, col2 = container.columns(2)

    with col1:
        model = st.selectbox("Model", options=OPENAI_MODELS, index=2)
        max_batch_size = st.number_input("Max batch size:", 0, 100, 50)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.9, step=0.05)

    with col2:
        actions = st.multiselect("Actions", options=VALID_ACTIONS)
        num_samples = st.number_input("Num samples", 0, 10000, 100)
        output_file = st.text_input("Output file", value="dataset.csv")

    start_button = st.button(
        "START", use_container_width=True
    )

    if start_button:
        progress_bar = st.progress(0)
        with st.spinner("Generating dataset..."):
            load_dotenv()

            # Comment for using input API key
            # api_key = st.secrets["openai_api_key"]
            if api_key:
                start_time = time.time()
                dataset = generate_dataset(
                    actions=actions,
                    num_samples_per_action=num_samples,
                    max_batch_size=max_batch_size,
                    progress_bar=progress_bar,
                    model=model,
                    temperature=temperature,
                )
                dataset.to_csv(f"./streamlit/datasets/user_instructions/{output_file}", index=False)
                duration = time.time() - start_time

                st.success(f"Dataset generated and saved to '{output_file}'")
                st.write(f"Generated {len(dataset)} instructions in {duration:.2f} seconds.")

                with st.expander("Generated dataset"):
                    st.dataframe(dataset)
            else:
                st.error("Please add your OpenAI API key to continue.")


def page2():
    """ Page 2: Data analysis """

    st.markdown(
        "<h1 style='text-align: center;'>Data Analysis</h1>", unsafe_allow_html=True
    )

    # Create container
    container = st.container(border=True)

    # Get all csv dataset files
    csv_folder = './streamlit/datasets/user_instructions'
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    with container:
        selected_file = st.selectbox("File path", csv_files)
        if selected_file:
            df = pd.read_csv(os.path.join(csv_folder, selected_file))
            st.markdown("<h2>Data</h2>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)
            analyze_button = st.button(
                "START",
                use_container_width=True,
            )
            if analyze_button:
                pie_chart_fig, wordcloud_fig = analysis(os.path.join(csv_folder, selected_file))
                col1, col2 = container.columns(2)
                with col1:
                    st.markdown("<h3>Word Cloud</h3>", unsafe_allow_html=True)
                    st.pyplot(wordcloud_fig)
                with col2:
                    st.markdown("<h3>Actions Distribution</h3>", unsafe_allow_html=True)
                    st.pyplot(pie_chart_fig)


def page3():
    """Page 3: Model Training"""

    st.markdown(
        "<h1 style='text-align: center;'>Model Training</h1>", unsafe_allow_html=True
    )
    csv_folder = './streamlit/datasets/user_instructions'
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    # Model and model selection
    model_name = st.text_input("Model name", value='huawei-noah/TinyBERT_General_4L_312D')
    mode = st.selectbox("Mode", options=['TinyBert'])

    # Batch sizes and epochs
    col1, col2 = st.columns(2)
    eval_bs = col1.number_input("Eval batch size", min_value=1, max_value=1024, value=16)
    train_bs = col2.number_input("Train batch size", min_value=1, max_value=1024, value=8)

    col1, col2 = st.columns(2)
    train_epochs = col1.number_input("Train epochs", min_value=1, max_value=100, value=3)
    file_path = col2.selectbox("Select dataset", csv_files)

    # Directories
    col1, col2 = st.columns(2)
    output_dir = col1.text_input("Output directory", value='./streamlit/models')
    logging_dir = col2.text_input("Logging directory", value='./streamlit/logs')

    # Start button
    if st.button("START", use_container_width=True):
        st.write("Starting training process...")
        progress_bar = st.progress(0)

        kwargs = {
            "file_path": os.path.join(csv_folder, file_path),
            "model_name": model_name,
            "output_dir": output_dir,
            "logging_dir": logging_dir,
            "num_train_epochs": train_epochs,
            "train_batch_size": train_bs,
            "eval_batch_size": eval_bs,
            "progress_bar": progress_bar,
        }

        if mode == "TinyBert":
            train_log, eval_log, cls_report = model_train_tinybert(**kwargs)
            st.success("Training completed successfully!")

            st.expander("Training Log").json(train_log)
            st.expander("Evaluation Log").json(eval_log)
            st.expander("Classification Report").text(cls_report)
        elif mode == 'Bert':
            st.warning("Bert mode is not yet implemented.")


def page4():
    """ Page 4: Log analysis """

    st.markdown("<h1 style='text-align: center;'>Check Logs</h1>", unsafe_allow_html=True)

    logs_folder = './streamlit/logs'
    subfolders = [f.name for f in os.scandir(logs_folder) if f.is_dir()]

    if not subfolders:
        st.error("There are no subfolders in the log folder.")
    else:
        # Sort the folder names in descending order to have the latest at the top
        subfolders.sort(reverse=True)

        selected_folder = st.selectbox("Select the log folder", subfolders)
        folder_path = os.path.join(logs_folder, selected_folder)

    if st.button("Load Log"):
        train_log_path = os.path.join(folder_path, "train_df.xlsx")
        train_log_path_ = os.path.join(folder_path, "train_df_.xlsx")
        eval_log_path = os.path.join(folder_path, "eval_df.xlsx")
        log_path = os.path.join(folder_path, "log.json")

        if (
            os.path.exists(train_log_path)
            and os.path.exists(train_log_path_)
            and os.path.exists(eval_log_path)
        ):
            # Read metrics data
            train_df = pd.read_excel(train_log_path)
            train_df_ = pd.read_excel(train_log_path_)
            eval_df = pd.read_excel(eval_log_path)

            # Get train and eval figures
            train_fig = plot_combined_train_logs(train_df, train_df_)
            eval_fig = plot_eval_logs(eval_df)

            # Markdown to html
            buffer1 = io.BytesIO()
            train_fig.savefig(buffer1, format='png')
            buffer1.seek(0)
            train_base64 = base64.b64encode(buffer1.read()).decode('utf-8')
            train_img_html = (
                f'<img src="data:image/png;base64,{train_base64}" '
                'style="width: 80%; max-width: 800px;"/>'
            )

            buffer2 = io.BytesIO()
            eval_fig.savefig(buffer2, format='png')
            buffer2.seek(0)
            eval_base64 = base64.b64encode(buffer2.read()).decode('utf-8')
            eval_img_html = (
                f'<img src="data:image/png;base64,{eval_base64}" '
                'style="width: 80%; max-width: 800px;"/>'
            )

            st.markdown("<h2>TrainLog</h2>", unsafe_allow_html=True)
            st.pyplot(train_fig)
            st.markdown("<h2>EvalLog</h2>", unsafe_allow_html=True)
            st.pyplot(eval_fig)

            # Load and Eval log
            with open(log_path, 'r', encoding='utf-8') as f:
                logs = json.loads(f.read())
            train_log, eval_log, cls_report = (
                logs['train_log'],
                logs['eval_log'],
                logs['cls_report'],
            )
            if train_log and eval_log and cls_report:
                train_log_md, eval_log_md, cls_report_md = generate_markdown(
                    train_log, eval_log, cls_report
                )
                train_html, eval_html, _ = (
                    markdown(train_log_md),
                    markdown(eval_log_md),
                    markdown(cls_report_md),
                )
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Markdown</title>
                    <style>
                        body {{
                            font-size: 14px;
                            line-height: 1;
                            margin: 10px;
                            font-family: Arial, sans-serif;
                        }}
                        img {{
                            display: block;
                            margin: 10px auto;
                        }}
                        .page-break {{
                            page-break-before: always;
                        }}
                    </style>
                </head>
                <body>
                    {train_html}
                    {train_img_html}
                    {eval_html}
                    {eval_img_html}

                </body>
                </html>
                """
                pdf_path = os.path.join(folder_path, "logs_report.pdf")
                convert_html_to_pdf(html_content, pdf_path)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "Download Log",
                        f,
                        file_name="logs_report.pdf",
                        mime="application/pdf",
                        key="download_pdf",
                    )
            else:
                st.error(
                    "No training logs, evaluation logs, or classification reports were found. "
                    "Please run page 3 first to generate these logs."
                )
        else:
            st.error("The log file in the specified path does not exist.")


def page5():
    """ Page 5: Model testing """

    st.markdown("<h1 style='text-align: center;'>Model Test</h1>", unsafe_allow_html=True)

    st.container(border=True)

    # Select test mode
    test_type = st.sidebar.radio("Select Test Type", ("Instruction", "File"))

    model_path = st.text_input("Model path", value="./streamlit/models/checkpoint-1000")
    tokenizer_name = st.text_input(
        "Tokenizer name", value="huawei-noah/TinyBERT_General_4L_312D"
    )
    label_mapping_path = st.text_input(
        "Label mapping path", value="./streamlit/models/label_mapping.json"
    )

    if test_type == "Instruction":
        instruction = st.text_input("Instruction")
        col_full = st.columns(1)
        with col_full[0]:
            test_button = st.button(
                "TEST", use_container_width=True
            )
        if test_button:
            test_instruction(instruction, tokenizer_name, model_path, label_mapping_path)

    elif test_type == "File":
        datetime.now()
        uploaded_file = st.file_uploader("Upload a user instruction file", type=['txt'])
        if uploaded_file:
            col_full = st.columns(1)
            with col_full[0]:
                test_button = st.button(
                    "TEST", use_container_width=True
                )
                if test_button:
                    save_path = test_file(
                        uploaded_file, tokenizer_name, model_path, label_mapping_path
                    )
                    st.success(f'The test was successful and the results were saved {save_path}')

                    with open(save_path, 'r') as file:
                        result_data = file.read()

                    # DownLoad result
                    st.download_button(
                        label="Download Result",
                        data=result_data,
                        file_name=os.path.basename(save_path),
                        mime="text/plain",
                    )
