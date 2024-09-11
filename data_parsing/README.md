## Getting Started

### Prerequisites

Log in to your OpenAI account and navigate to the "[View API keys](https://beta.openai.com/account/api-keys)" section > "Create new secret key."

Ensure you have a `.env` file in data_parsing containing your OpenAI API key:

```plaintext
OPENAI_API_KEY=your_api_key_here
```

To install the necessary dependencies, please follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/TheRoboticsClub/gsoc2024-ZebinHuang.git
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   cd data_parsing
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Dataset generation

We provide three different methods for data generation. It is recommended to use `dataset_generator_batch_optimized.py`.

You can also use the following methods, but their performance will be significantly worse. They are retained to match the experiments in the blogs:
- `dataset_generator_batch.py`
- `dataset_generator_iter.py`

Generate a dataset with 4000 samples:

```bash
python dataset_generator_batch_optimized.py --actions Straight Right LaneFollow Left --num_samples 1000 --output_file ./datasets/user_instructions/dataset_4000.csv --max_batch_size 100
```

- actions: List of actions to generate instructions for. The default is VALID_ACTIONS.
- num_samples: Number of unique samples per action. The default is 25.
- output_file: Output CSV file to save the dataset. The default is dataset.csv.
- max_batch_size: Maximum batch size for generating instructions. The default is 50.

### Instruction analysis
Analyze the instructions from the generated datasets:

```bash
python instruction_analysis.py --file_path ./datasets/user_instructions/dataset_4000.csv --output_dir ./results
```

### Model training and testing

1. Train a model using BERT:
    ```bash
    python train_model.py --file_path ./datasets/user_instructions/dataset_4000.csv --output_dir ./models
    ```

2. Train a model using TinyBERT:
    ```bash
    python train_model_tinybert.py --file_path ./datasets/user_instructions/dataset_4000.csv --output_dir ./models
    ```

- file_path: Path to the CSV dataset file. This argument is required.
- model_name: Model name or path. The default is `bert-base-uncased`.
- output_dir: Directory to save the model and results. The default is `./results`.
- logging_dir: Directory to save the logs. The default is `./logs`.
- num_train_epochs: Number of training epochs. The default is 3.
- train_batch_size: Training batch size. The default is 8.
- eval_batch_size: Evaluation batch size. The default is 16.

3. Test a single instruction using a trained TinyBERT model:
    ```bash
    python test_single_instruction_tinybert.py --instruction "Continue straight on the highway for the next 10 miles." --model_path ./models/checkpoint-1000 --label_mapping_path ./models/label_mapping.csv
    ```

4. Test with a file containing an instruction:

    ```bash
    python test_instruction_file.py --file_path ./datasets/translated_test_suites/Town02_All.txt --model_path ./models/checkpoint-1000 --tokenizer_name huawei-noah/TinyBERT_General_4L_312D --label_mapping_path ./models/label_mapping.json
    ```

### Configuration translation

Translate configuration data:

```bash
python config_translator.py ./datasets/test_suites/Town01_All.txt ./translated_test_suite
```
