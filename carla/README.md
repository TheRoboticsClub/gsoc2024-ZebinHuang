## Setup Instructions

```bash
# Clone the Repository
git clone https://github.com/TheRoboticsClub/gsoc2023-Meiqi_Zhao.git
```
Follow the instructions in the repository to configure, install, and test the project.

### Download Additional Files
```bash
cd gsoc2023-Meiqi_Zhao/src
wget https://raw.githubusercontent.com/TheRoboticsClub/gsoc2024-ZebinHuang/carla/carla/evaluate_model_instructions.py

# Download the preprocessing and command scripts (Note: `-O` will overwrite the files if they already exist):
cd gsoc2023-Meiqi_Zhao/src/utils
wget -O preprocess.py https://raw.githubusercontent.com/TheRoboticsClub/gsoc2024-ZebinHuang/carla/carla/preprocess.py
wget -O high_level_command_instructions.py https://raw.githubusercontent.com/TheRoboticsClub/gsoc2024-ZebinHuang/carla/carla/high_level_command_instructions.py
```

### Testing

```bash
# Run CARLA in the installation path, run:
./CarlaUE4.sh

# Evaluate the model with the following command:
python evaluate_model_instructions.py --episode_file ./datasets/translated_test_suites/Town02_All.txt --model "ModifiedDeepestLSTMTinyPilotNet/models/v10.0.pth" --n_episodes 100 --combined_control --bert_model_path ./models/tinybert_model.pt --tokenizer_name huawei-noah/TinyBERT_General_4L_312D --label_mapping_path ./models/label_mapping.json
```
The datasets are from the data_parsing folder, and the model files have been uploaded to Hugging Face (see main README).
