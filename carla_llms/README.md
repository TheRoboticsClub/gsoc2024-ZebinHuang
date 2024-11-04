## Setup

Follow setup instructions in the [project repository](https://github.com/TheRoboticsClub/gsoc2023-Meiqi_Zhao).

## Usage

### Data Collection
Example usage:
```bash
cd carla_llms
mkdir -p data  # create `./data` directory first
python data_collector_v4.py --dataset_path ./data --episode_file test_suites/Town01_All.txt --n_episodes 30
```
This command will randomly sample 30 episodes from `Town01_All.txt` and save the hdf5 files in `./data`.

### Training

Place the collected data files into the following directories for training and validation:
- Training data: `./data/Town01/train`
- Validation data: `./data/Town01/val`

```bash
cd carla_llms
python train_v2.py
```

### Testing
There are two test cases available: `Town02_case1.txt` and `Town02_case2.txt`. Example usage:
```bash
cd carla_llms
python evaluate_model_v3.py --episode_file test_suites/Town02_case1.txt --model "ModifiedDeepestLSTMTinyPilotNet/models/v1.0.pth" --n_episodes 1 --combined_control
```
