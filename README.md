# Multimodal Pretraining of Medical Time Series and Notes


## GPU environment

```bash
# 
sudo apt update && sudo apt upgrade
sudo apt install build-essential

# Installing Nvidia-smi
sudo apt install -y software-properties-common
sudo apt install ubuntu-drivers-common -y

# Detect recommended drivers
sudo ubuntu-drivers devices

# Install the driver(if you see the recommend GPU driver in there)
sudo ubuntu-drivers install

# Reboot
sudo reboot

# install nvcc
sudo apt install nvidia-cuda-toolkit -y
```

## Development environment

Let's lock down the Python version to 3.9.16. Otherwise will cause insence issues, like the library incompatibilities. 

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

echo "export PATH=~/miniconda3/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc
```

```bash
conda create -n mutimodal_mimic python=3.9.16 -y
```

```bash
conda init
```

```bash
conda activate mutimodal_mimic
```

```bash
# installing PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# install torchmimic
pip install 'torchmimic @ git+https://github.com/kingrc15/torchmimic.git@b3610fc98025ec42903e7646c715b4a5faeac403'
pip install pandas
pip install transformers
conda install -c conda-forge py-xgboost-gpu
conda install -c conda-forge statsmodels
```

```bash
# deactivate
conda deactivate
```

# GPU training
```
tmux new -s session_name
tmux ls
tmux a -t session_name
python experiments/measurement_notes/measurement_notes_downstream.py > train_log.txt 2>&1
Control+B D
htop
nvidia-smi
less train_log.txt
tail -f train_log.txt
```

## Data

### MIMIC-III

The dataset used for this paper is MIMIC-III. The data can be downloaded here [https://physionet.org/content/mimiciii/1.4/](https://physionet.org/content/mimiciii/1.4/). **NOTE**: To gain access to this dataset, you will need to complete the required training. 

### MIMIC-III Benchmark

Once you've downloaded the MIMIC-III dataset, you will need to build the MIMIC-III Benchmark from [https://github.com/YerevaNN/mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks). We used a modified version of this code so that we can index patient IDs and read in and out times without opening files. Replace:

```
mimic3-benchmarks/mimic3benchmark/scripts/extract_episodes_from_subjects.py
mimic3-benchmarks/mimic3benchmark/scripts/create_decompensation.py
mimic3-benchmarks/mimic3benchmark/scripts/create_in_hospital_mortality.py
mimic3-benchmarks/mimic3benchmark/scripts/create_length_of_stay.py
mimic3-benchmarks/mimic3benchmark/scripts/create_phenotyping.py
mimic3-benchmarks/mimic3benchmark/scripts/create_multitask.py
```

with:

```
multimodal-medical-pretraining/mimic3benchmark/extract_episodes_from_subjects.py
multimodal-medical-pretraining/mimic3benchmark/create_decompensation.py
multimodal-medical-pretraining/mimic3benchmark/create_in_hospital_mortality.py
multimodal-medical-pretraining/mimic3benchmark/create_length_of_stay.py
multimodal-medical-pretraining/mimic3benchmark/create_phenotyping.py
multimodal-medical-pretraining/mimic3benchmark/create_multitask.py
```

Once you've replaced that file, build the benchmarks as described here: [https://github.com/YerevaNN/mimic3-benchmarks/tree/master#building-the-benchmark](https://github.com/YerevaNN/mimic3-benchmarks/tree/master#building-the-benchmark).

For our semi-supervised experiments, we created new listfiles which can be downloaded [here](https://drive.google.com/drive/folders/1wB-4kUrNB9cHqD1qvR5fFEOaIUXmXTxI?usp=sharing). These listfiles need to be placed in the root directory of your MIMIC-III Benchmark data.

After adding the files, the structure of your MIMIC-III Benchmark folder should look like this:

Structure for ImageNet Data
```
mimic3-benchmarks
├── phenotyping
│   ├── 1percent_train_listfile.csv
│   ├── 10percent_train_listfile.csv
│   ├── 50percent_train_listfile.csv
│   ├── 1percent_val_listfile.csv
│   ├── 10percent_val_listfile.csv
│   ├── 50percent_val_listfile.csv
│   ├── train_listfile.csv
│   ├── val_listfile.csv
│   ├── test_listfile.csv
│   └── 
├── in-hospital-mortality
│   ├── 1percent_train_listfile.csv
│   ├── 10percent_train_listfile.csv
│   ├── 50percent_train_listfile.csv
│   ├── 1percent_val_listfile.csv
│   ├── 10percent_val_listfile.csv
│   ├── 50percent_val_listfile.csv
│   ├── train_listfile.csv
│   ├── val_listfile.csv
│   ├── test_listfile.csv
│   └── 
├── root
│   ├── 
│   └── 
```

### Requirements

The python version used to run our experiments is 3.9.16. Requirements can be found in the requirements.txt file. Install them by running:

`pip install -r requirements.txt`

## Pretraining

Pretraining can be run using the script located at `experiments/measurement_notes/measurement_notes_pretraining.py`. We've included an example command in `pretrain`

## Finetune

Pretraining can be run using the script located at `experiments/measurement_notes/measurement_notes_downstream.py`. We've included an example command in `finetuning`. This command requires an experiment and task be provided. Possible experiments include `semi_0_5_eval`, `semi_0_1_eval`, `semi_0_01_eval`, `full_eval`, and `linear_eval`. Possible tasks include: `IHM` and `Phenotyping`. A pretrained model can be provided using `pretrained_path`. An example of our finetuning experiment can be found at `linear_eval`. Download the model used for this evaluation [here](https://drive.google.com/drive/folders/1wB-4kUrNB9cHqD1qvR5fFEOaIUXmXTxI?usp=sharing)
