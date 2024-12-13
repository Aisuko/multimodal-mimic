# Dataset

> Note: This dataset is provided to facilitate code execution and result replication; it is not intended for data sharing. I have really bad experience on pre-processing the datset.

The pre-processed dataset:

|Dataset|Size|URL|
|---|---|---|
|aisuko/in-hospital-mortality-6-to-48|173MB|https://huggingface.co/datasets/aisuko/in-hospital-mortality-6-to-48|


## MIMIC-III

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



After adding the files, the structure of your MIMIC-III Benchmark folder should look like this:

> ignore the validation

```
mimic3-benchmarks
├── in-hospital-mortality
│   ├── train_listfile.csv
│   ├── test_listfile.csv
│   ├── train
│   ├── test
│   └── 
├── root
│   ├── 
│   └── 
```
