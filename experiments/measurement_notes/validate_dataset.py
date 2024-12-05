import os
import sys
import torch
import numpy as np
import xgboost as xgb
from torchvision import transforms
from torch.utils.data import DataLoader


currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, "../.."))

if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)

from downstream_argparser import parser
from multimodal_clinical_pretraining.data.utils import ShuffleTransform
from torchmimic.data import IHMDataset
from torchmimic.data.preprocessing import Normalizer
from torchmimic.utils import pad_colalte
import pandas as pd


import torch
import numpy as np
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import acf

def compute_snr(data):
    """
    Compute the Signal-to-Noise Ratio (SNR) for the dataset.
    Args:
        data (torch.Tensor): Input data tensor of shape (batch_size, seq_len, num_features).
    Returns:
        float: SNR value.
    """
    signal_power = torch.mean(data ** 2)
    noise_power = torch.var(data)
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()

def compute_feature_correlation(data):
    """
    Compute the average absolute Pearson correlation coefficient between features.
    Args:
        data (torch.Tensor): Input data tensor of shape (batch_size, seq_len, num_features).
    Returns:
        float: Average absolute correlation coefficient.
    """
    num_features = data.shape[-1]
    correlations = []
    for i in range(num_features):
        for j in range(i + 1, num_features):
            feature_i = data[:, :, i].flatten()
            feature_j = data[:, :, j].flatten()
            if torch.std(feature_i) > 0 and torch.std(feature_j) > 0:
                corr, _ = pearsonr(feature_i.cpu().numpy(), feature_j.cpu().numpy())
                correlations.append(abs(corr))
    avg_correlation = np.mean(correlations) if correlations else 0
    return avg_correlation

def compute_autocorrelation_pandas(data, lag=1):
    """
    Compute the autocorrelation using Pandas.
    Args:
        data (torch.Tensor): Input data tensor of shape (batch_size, seq_len, num_features).
        lag (int): The lag at which to compute the autocorrelation.
    Returns:
        float: Average autocorrelation value.
    """
    num_features = data.shape[-1]
    autocorrelations = []
    for i in range(num_features):
        feature_data = data[:, :, i].flatten().cpu().numpy()
        series = pd.Series(feature_data)
        autocorr = series.autocorr(lag=lag)
        if pd.notna(autocorr):
            autocorrelations.append(autocorr)
    avg_autocorrelation = np.mean(autocorrelations) if autocorrelations else float('nan')
    return avg_autocorrelation

def analyze_dataloader(dataloader):
    """
    Analyze the DataLoader to compute SNR, feature correlation, and autocorrelation.
    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader object.
    Returns:
        dict: Dictionary containing SNR, average feature correlation, and average autocorrelation.
    """
    all_data = []
    for batch in dataloader:
        x, _, _, _ = batch  # Assuming batch is a tuple (x, y, lens, mask)
        all_data.append(x)
    all_data = torch.cat(all_data, dim=0)
    
    snr = compute_snr(all_data)
    feature_corr = compute_feature_correlation(all_data)
    autocorr = compute_autocorrelation_pandas(all_data)
    
    return {
        'SNR': snr,
        'Average Feature Correlation': feature_corr,
        'Average Autocorrelation': autocorr
    }



if __name__ == "__main__":
    args=parser()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = True

    listfile=""

    train_measurement_transform = transforms.Compose(
        [
            ShuffleTransform(args.measurement_max_seq_len),
        ]
    )

    test_measurement_transform = transforms.Compose(
        [
            ShuffleTransform(args.measurement_max_seq_len),
        ]
    )

    root = os.path.join(args.mimic_benchmark_root, "in-hospital-mortality")
    train_listfile = listfile + "1percent_train_listfile.csv"
    val_listfile = "1percent_val_listfile.csv"
    test_listfile = "1percent_test_listfile.csv"

    train_dataset = IHMDataset(
        root, customListFile=os.path.join(root, train_listfile), train=True
    )
    test_dataset = IHMDataset(
        root, customListFile=os.path.join(root, test_listfile), train=False
    )

    discretizer_header = train_dataset.discretizer.transform(
        train_dataset.reader.read_example(0)["X"]
    )[1].split(",")
    cont_channels = [
        i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1
    ]

    train_dataset.normalizer = Normalizer(fields=cont_channels)
    train_dataset.normalizer.load_params(
        "./multimodal_clinical_pretraining/resources/normalizer_params"
    )

    test_dataset.normalizer = Normalizer(fields=cont_channels)
    test_dataset.normalizer.load_params(
        "./multimodal_clinical_pretraining/resources/normalizer_params"
    )
    print(f"Length of training dataset = {len(train_dataset)}")
    print(f"Length of test dataset = {len(test_dataset)}")

    args.n = len(train_dataset)
    args.n_features = 76

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=pad_colalte,
        pin_memory=True,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        num_workers=0,
        collate_fn=pad_colalte,
        pin_memory=True,
        shuffle=False,
    )
    args.use_measurements = True
    args.use_notes = False

    if args.task == "IHM":
        args.n_classes = 1


    torch.manual_seed(42)
    np.random.seed(42)

    results=analyze_dataloader(train_dataloader)
    print("Data Results:", results)