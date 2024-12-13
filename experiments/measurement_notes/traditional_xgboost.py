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

from llm_argparser import parser
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from multimodal_clinical_pretraining.data.utils import ShuffleTransform

from torchmimic.data import IHMDataset
from torchmimic.data.preprocessing import Normalizer
from torchmimic.utils import pad_colalte

def train(args, train_dataloader, test_dataloader):

    all_values=[]
    all_labels=[]


    for values, lables, _, _ in train_dataloader:
        all_values.append(values.cpu().numpy())
        all_labels.append(lables.cpu().numpy())

    X_train=np.concatenate(all_values, axis=0)
    y_train=np.concatenate(all_labels, axis=0)

    n_samples, dim1, dim2 = X_train.shape
    X_train_reshaped = X_train.reshape(n_samples, dim1 * dim2)
    print(X_train_reshaped.shape)

    dtrain=xgb.DMatrix(X_train_reshaped, label=y_train)

    params={
        'tree_method': 'hist',  # Use GPU for training
        'device': 'cuda',
        'objective': 'binary:logistic',  # Specify the learning task
        # Add other parameters as needed
    }
    # Train the model
    xgb_model = xgb.train(params, dtrain)

    test(xgb_model, test_dataloader)



def test(xgb_model, test_dataloader):
    test_data=[]
    test_labels=[]

    for values, labels,_,_ in test_dataloader:
        test_data.append(values.cpu().numpy())
        test_labels.append(labels.cpu().numpy())

    X_test=np.concatenate(test_data, axis=0)
    y_test=np.concatenate(test_labels, axis=0)

    n_samples, dim1, dim2 = X_test.shape
    X_test_reshaped = X_test.reshape(n_samples, dim1 * dim2)
    print(X_test_reshaped.shape)

    dtest=xgb.DMatrix(X_test_reshaped, label=y_test)

    # Predict using the trained model
    y_pred_prob = xgb_model.predict(dtest)
    y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc=roc_auc_score(y_test, y_pred_prob)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}")




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
    train_listfile = listfile + "train_listfile.csv"
    test_listfile = "test_listfile.csv"

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

    output = train(
        args,
        train_dataloader,
        # val_dataloader,
        test_dataloader,
    )
