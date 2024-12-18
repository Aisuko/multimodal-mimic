import os
import sys
import warnings

import numpy as np

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms

from torchmimic.utils import pad_colalte
from torchmimic.data.preprocessing import Normalizer
from torchmimic.data import IHMDataset
from torchmimic.loggers import IHMLogger

from llm_argparser import parser

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, "../.."))

if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)


warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from multimodal_clinical_pretraining.data.utils import ShuffleTransform
from multimodal_clinical_pretraining.utils import load_pretrained_model
from multimodal_clinical_pretraining.scheduler import create_scheduler
from multimodal_clinical_pretraining.models import create_model

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class ClinicalMultiModal(nn.Module):
    def __init__(self, base_model, measurement_emb_size, n_classes):
        super(ClinicalMultiModal, self).__init__()
        self.base_model=base_model
        self.classifier=nn.Linear(measurement_emb_size, n_classes)
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()
    def forward(self,x):
        features=self.base_model(x)
        logits=self.classifier(features[:,0,:])
        return logits
    

def evaluate_model(model, dataloader, device, criterion, logger, epoch, threshold=0.5, task="IHM", use_measurements=True):
    """
    Evaluate the model on the provided DataLoader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to perform evaluation on.
        criterion (torch.nn.Module): Loss function.
        logger (Logger): Logger to track metrics.
        threshold (float, optional): Threshold for binary classification. Defaults to 0.5.
        task (str, optional): Specific task identifier. Defaults to "IHM".
        use_measurements (bool, optional): Flag to use measurement data. Defaults to True.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    model.eval()
    ys = []
    preds = []

    with torch.no_grad():
        for values, labels, seq_lengths, _ in dataloader:
            input_list = []

            # Prepare measurement data
            measurement_x = values.to(device)
            labels = labels.to(device)

            if use_measurements:
                input_list.append({"x": measurement_x})

            logits = model(input_list)
            logits = torch.sigmoid(logits)

            if task == "IHM":
                logits = logits[:, 0]

            y = labels
            loss = criterion(logits, y)
            pred = logits

            ys.extend(y.cpu().numpy())
            preds.extend(pred.cpu().numpy())

            logger.update(pred, y, loss)

    logger.print_metrics(epoch,split="Test")

    preds = np.array(preds)
    binary_preds = (preds > threshold).astype(int)  # Apply threshold for binary classification

    # Compute precision, recall, and F1-score
    accuracy = accuracy_score(ys, binary_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(ys, binary_preds, average='binary')


    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")


def train(args, train_dataloader, test_dataloader):
    base_model = create_model(args)

    if args.pretrained_path is not None:
        base_model = load_pretrained_model(base_model, args)

    model = ClinicalMultiModal(
        base_model,
        args.measurement_emb_size,
        args.n_classes
    ).to(args.device)


    params = model.classifier.parameters()
    model.base_model.eval()

    optimizer = optim.AdamW(
        params,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
    )

    lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.task == "IHM":
        logger = IHMLogger(args.exp_name, args, log_wandb=False)

    criteria = nn.BCELoss()

    for epoch in range(args.epochs):
        ys = []
        preds = []
        print(f"LR = {optimizer.param_groups[0]['lr']}")
        
        model.classifier.train()
        logger.reset()

        for values, labels, seq_lengths, _ in train_dataloader:
            optimizer.zero_grad()

            # Prepare measurement data
            measurement_x = values.to(args.device)
            labels = labels.to(args.device)

            input_list = []

            if args.use_measurements:
                input_list.append(
                    {
                        "x": measurement_x,
                    }
                )

            seq_lengths = torch.LongTensor(seq_lengths)
            logits = model(input_list)
            logits = F.sigmoid(logits)

            if args.task == "IHM":
                logits = logits[:, 0]

            y = labels

            loss = criteria(logits, y)
            loss.backward()
            optimizer.step()

            pred = logits

            ys.extend(y.detach().cpu().numpy())
            preds.extend(pred.detach().cpu().numpy())

            logger.update(pred, y, loss)

        lr_scheduler.step(epoch + 1)
        logger.print_metrics(epoch, split="Train")
        logger.reset()

        preds=np.array(preds)
        binary_preds=(preds>=0.5).astype(int)
        # Compute accuracy_score, precision, recall, and F1-score
        accuracy=accuracy_score(ys, binary_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(ys, binary_preds, average='binary')
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")


        evaluate_model(model,test_dataloader,args.device,criteria,logger,epoch)
        

if __name__ == "__main__":
    activation_map = {"GELU": nn.GELU(), "ReLU": nn.ReLU()}
    args = parser()
    args.measurement_activation = activation_map[args.measurement_activation]

    args.use_projector = False

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = True

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

    listfile=""
    if args.task == "IHM":
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
        test_dataloader,
    )
