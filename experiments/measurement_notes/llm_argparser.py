import argparse


def parser():
    parser = argparse.ArgumentParser("multimodal_downstream")

    # Experiment
    parser.add_argument("--exp_name", type=str, default="exp")
    parser.add_argument("--task", choices=["IHM", "Phenotyping"], default="IHM")
    # parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--pretrained_path", type=str, default='/workspaces/multimodal-mimic/multimodal_clinical_pretraining/pretrain/pretrained_model.pth')
    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument(
        "--report_freq", type=float, default=100, help="report frequency"
    )

    # Data
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--mimic_root", type=str, default="/workspaces/mimic_iii_1.4"
    )
    parser.add_argument(
        "--mimic_benchmark_root",
        type=str,
        default="/workspaces/multimodal-mimic/mimic3-benchmarks",
    )

    # Training
    parser.add_argument("--opt", default="adamW") 
    parser.add_argument("--sched", default="cosine") # dynamic asjust the learning rate during training
    parser.add_argument("--lr", default=1e-3, type=float) # 
    parser.add_argument("--linear_lr", default=0.001, type=float)
    parser.add_argument("--min_lr", default=1e-4, type=float)
    parser.add_argument("--warmup", default=True, type=bool)
    parser.add_argument("--warmup_lr", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=0.02, type=float)
    parser.add_argument("--decay_rate", default=0.9, type=float) # ensure a gradual reduction in the learning rate
    parser.add_argument("--warmup_epochs", default=0, type=int) # for fitting 1e-3, we set 5-10 epochs to allow the model to gradually adapt to the learning process
    parser.add_argument("--cooldown_epochs", default=5, type=int)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--grad_clip", type=float, default=5)

    # Multimodal Model
    parser.add_argument("--use_cls_token", action="store_true")
    parser.add_argument("--mlp", default="128")

    # Measurement Model
    parser.add_argument("--use_measurements", action="store_true")
    parser.add_argument(
        "--measurement_model", default="Transformer", choices=["Transformer"]
    )
    parser.add_argument("--measurement_emb_size", type=int, default=128)
    parser.add_argument("--measurement_num_heads", type=int, default=8)
    parser.add_argument("--measurement_num_layers", type=int, default=8)
    parser.add_argument("--use_pos_emb", action="store_true")
    parser.add_argument(
        "--measurement_activation", choices=["ReLU", "GELU"], default="GELU"
    )
    parser.add_argument("--measurement_mask_rate", type=float, default=0.1)
    parser.add_argument("--measurement_max_seq_len", type=int, default=256)
    parser.add_argument("--measurement_dropout", type=float, default=0)

    # Notes Model
    parser.add_argument("--text_model", default="BERT", choices=["BERT"])
    parser.add_argument("--use_notes", action="store_true")
    parser.add_argument("--notes_emb_size", type=int, default=128)
    parser.add_argument("--notes_num_heads", type=int, default=8)
    parser.add_argument("--notes_num_layers", type=int, default=8)
    parser.add_argument("--notes_mask_rate", type=float, default=0.4)
    parser.add_argument("--notes_dropout", type=float, default=0)
    parser.add_argument("--notes_max_seq_len", type=int, default=256)

    parser.add_argument("--device", default="cuda")

    return parser.parse_args()
