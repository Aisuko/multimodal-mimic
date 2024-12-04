# XGBoost

```bash
/home/ubuntu/workspace/multimodal-mimic/experiments/measurement_notes/traditional_xgboost.py 
Length of training dataset = 14681
Length of test dataset = 3236
(73405, 3648)
(3236, 3648)
Accuracy: 0.8922, Precision: 0.5740, Recall: 0.2594, F1-Score: 0.3573, AUC-ROC: 0.8295
```

# LogisticRegression

```bash
/home/ubuntu/workspace/multimodal-mimic/experiments/measurement_notes/traditional_logistic_regression.py 
Length of training dataset = 14681
Length of test dataset = 3236
Training dataset shape (14681, 3648)
(3236, 3648)
Accuracy: 0.8643, Precision: 0.4079, Recall: 0.3850, F1-Score: 0.3961, AUC-ROC: 0.7718
```

# RandomForestClassifier

```bash
/home/ubuntu/workspace/multimodal-mimic/experiments/measurement_notes/traditional_random_forest.py 
Length of training dataset = 14681
Length of test dataset = 3236
(14681, 3648)
(3236, 3648)
Accuracy: 0.8962, Precision: 0.6827, Recall: 0.1898, F1-Score: 0.2971, AUC-ROC: 0.8466
```

# Multimodal-mimic

```bash
Test:  Epoch 400, Loss=0.28680065053207765, AUC-ROC=0.8182182535678652, AUC-PR=0.3747169427397216
Accuracy: 0.8869, Precision: 0.5426, Recall: 0.1364, F1-Score: 0.2179
```
