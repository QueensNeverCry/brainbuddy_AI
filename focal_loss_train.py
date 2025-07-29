import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.simple_engagement_model import SimpleEngagementModel
from feature_dataset import CNNFeatureDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import torch.nn.functional as F
import random

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def objective(trial):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CNNFeatureDataset([
        "./cnn_features/features/train_20_01.pkl",
        "./cnn_features/features/train_20_03.pkl",
        "./cnn_features/features/D_train.pkl"
    ])
    val_dataset = CNNFeatureDataset([
        "./cnn_features/features/valid_20_01.pkl",
        "./cnn_features/features/valid_20_03.pkl",
        "./cnn_features/features/D_val.pkl"
    ])

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = SimpleEngagementModel().to(device)

    alpha = trial.suggest_float("alpha", 0.25, 0.95)
    gamma = trial.suggest_float("gamma", 1.0, 5.0)
    criterion = FocalLoss(alpha=alpha, gamma=gamma)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_f1 = 0
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        for features, labels in train_loader:
            features = features.to(device).float()
            labels = labels.to(device).float().view(-1)

            optimizer.zero_grad()
            outputs = model(features).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        all_probs, all_labels = [], []
        val_loss = 0.0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device).float()
                labels = labels.to(device).float().view(-1)

                outputs = model(features).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs).detach().cpu()
                all_probs.append(probs)
                all_labels.append(labels.cpu())

        scheduler.step(val_loss)
        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # threshold tuning
        best_threshold = 0.5
        best_f1_local = 0.0
        for t in np.arange(0.1, 0.9, 0.05):
            preds = (all_probs > t).astype(int)
            f1 = f1_score(all_labels, preds)
            if f1 > best_f1_local:
                best_f1_local = f1
                best_threshold = t

        if best_f1_local > best_val_f1:
            best_val_f1 = best_f1_local

    return best_val_f1  # Maximize F1

def tune():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (F1): {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == '__main__':
    tune()
