import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from clmnet_model import CLMNet

def evaluate(model, dataloader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    acc = accuracy_score(y_true, y_pred > 0.5)
    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred > 0.5)
    print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
