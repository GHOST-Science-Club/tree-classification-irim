from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import torch

def calculate_metrics_per_class(y_true, y_pred):
    y_pred = torch.softmax(y_pred, dim=1).argmax(dim=1).cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()

    classes = np.unique(y_true)
    metrics = {}

    for cls in classes:
        y_true_binary = (y_true == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)

        metrics[cls.item()] = {
            'accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'precision': precision_score(y_true_binary, y_pred_binary),
            'recall': recall_score(y_true_binary, y_pred_binary),
            'f1': f1_score(y_true_binary, y_pred_binary)
        }

    return metrics


