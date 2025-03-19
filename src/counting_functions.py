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

        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        union = tp + fp + fn

        metrics[cls.item()] = {
            'accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'precision': precision_score(y_true_binary, y_pred_binary),
            'recall': recall_score(y_true_binary, y_pred_binary),
            'f1': f1_score(y_true_binary, y_pred_binary),
            'IoU': tp/union if union != 0 else 0
        }

    return metrics


def count_metrics(y_true, y_pred):
    y_pred = torch.softmax(y_pred, dim=1).argmax(dim=1).cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Count mIoU
    classes = np.unique(y_true)
    iou = []

    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        union = tp + fp + fn
        iou.append(tp/union if union != 0 else 0)

    mIoU = np.mean(iou)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mIoU': mIoU
    }
