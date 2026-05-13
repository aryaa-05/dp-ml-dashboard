import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class DpSurvivalNetwork(nn.Module):
    """
    Simple Neural Network Architecture for the METABRIC survival task.
    """
    def __init__(self, input_dim):
        super(DpSurvivalNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

def evaluate_model(y_true, y_pred, y_prob=None):
    """
    Evaluates the model and returns a dictionary of metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            pass
    return metrics

def get_dummy_predictions(features_df):
    """
    Returns dummy predictions for UI demonstration when model weights aren't loaded.
    """
    import numpy as np
    preds = np.random.randint(0, 2, size=len(features_df))
    probs = np.random.rand(len(features_df))
    return preds, probs
