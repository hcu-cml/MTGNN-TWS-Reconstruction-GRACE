import json
import torch
import numpy as np
from net.util import metric

def save_metrics(metrics_dict, path):
    with open(path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)

def log_metrics(logger, title, metrics):
    logger.info(f"{title} - MSE: {metrics['mse']:.4f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")

def evaluate_model(engine, data_loader, device, target_scaler=None):
    engine.model.eval()
    preds = []
    real = []

    with torch.no_grad():
        for x, y in data_loader.get_iterator():
            x_tensor = torch.Tensor(x).to(device).permute(0, 3, 2, 1)
            y_tensor = torch.Tensor(y).to(device).permute(0, 3, 2, 1)

            pred = engine.model(x_tensor).transpose(1, 3).squeeze()  # [B, N]
            preds.append(pred)
            real.append(y_tensor[:, 0, :, :].squeeze())

    yhat = torch.cat(preds, dim=0)
    ytrue = torch.cat(real, dim=0)

    ytrue = ytrue.squeeze(-1) if ytrue.dim() == 3 else ytrue

    if target_scaler is not None:
        yhat_np = yhat.cpu().numpy()
        ytrue_np = ytrue.cpu().numpy()

        yhat_np = target_scaler[0].inverse_transform(yhat_np)  # falls nur 1 Feature im Target
        ytrue_np = target_scaler[0].inverse_transform(ytrue_np)

        yhat = torch.tensor(yhat_np)
        ytrue = torch.tensor(ytrue_np)

    mse, rmse, mae = metric(yhat, ytrue)
    return mse, rmse, mae