import os
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



class TrainLogger:
    def __init__(self, exp_id, run_id, base_dir="save/"):
        self.log_dir = os.path.join(base_dir, f"exp{exp_id}", f"run{run_id}")
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = self._init_logger()
        self.summary = {}

    def _init_logger(self):
        logger = logging.getLogger(f"exp_logger_{datetime.now().isoformat()}")
        logger.setLevel(logging.INFO)

        log_file = os.path.join(self.log_dir, "train.log")
        file_handler = logging.FileHandler(log_file)
        stream_handler = logging.StreamHandler()

        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return logger

    def log(self, message):
        self.logger.info(message)

    def log_epoch(self, epoch, train_loss, val_loss):
        self.logger.info(f"Epoch {epoch} | "
                         f"Train Loss (MSE): {train_loss:.4f} | "
                         f"Val Loss (MSE): {val_loss:.4f}")

    def log_metrics(self, val_metrics, test_metrics):
        self.summary["val_mse"] = float(val_metrics[0])
        self.summary["val_rmse"] = float(val_metrics[1])
        self.summary["val_mae"] = float(val_metrics[2])

        self.summary["test_mse"] = float(test_metrics[0])
        self.summary["test_rmse"] = float(test_metrics[1])
        self.summary["test_mae"] = float(test_metrics[2])

        self.logger.info(f"Validation MSE: {val_metrics[0]:.4f}, RMSE: {val_metrics[1]:.4f},  MAE: {val_metrics[2]:.4f}")
        self.logger.info(f"Test MSE: {test_metrics[0]:.4f}, RMSE: {test_metrics[1]:.4f}, MAE: {test_metrics[2]:.4f}")

    def save_training_summary(self, train_loss, val_loss, train_time, val_time, args):
        self.summary["train_loss_per_epoch"] = list(map(float, train_loss))
        self.summary["val_loss_per_epoch"] = list(map(float, val_loss))
        self.summary["train_time"] = list(map(float, train_time))
        self.summary["val_time"] = list(map(float, val_time))

        summary_path = os.path.join(self.log_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(self.summary, f, indent=4)

        # Save args as well
        args_path = os.path.join(self.log_dir, "args.json")
        with open(args_path, "w") as f:
            json.dump(vars(args), f, indent=4)
        
        self._save_plots(train_loss, val_loss)

    def _save_plots(self, train_loss, val_loss):
        plt.figure()
        plt.plot(train_loss, label="Train Loss")
        plt.plot(val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "loss_plot.png"))
        plt.close()

    def append_to_global_summary(self, args, val_metrics, test_metrics, run_id):
        exp_dir = os.path.join(self.log_dir)  
        base_exp_dir = os.path.dirname(self.log_dir)
        summary_path = os.path.join(base_exp_dir, "summary.csv")

        row = {
            "expid": args.expid,
            "run": run_id,
            "val_mse": val_metrics[0],
            "val_rmse": val_metrics[1],
            "val_mae": val_metrics[2],
            "test_mse": test_metrics[0],
            "test_rmse": test_metrics[1],
            "test_mae": test_metrics[2]
        }

        for key, value in vars(args).items():
            if key not in row:
                row[key] = value

        df = pd.DataFrame([row])
        if not os.path.exists(summary_path):
            df.to_csv(summary_path, index=False)
        else:
            df.to_csv(summary_path, mode='a', header=False, index=False)


