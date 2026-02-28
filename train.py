import os
import sys
import random
import multiprocessing as mp

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from bayes_opt import BayesianOptimization

from resnet import Net
from physics_models import precompute_physics_predictions

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

seed = 2025
np.random.seed(seed)
random.seed(seed)

class Args:
    def __init__(self):
        self.train_path = "./train1/train_data1_split.csv"
        self.val_path = "./train1/val_data1.csv"
        self.test_path = "./train1/test_data1.csv"
        self.epoch = 2000
        self.lr = 0.001
        self.batch_size = 128
        self.physics_weight = 0.1
        self.metrics_save_path_base = "./training_metrics"
        self.model_save_path_base = "./mopie_model"
        self.bo_init_points = 8
        self.bo_n_iter = 12
        self.log_every = 100
        self.num_workers = 0
        self.min_required_columns = [
            "弹体重量",
            "弹体直径",
            "弹体长度",
            "侵彻速度",
            "靶体抗压强度",
            "弹体形状系数",
            "侵彻深度",
        ]


def parse_args():
    return Args()


args = parse_args()

initial_physics_weight = args.physics_weight
total_epochs_per_evaluation = args.epoch
base_lr = args.lr
device = "cuda" if torch.cuda.is_available() else "cpu"


def validate_columns(df, required_cols, df_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing required columns: {missing}")


class MinMaxScalerByTrain:
    def __init__(self):
        self.min_vals = None
        self.max_vals = None
        self.range_vals = None
        self.numeric_cols = None

    def fit(self, df):
        numeric_cols = df.select_dtypes(exclude=["object"]).columns.tolist()
        self.numeric_cols = numeric_cols
        self.min_vals = df[numeric_cols].min()
        self.max_vals = df[numeric_cols].max()
        self.range_vals = self.max_vals - self.min_vals
        self.range_vals[self.range_vals == 0] = 1.0
        return self

    def transform(self, df):
        out = df.copy()
        out[self.numeric_cols] = (out[self.numeric_cols] - self.min_vals) / self.range_vals
        return out


class MoPIEDataset(Dataset):
    def __init__(self, features_norm_tensor, features_orig_np, labels_tensor, target_densities=None,
                 physics_preds=None, physics_masks=None):
        self.features_norm = features_norm_tensor
        self.features_orig_np = features_orig_np
        self.labels = labels_tensor
        self.target_densities = target_densities
        self.physics_preds = physics_preds
        self.physics_masks = physics_masks

        if self.target_densities is not None and len(self.target_densities) != len(self.labels):
            raise ValueError("target_densities length must match number of samples")
        if self.physics_preds is not None and len(self.physics_preds) != len(self.labels):
            raise ValueError("physics_preds length must match number of samples")
        if self.physics_masks is not None and len(self.physics_masks) != len(self.labels):
            raise ValueError("physics_masks length must match number of samples")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "norm": self.features_norm[idx],
            "orig": self.features_orig_np[idx],
            "label": self.labels[idx],
        }
        if self.target_densities is not None:
            item["q"] = float(self.target_densities[idx])
        if self.physics_preds is not None:
            item["physics_pred"] = self.physics_preds[idx]
        if self.physics_masks is not None:
            item["physics_mask"] = self.physics_masks[idx]
        return item


def collate_fn(batch):
    batch_dict = {}
    for key in batch[0].keys():
        if key in ["norm", "label"]:
            batch_dict[key] = torch.stack([x[key] for x in batch])
        elif key in ["q", "physics_pred"]:
            batch_dict[key] = torch.tensor([x[key] for x in batch], dtype=torch.float32)
        elif key == "physics_mask":
            batch_dict[key] = torch.tensor([x[key] for x in batch], dtype=torch.bool)
        elif key == "orig":
            batch_dict[key] = np.array([x[key] for x in batch], dtype=np.float32)
        else:
            batch_dict[key] = [x[key] for x in batch]
    return batch_dict


def make_tensors(df_raw_features, feature_cols, label_col):
    x = torch.tensor(df_raw_features[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df_raw_features[label_col].values, dtype=torch.float32).view(-1, 1)
    return x, y


def compute_metrics(y_true_np, y_pred_np):
    eps = 1e-12
    y_true_np = y_true_np.reshape(-1)
    y_pred_np = y_pred_np.reshape(-1)

    mae = np.mean(np.abs(y_pred_np - y_true_np))
    mse = np.mean((y_pred_np - y_true_np) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_true_np - y_pred_np) ** 2)
    ss_tot = np.sum((y_true_np - np.mean(y_true_np)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + eps))

    valid = np.abs(y_true_np) > eps
    a20 = np.mean(np.abs(y_pred_np[valid] - y_true_np[valid]) / np.abs(y_true_np[valid]) <= 0.2) if np.any(
        valid) else 0.0

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "a20-index": float(a20),
    }


def evaluate_full_metrics(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["norm"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            pred = model(x)
            preds.append(pred.detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())

    if len(preds) == 0:
        return {"mae": float("inf"), "mse": float("inf"), "rmse": float("inf"), "r2": float("-inf"), "a20-index": 0.0}

    return compute_metrics(np.vstack(trues), np.vstack(preds))


def evaluate_mse(model, loader, mse_criterion):
    model.eval()
    total, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch["norm"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            pred = model(x)
            total += mse_criterion(pred, y).item()
            n_batches += 1
    return total / n_batches if n_batches > 0 else float("inf")


def prepare_data(train_df, val_df, test_df, feature_cols, label_col, feature_index):
    numeric_feature_cols = train_df[feature_cols].select_dtypes(exclude=["object"]).columns.tolist()
    train_medians = train_df[numeric_feature_cols].median()

    train_df[numeric_feature_cols] = train_df[numeric_feature_cols].fillna(train_medians)
    val_df[numeric_feature_cols] = val_df[numeric_feature_cols].fillna(train_medians)
    test_df[numeric_feature_cols] = test_df[numeric_feature_cols].fillna(train_medians)

    scaler = MinMaxScalerByTrain().fit(train_df[feature_cols])
    train_features_norm_df = scaler.transform(train_df[feature_cols])
    val_features_norm_df = scaler.transform(val_df[feature_cols])
    test_features_norm_df = scaler.transform(test_df[feature_cols])

    train_x_norm, train_y = make_tensors(pd.concat([train_features_norm_df, train_df[[label_col]]], axis=1),
                                         feature_cols, label_col)
    val_x_norm, val_y = make_tensors(pd.concat([val_features_norm_df, val_df[[label_col]]], axis=1), feature_cols,
                                     label_col)
    test_x_norm, test_y = make_tensors(pd.concat([test_features_norm_df, test_df[[label_col]]], axis=1), feature_cols,
                                       label_col)

    train_x_orig_np = train_df[feature_cols].values.astype(np.float32)
    val_x_orig_np = val_df[feature_cols].values.astype(np.float32)
    test_x_orig_np = test_df[feature_cols].values.astype(np.float32)

    train_q = train_df["靶体密度"].tolist() if "靶体密度" in train_df.columns else [2300.0] * len(train_df)
    val_q = val_df["靶体密度"].tolist() if "靶体密度" in val_df.columns else [2300.0] * len(val_df)
    test_q = test_df["靶体密度"].tolist() if "靶体密度" in test_df.columns else [2300.0] * len(test_df)

    print("Precomputing training set physics predictions...")
    train_physics_preds, train_physics_masks = precompute_physics_predictions(train_x_orig_np, train_q, feature_index)

    print("Precomputing validation set physics predictions...")
    val_physics_preds, val_physics_masks = precompute_physics_predictions(val_x_orig_np, val_q, feature_index)

    print("Precomputing test set physics predictions...")
    test_physics_preds, test_physics_masks = precompute_physics_predictions(test_x_orig_np, test_q, feature_index)

    train_dataset = MoPIEDataset(train_x_norm, train_x_orig_np, train_y, target_densities=train_q,
                                 physics_preds=train_physics_preds, physics_masks=train_physics_masks)
    val_dataset = MoPIEDataset(val_x_norm, val_x_orig_np, val_y, target_densities=val_q,
                               physics_preds=val_physics_preds, physics_masks=val_physics_masks)
    test_dataset = MoPIEDataset(test_x_norm, test_x_orig_np, test_y, target_densities=test_q,
                                physics_preds=test_physics_preds, physics_masks=test_physics_masks)

    pin = torch.cuda.is_available()
    train_iter = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=pin,
        num_workers=args.num_workers,
    )
    val_iter = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=pin,
        num_workers=args.num_workers,
    )
    test_iter = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=pin,
        num_workers=args.num_workers,
    )

    return train_iter, val_iter, test_iter, scaler


def run_single_training(train_iter, val_iter, test_iter, physics_decay_epochs):
    physics_decay_epochs = int(round(physics_decay_epochs))
    net = Net().to(device)

    criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=base_lr, weight_decay=1e-4)

    trial_train_losses, trial_train_mse_losses, trial_train_physics_losses = [], [], []
    trial_train_maes, trial_val_mses = [], []
    trial_test_maes, trial_test_mses, trial_test_rmses, trial_test_r2s, trial_test_a20s = [], [], [], [], []

    use_amp = torch.cuda.is_available()
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    for e in range(total_epochs_per_evaluation):
        if physics_decay_epochs > 0 and e < physics_decay_epochs:
            current_physics_weight = initial_physics_weight * max(0.0,
                                                                  (physics_decay_epochs - e) / physics_decay_epochs)
        else:
            current_physics_weight = 0.0

        net.train()
        running_total_loss, running_mse_loss, running_physics_loss, running_mae = 0.0, 0.0, 0.0, 0.0
        valid_batches = 0

        for batch in train_iter:
            x_norm_batch = batch["norm"].to(device, non_blocking=True)
            y_batch = batch["label"].to(device, non_blocking=True)

            physics_preds_batch = batch["physics_pred"].to(device, non_blocking=True).view(-1, 1)
            physics_masks_batch = batch["physics_mask"].to(device, non_blocking=True).view(-1, 1)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = net(x_norm_batch)
                mse_loss = criterion(output, y_batch)

                physics_relevant = physics_masks_batch.squeeze()
                physics_loss = (
                    criterion(output[physics_relevant], physics_preds_batch[physics_relevant])
                    if torch.any(physics_relevant)
                    else torch.tensor(0.0, device=device)
                )
                total_loss = mse_loss + current_physics_weight * physics_loss
                mae = mae_criterion(output, y_batch)

            scaler_amp.scale(total_loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()

            running_total_loss += total_loss.item()
            running_mse_loss += mse_loss.item()
            running_physics_loss += physics_loss.item()
            running_mae += mae.item()
            valid_batches += 1

        if valid_batches > 0:
            train_total_loss = running_total_loss / valid_batches
            trial_train_losses.append(train_total_loss)
            trial_train_mse_losses.append(running_mse_loss / valid_batches)
            trial_train_physics_losses.append(running_physics_loss / valid_batches)
            trial_train_maes.append(running_mae / valid_batches)
        else:
            train_total_loss = 0.0
            trial_train_losses.append(0.0)
            trial_train_mse_losses.append(0.0)
            trial_train_physics_losses.append(0.0)
            trial_train_maes.append(0.0)

        current_val_mse = evaluate_mse(net, val_iter, criterion)
        trial_val_mses.append(current_val_mse)

        test_metrics = evaluate_full_metrics(net, test_iter)
        trial_test_maes.append(test_metrics["mae"])
        trial_test_mses.append(test_metrics["mse"])
        trial_test_rmses.append(test_metrics["rmse"])
        trial_test_r2s.append(test_metrics["r2"])
        trial_test_a20s.append(test_metrics["a20-index"])

        if (e + 1) % args.log_every == 0 or e == 0:
            print(
                f"Epoch {e + 1}/{total_epochs_per_evaluation} | Train Loss {train_total_loss:.4f} | Val MSE {current_val_mse:.4f}")

    metrics_save_path = f"{args.metrics_save_path_base}_fixed_split_decay{physics_decay_epochs}.csv"
    model_save_path = f"{args.model_save_path_base}_fixed_split_decay{physics_decay_epochs}.pt"

    metrics_df = pd.DataFrame(
        {
            "Epoch": range(1, len(trial_train_losses) + 1),
            "Total Weighted Loss": trial_train_losses,
            "Data Loss (MSE)": trial_train_mse_losses,
            "Physics Loss": trial_train_physics_losses,
            "Train MAE": trial_train_maes,
            "Val MSE": trial_val_mses,
            "Test MAE": trial_test_maes,
            "Test MSE": trial_test_mses,
            "Test RMSE": trial_test_rmses,
            "Test R2": trial_test_r2s,
            "Test A20": trial_test_a20s,
        }
    )
    metrics_df.to_csv(metrics_save_path, index=False)
    torch.save(net.state_dict(), model_save_path)

    final_val_metrics = evaluate_full_metrics(net, val_iter)
    final_test_metrics = evaluate_full_metrics(net, test_iter)

    return -trial_val_mses[-1], final_val_metrics, final_test_metrics, model_save_path


def main():
    try:
        train_data = pd.read_csv(args.train_path, encoding="gbk")
        val_data = pd.read_csv(args.val_path, encoding="gbk")
        test_data = pd.read_csv(args.test_path, encoding="gbk")
    except FileNotFoundError as e:
        print(f"Error: file not found: {e.filename}")
        print("Please run select_validation_set.py first to generate train_data1_split.csv and val_data1.csv")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    try:
        validate_columns(train_data, args.min_required_columns, "train_data")
        validate_columns(val_data, args.min_required_columns, "val_data")
        validate_columns(test_data, args.min_required_columns, "test_data")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    label_col = "侵彻深度"
    feature_cols = [c for c in train_data.columns if c != label_col]

    feature_index = {c: i for i, c in enumerate(feature_cols)}
    needed_feature_for_physics = ["弹体重量", "弹体直径", "弹体长度", "侵彻速度", "靶体抗压强度", "弹体形状系数"]
    for c in needed_feature_for_physics:
        if c not in feature_index:
            print(f"Error: missing feature column {c}, cannot compute physics loss.")
            sys.exit(1)

    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(val_data)}")
    print(f"Test size: {len(test_data)}")

    train_iter, val_iter, test_iter, scaler = prepare_data(
        train_data, val_data, test_data, feature_cols, label_col, feature_index
    )

    def bo_objective(physics_decay_epochs):
        score, _, _, _ = run_single_training(train_iter, val_iter, test_iter, physics_decay_epochs)
        return score

    pbounds = {"physics_decay_epochs": (0, total_epochs_per_evaluation)}
    bo = BayesianOptimization(
        f=bo_objective,
        pbounds=pbounds,
        random_state=seed,
        verbose=0,
    )

    bo.maximize(init_points=args.bo_init_points, n_iter=args.bo_n_iter)

    best_physics_decay_epochs = int(round(bo.max["params"]["physics_decay_epochs"]))

    _, val_metrics, test_metrics, best_model_path = run_single_training(
        train_iter, val_iter, test_iter, best_physics_decay_epochs
    )

    result = {
        "best_physics_decay_epochs": best_physics_decay_epochs,
        "val_mae": val_metrics["mae"],
        "val_mse": val_metrics["mse"],
        "val_rmse": val_metrics["rmse"],
        "val_r2": val_metrics["r2"],
        "val_a20": val_metrics["a20-index"],
        "test_mae": test_metrics["mae"],
        "test_mse": test_metrics["mse"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_a20": test_metrics["a20-index"],
        "train_size": len(train_data),
        "val_size": len(val_data),
    }

    result_df = pd.DataFrame([result])
    result_save_path = f"{args.metrics_save_path_base}_fixed_split_result.csv"
    result_df.to_csv(result_save_path, index=False)

    print("\n" + "=" * 80)
    print("Final Results:")
    print("=" * 80)
    print(result_df.to_string(index=False))
    print(f"\nMetrics saved to: {result_save_path}")
    print(f"Model saved to: {best_model_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()

