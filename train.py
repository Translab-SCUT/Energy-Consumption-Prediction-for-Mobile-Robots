from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from model import (
    PLAN_CONTINUOUS_DIM,
    PLAN_OUTPUTS,
    REGRESSION_OUTPUTS,
    MultiTaskLSTMAttentionModel,
)


SCRIPT_DIR = Path(__file__).resolve().parent
MIN_SINGLE_FILE_ISOLATION_SECONDS = 390.0
ABSOLUTE_POSITION_COLUMNS = ["x", "y", "z"]
DEPRECATED_POSITION_DELTA_COLUMNS = ["positionDeltaX", "positionDeltaY", "positionDeltaZ"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Train the physics-structured TAE + future-plan model."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(SCRIPT_DIR / "data" / "robot.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SCRIPT_DIR / "results" / "lstm_attention_multitask_plan_v3"),
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--split-mode",
        choices=["file", "date"],
        default="file",
        help="date classification",
    )
    parser.add_argument(
        "--isolation-seconds",
        type=float,
        default=MIN_SINGLE_FILE_ISOLATION_SECONDS,
        help="Single-file time block",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--regression-loss",
        choices=["mse", "huber"],
        default="huber",
        help="Loss for TAE only; energy/voltage/temperature always use MSE.",
    )
    parser.add_argument("--huber-beta", type=float, default=0.5)
    parser.add_argument("--target-loss-weights", type=str, default="2,1.5,0.1,0.1")
    parser.add_argument("--beta-loss-weight", type=float, default=1.0)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.1)
    parser.add_argument("--plan-loss-weight", type=float, default=0.05)
    parser.add_argument(
        "--selection-metric",
        choices=["tae_energy_rmse", "tae_wh_rmse", "loss"],
        default="tae_energy_rmse",
        help=(
            "Model-selection metric"
        ),
    )
    parser.add_argument("--lstm-hidden-dim", type=int, default=64)
    parser.add_argument("--lstm-num-layers", type=int, default=2)
    parser.add_argument("--shared-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def compute_selection_value(
    selection_metric: str,
    val_metrics: dict[str, float],
    stats: dict[str, np.ndarray],
) -> tuple[float, dict[str, float]]:


    tae_scale = max(float(np.asarray(stats["target_std"])[0]), 1e-12)
    energy_scale = max(float(np.asarray(stats["target_std"])[1]), 1e-12)
    tae_rmse = float(val_metrics["tae_wh_rmse"])
    energy_rmse = float(val_metrics["future_energy_wh_rmse"])
    if not np.isfinite(tae_rmse):
        raise ValueError(
            "The validation split contains no reliable TAE samples"
        )
    normalized = {
        "tae_rmse_normalized": tae_rmse / tae_scale,
        "future_energy_rmse_normalized": energy_rmse / energy_scale,
    }
    if selection_metric == "tae_energy_rmse":
        return (
            0.7 * normalized["tae_rmse_normalized"]
            + 0.3 * normalized["future_energy_rmse_normalized"],
            normalized,
        )
    if selection_metric == "tae_wh_rmse":
        return tae_rmse, normalized
    if selection_metric == "loss":
        return float(val_metrics["loss"]), normalized
    raise ValueError(f"Unsupported selection metric: {selection_metric}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch = require_torch()
    set_seed(args.seed)

    data = load_dataset(args.data_path)
    train_idx, val_idx, test_idx, split_info = split_train_val_test_indices(
        data=data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_mode=args.split_mode,
        isolation_seconds=args.isolation_seconds,
    )
    stats = fit_stats(data, train_idx)
    train_dataset = MultiTaskDataset(data, train_idx, stats)
    val_dataset = MultiTaskDataset(data, val_idx, stats)
    test_dataset = MultiTaskDataset(data, test_idx, stats)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    head_initial_values = fit_head_initial_values(data, train_idx)
    model = MultiTaskLSTMAttentionModel(
        category_cardinalities=data["category_cardinalities"],
        continuous_dim=data["x_cont"].shape[-1],
        init_dim=data["x_init"].shape[-1],
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_num_layers=args.lstm_num_layers,
        shared_hidden_dim=args.shared_hidden_dim,
        dropout=args.dropout,
        num_regression_targets=len(REGRESSION_OUTPUTS),
        plan_target_mean=stats["plan_target_mean"].tolist(),
        plan_target_std=stats["plan_target_std"].tolist(),
        initial_dock_probability=float(stats["plan_dock_positive_rate"][0]),
        **head_initial_values,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    loss_fn = build_multitask_loss(torch, args, stats).to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_state: dict[str, Any] | None = None
    best_selection_value = float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []

    overlap_count = int(split_info.get("train_test_overlap_count", 0))
    overlap_note = (
        "untouched"
        if overlap_count == 0
        else f"including {overlap_count} exact training-window duplicates"
    )
    print(
        f"Training plan+physics model on {len(train_idx)} samples; "
        f"validating on {len(val_idx)} samples; testing on {len(test_idx)} samples "
        f"({overlap_note}); device={device}; split={split_info['strategy']}."
    )
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip=args.grad_clip,
        )
        val_metrics, _, _, _ = evaluate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            target_names=data["regression_targets"],
            plan_target_names=data["plan_targets"],
            stats=stats,
        )
        selection_value, selection_components = compute_selection_value(
            args.selection_metric,
            val_metrics,
            stats,
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": float(epoch),
            "learning_rate": current_lr,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{
                f"val_{key}": value
                for key, value in val_metrics.items()
                if key.endswith("loss")
            },
            "val_tae_wh_rmse": float(val_metrics["tae_wh_rmse"]),
            "val_future_energy_wh_rmse": float(val_metrics["future_energy_wh_rmse"]),
            "val_tae_wh_rmse_normalized": selection_components["tae_rmse_normalized"],
            "val_future_energy_wh_rmse_normalized": selection_components[
                "future_energy_rmse_normalized"
            ],
            "val_future_dock_event_f1": float(
                val_metrics.get("future_dock_event_f1", float("nan"))
            ),
            "selection_value": selection_value,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train={train_metrics['loss']:.6f} | "
            f"val={val_metrics['loss']:.6f} | plan={val_metrics['plan_loss']:.6f} | "
            f"TAE_RMSE={val_metrics['tae_wh_rmse']:.3f} Wh | "
            f"E_H_RMSE={val_metrics['future_energy_wh_rmse']:.3f} Wh | "
            f"select={selection_value:.6f} ({args.selection_metric}) | lr={current_lr:.2e}"
        )

        if selection_value < best_selection_value:
            best_selection_value = selection_value
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            torch.save(
                make_checkpoint(
                    model_state=best_state,
                    epoch=epoch,
                    args=args,
                    data=data,
                    stats=stats,
                    head_initial_values=head_initial_values,
                    split_info=split_info,
                ),
                output_dir / "best_model.pt",
            )
        else:
            stale_epochs += 1

        scheduler.step(selection_value)
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint.")

    model.load_state_dict(best_state)
    val_metrics, val_y_true, val_y_pred, val_auxiliary = evaluate(
        model=model,
        dataloader=val_loader,
        loss_fn=loss_fn,
        device=device,
        target_names=data["regression_targets"],
        plan_target_names=data["plan_targets"],
        stats=stats,
    )
    test_metrics, test_y_true, test_y_pred, test_auxiliary = evaluate(
        model=model,
        dataloader=test_loader,
        loss_fn=loss_fn,
        device=device,
        target_names=data["regression_targets"],
        plan_target_names=data["plan_targets"],
        stats=stats,
    )
    val_metrics["best_epoch"] = float(best_epoch)
    val_metrics["best_val_loss"] = float(best_val_loss)
    val_metrics["best_selection_value"] = float(best_selection_value)
    test_metrics["selected_epoch"] = float(best_epoch)
    save_outputs(
        output_dir=output_dir,
        data=data,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        split_info=split_info,
        val_metrics=val_metrics,
        val_y_true=val_y_true,
        val_y_pred=val_y_pred,
        val_auxiliary=val_auxiliary,
        test_metrics=test_metrics,
        test_y_true=test_y_true,
        test_y_pred=test_y_pred,
        test_auxiliary=test_auxiliary,
        history=history,
        args=args,
        stats=stats,
    )
    print_summary(
        val_metrics, data["regression_targets"], data["plan_targets"], "Validation"
    )
    print_summary(
        test_metrics, data["regression_targets"], data["plan_targets"], "Test"
    )


def validate_args(args: argparse.Namespace) -> None:
    split_ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(ratio <= 0 for ratio in split_ratios) or not np.isclose(sum(split_ratios), 1.0):
        raise ValueError("--train-ratio, --val-ratio and --test-ratio must be positive and sum to 1.")
    if args.plan_loss_weight < 0 or args.consistency_loss_weight < 0 or args.beta_loss_weight < 0:
        raise ValueError("Loss weights must be non-negative.")
    if args.isolation_seconds < MIN_SINGLE_FILE_ISOLATION_SECONDS:
        raise ValueError(
            f"--isolation-seconds must be at least {MIN_SINGLE_FILE_ISOLATION_SECONDS:.0f}."
        )
    if not 0 < args.lr_factor < 1:
        raise ValueError("--lr-factor must be in (0,1).")


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required. Use E:\\Anaconda3\\envs\\DL\\python.exe to run this script."
        ) from exc
    return torch


def set_seed(seed: int) -> None:
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_target_loss_weights(raw_weights: str, n_targets: int) -> np.ndarray:
    values = [float(value.strip()) for value in raw_weights.split(",") if value.strip()]
    if len(values) != n_targets:
        raise ValueError(f"--target-loss-weights expected {n_targets} values, got {len(values)}.")
    if any(value <= 0 for value in values):
        raise ValueError("--target-loss-weights values must be positive.")
    return np.asarray(values, dtype=np.float32)


def build_multitask_loss(
    torch: Any, args: argparse.Namespace, stats: dict[str, np.ndarray]
) -> Any:


    target_weights = parse_target_loss_weights(
        args.target_loss_weights, len(REGRESSION_OUTPUTS)
    )

    class PhysicsPlanMultiTaskLoss(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "target_mean", torch.as_tensor(stats["target_mean"]).view(1, -1)
            )
            self.register_buffer(
                "target_std", torch.as_tensor(stats["target_std"]).view(1, -1)
            )
            self.register_buffer(
                "target_weights", torch.as_tensor(target_weights).view(-1)
            )
            self.register_buffer(
                "plan_mean", torch.as_tensor(stats["plan_target_mean"]).view(1, -1)
            )
            self.register_buffer(
                "plan_std", torch.as_tensor(stats["plan_target_std"]).view(1, -1)
            )
            self.register_buffer(
                "dock_pos_weight", torch.as_tensor(stats["plan_dock_pos_weight"]).view(1)
            )
            self.register_buffer(
                "beta_mean", torch.as_tensor(stats["energy_per_soc_mean"]).view(1)
            )
            self.register_buffer(
                "beta_std", torch.as_tensor(stats["energy_per_soc_std"]).view(1)
            )
            self.beta_weight = float(args.beta_loss_weight)
            self.consistency_weight = float(args.consistency_loss_weight)
            self.plan_weight = float(args.plan_loss_weight)


            self.tae_base_loss = (
                torch.nn.MSELoss(reduction="none")
                if args.regression_loss == "mse"
                else smooth_l1(torch, args.huber_beta)
            )
            self.other_physics_base_loss = torch.nn.MSELoss(reduction="none")

            self.plan_huber = smooth_l1(torch, args.huber_beta)

        def forward(
            self,
            outputs: dict[str, Any],
            target: Any,
            y_plan: Any,
            soc_drop_fraction: Any,
            energy_per_soc_target: Any,
            tae_valid_mask: Any,
            consistency_mask: Any,
        ) -> dict[str, Any]:
            prediction = outputs["regression"]
            plan_prediction = outputs["plan_prediction"]
            if prediction.shape != target.shape:
                raise ValueError(
                    f"Regression prediction/target shape mismatch: {prediction.shape} vs {target.shape}."
                )
            if y_plan.ndim != 2 or y_plan.shape[1] != len(PLAN_OUTPUTS):
                raise ValueError(f"y_plan must be [batch,{len(PLAN_OUTPUTS)}].")
            if plan_prediction.shape != y_plan.shape:
                raise ValueError("plan_prediction and y_plan shapes do not match.")

            pred_norm = (prediction - self.target_mean) / self.target_std
            target_norm = (target - self.target_mean) / self.target_std
            tae_element = self.tae_base_loss(pred_norm[:, 0], target_norm[:, 0])
            other_element = self.other_physics_base_loss(
                pred_norm[:, 1:], target_norm[:, 1:]
            )
            element = torch.cat([tae_element.unsqueeze(1), other_element], dim=1)


            tae_mask = tae_valid_mask.reshape(-1).to(element.dtype)
            valid_count = torch.sum(tae_mask)
            tae_loss = torch.sum(element[:, 0] * tae_mask) / valid_count.clamp_min(1.0)
            per_target = torch.stack(
                [tae_loss, *[element[:, index].mean() for index in range(1, element.shape[1])]]
            )
            supervised_loss = torch.sum(per_target * self.target_weights)

            beta_prediction_norm = (
                outputs["energy_per_soc_wh"].reshape(-1) - self.beta_mean
            ) / self.beta_std
            beta_target_norm = (
                energy_per_soc_target.reshape(-1) - self.beta_mean
            ) / self.beta_std
            beta_element = self.tae_base_loss(beta_prediction_norm, beta_target_norm)
            beta_loss = torch.sum(beta_element * tae_mask) / valid_count.clamp_min(1.0)

            energy_from_beta = outputs["energy_per_soc_wh"] * soc_drop_fraction
            energy_index = REGRESSION_OUTPUTS.index("future_energy_wh")
            energy_mean = self.target_mean[:, energy_index : energy_index + 1]
            energy_std = self.target_std[:, energy_index : energy_index + 1]
            energy_head_norm = (outputs["future_energy_wh"] - energy_mean) / energy_std
            energy_beta_norm = (energy_from_beta - energy_mean) / energy_std
            consistency_element = self.other_physics_base_loss(
                energy_head_norm, energy_beta_norm
            )
            mask = consistency_mask.to(consistency_element.dtype) * tae_valid_mask.to(
                consistency_element.dtype
            )
            consistency_loss = torch.sum(consistency_element * mask) / torch.sum(mask).clamp_min(
                1.0
            )
            physics_multitask_loss = (
                supervised_loss
                + self.beta_weight * beta_loss
                + self.consistency_weight * consistency_loss
            )

            plan_pred_norm = (
                plan_prediction[:, :PLAN_CONTINUOUS_DIM] - self.plan_mean
            ) / self.plan_std
            plan_true_norm = (y_plan[:, :PLAN_CONTINUOUS_DIM] - self.plan_mean) / self.plan_std
            plan_continuous_loss = self.plan_huber(plan_pred_norm, plan_true_norm).mean()
            dock_logit = plan_prediction[:, PLAN_CONTINUOUS_DIM]
            dock_true = y_plan[:, PLAN_CONTINUOUS_DIM]
            plan_dock_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                dock_logit,
                dock_true,
                pos_weight=self.dock_pos_weight,
            )
            plan_loss = plan_continuous_loss + plan_dock_loss
            total_loss = physics_multitask_loss + self.plan_weight * plan_loss
            return {
                "loss": total_loss,
                "physics_multitask_loss": physics_multitask_loss,
                "supervised_loss": supervised_loss,
                "beta_loss": beta_loss,
                "consistency_loss": consistency_loss,
                "plan_loss": plan_loss,
                "plan_continuous_loss": plan_continuous_loss,
                "plan_dock_loss": plan_dock_loss,
                "tae_valid_ratio": tae_mask.mean(),
                **{
                    f"{target_name}_loss": per_target[index]
                    for index, target_name in enumerate(REGRESSION_OUTPUTS)
                },
            }

    return PhysicsPlanMultiTaskLoss()


def smooth_l1(torch: Any, beta: float) -> Any:
    try:
        return torch.nn.SmoothL1Loss(beta=beta, reduction="none")
    except TypeError:
        return torch.nn.SmoothL1Loss(reduction="none")


def load_dataset(data_path: str | Path) -> dict[str, Any]:
    path = Path(data_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Run build_dataset.py first.")

    with np.load(path, allow_pickle=True) as raw:
        required_keys = {
            "x_cat",
            "x_cont",
            "x_init",
            "y_reg",
            "y_plan",
            "plan_targets",
            "future_soc_drop_fraction",
            "energy_per_soc_wh",
            "tae_valid_mask",
            "consistency_mask",
            "sample_meta",
        }
        missing = required_keys.difference(raw.files)
        if missing:
            raise ValueError(
                f"Dataset {path} is missing {sorted(missing)}. Rebuild it with the current "
                "masked-TAE build_dataset.py."
            )
        regression_targets = [str(value) for value in raw["regression_targets"].tolist()]
        plan_targets = [str(value) for value in raw["plan_targets"].tolist()]
        if regression_targets != REGRESSION_OUTPUTS:
            raise ValueError(f"Expected {REGRESSION_OUTPUTS}, got {regression_targets}.")
        if plan_targets != PLAN_OUTPUTS:
            raise ValueError(f"Expected plan targets {PLAN_OUTPUTS}, got {plan_targets}.")
        mappings = json.loads(str(raw["mappings"].reshape(-1)[0]))
        categorical_columns = [str(value) for value in raw["categorical_columns"].tolist()]
        sample_meta = json.loads(str(raw["sample_meta"].reshape(-1)[0]))
        category_cardinalities = [
            max((int(value) for value in mappings.get(column, {}).values()), default=-1) + 1
            for column in categorical_columns
        ]
        data = {
            "path": str(path),
            "x_cat": raw["x_cat"].astype(np.int64),
            "x_cont": raw["x_cont"].astype(np.float32),
            "x_init": raw["x_init"].astype(np.float32),
            "y_reg": raw["y_reg"].astype(np.float32),
            "y_plan": raw["y_plan"].astype(np.float32),
            "future_soc_drop_fraction": raw["future_soc_drop_fraction"].astype(
                np.float32
            ).reshape(-1, 1),
            "energy_per_soc_wh": raw["energy_per_soc_wh"].astype(np.float32).reshape(-1, 1),
            "tae_valid_mask": raw["tae_valid_mask"].astype(np.float32).reshape(-1, 1),
            "consistency_mask": raw["consistency_mask"].astype(np.float32).reshape(-1, 1),
            "soc_gate_threshold": float(raw["soc_gate_threshold"].reshape(-1)[0]),
            "sample_rate_hz": float(raw["sample_rate_hz"].reshape(-1)[0]),
            "history_seconds": float(raw["history_seconds"].reshape(-1)[0]),
            "future_horizon_seconds": float(raw["future_horizon_seconds"].reshape(-1)[0]),
            "stride_seconds": (
                float(raw["stride_seconds"].reshape(-1)[0])
                if "stride_seconds" in raw.files
                else float("nan")
            ),
            "nominal_energy_wh": float(raw["nominal_energy_wh"].reshape(-1)[0]),
            "categorical_columns": categorical_columns,
            "continuous_columns": [str(value) for value in raw["continuous_columns"].tolist()],
            "init_columns": [str(value) for value in raw["init_columns"].tolist()],
            "regression_targets": regression_targets,
            "plan_targets": plan_targets,
            "category_cardinalities": [max(value, 1) for value in category_cardinalities],
            "mappings": mappings,
            "sample_meta": sample_meta,
        }
    validate_loaded_dataset(data)
    return data


def validate_loaded_dataset(data: dict[str, Any]) -> None:
    n = int(data["y_reg"].shape[0])
    continuous_columns = data["continuous_columns"]
    missing_absolute_position = [
        column for column in ABSOLUTE_POSITION_COLUMNS if column not in continuous_columns
    ]
    retained_position_deltas = [
        column for column in DEPRECATED_POSITION_DELTA_COLUMNS if column in continuous_columns
    ]
    if missing_absolute_position or retained_position_deltas:
        raise ValueError(
            "Dataset must use absolute x/y/z continuous inputs instead of first-order position "
            "deltas. "
            f"Missing absolute columns: {missing_absolute_position}; "
            f"deprecated delta columns present: {retained_position_deltas}. "
            "Rebuild it with the current build_dataset.py."
        )
    expected_first_dim = [
        "x_cat",
        "x_cont",
        "x_init",
        "y_reg",
        "y_plan",
        "future_soc_drop_fraction",
        "energy_per_soc_wh",
        "tae_valid_mask",
        "consistency_mask",
    ]
    mismatched = {
        key: int(data[key].shape[0])
        for key in expected_first_dim
        if int(data[key].shape[0]) != n
    }
    if mismatched:
        raise ValueError(f"Dataset sample dimensions do not match N={n}: {mismatched}")
    if data["x_cat"].ndim != 3 or data["x_cont"].ndim != 3:
        raise ValueError("x_cat and x_cont must be 3-D [N,L,F].")
    if data["x_cont"].shape[-1] != len(continuous_columns):
        raise ValueError("x_cont feature width does not match continuous_columns metadata.")
    if data["x_cat"].shape[:2] != data["x_cont"].shape[:2]:
        raise ValueError("x_cat and x_cont history axes do not match.")
    if data["x_init"].ndim != 2:
        raise ValueError("x_init must be 2-D [N,F].")
    if data["y_reg"].shape != (n, len(REGRESSION_OUTPUTS)):
        raise ValueError(f"y_reg must have shape [N,{len(REGRESSION_OUTPUTS)}].")
    if data["y_plan"].shape != (n, len(PLAN_OUTPUTS)):
        raise ValueError(f"y_plan must have shape [N,{len(PLAN_OUTPUTS)}].")
    if len(data["sample_meta"]) != n:
        raise ValueError("sample_meta length must equal the number of samples.")
    for key in [
        "x_cont",
        "x_init",
        "y_reg",
        "y_plan",
        "future_soc_drop_fraction",
        "energy_per_soc_wh",
        "tae_valid_mask",
        "consistency_mask",
    ]:
        if not np.all(np.isfinite(data[key])):
            raise ValueError(f"{key} contains NaN or Inf.")
    if np.any(data["x_cat"] < 0):
        raise ValueError("x_cat contains negative category ids.")
    dock = data["y_plan"][:, PLAN_OUTPUTS.index("future_dock_event")]
    if not np.all(np.isin(dock, [0.0, 1.0])):
        raise ValueError("future_dock_event labels must be binary 0/1.")
    for mask_name in ["tae_valid_mask", "consistency_mask"]:
        if not np.all(np.isin(data[mask_name], [0.0, 1.0])):
            raise ValueError(f"{mask_name} must contain only binary 0/1 values.")
    if np.any(data["consistency_mask"] > data["tae_valid_mask"]):
        raise ValueError("consistency_mask cannot activate an invalid TAE sample.")


def split_train_val_test_indices(
    data: dict[str, Any],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_mode: str,
    isolation_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:

    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios <= 0) or not np.isclose(float(ratios.sum()), 1.0):
        raise ValueError("train_ratio, val_ratio and test_ratio must be positive and sum to 1.")

    n_samples = len(data["y_reg"])
    meta = data.get("sample_meta")
    if not isinstance(meta, list) or len(meta) != n_samples:
        raise ValueError("Strict split requires complete sample_meta")
    required_meta = {"source_file", "history_start", "future_end"}
    for index, row in enumerate(meta):
        missing = required_meta.difference(row)
        if missing:
            raise ValueError(f"sample_meta[{index}] is missing {sorted(missing)}.")
        parse_meta_time(row["history_start"], index, "history_start")
        parse_meta_time(row["future_end"], index, "future_end")

    source_count = len({str(row["source_file"]) for row in meta})
    if source_count > 1:
        grouping = split_mode
        groups = build_split_groups(meta, grouping)
        if len(groups) < 3 and grouping == "date":
            grouping = "file"
            groups = build_split_groups(meta, grouping)
        if len(groups) < 3:
            raise ValueError(
                "At least three chronological file/date groups are required for a leakage-free "
                "train/validation/test split."
            )
        train_idx, val_idx, test_idx, group_names = grouped_chronological_three_way_split(
            groups=groups,
            meta=meta,
            n_samples=n_samples,
            ratios=ratios,
        )
        counts = np.asarray([len(train_idx), len(val_idx), len(test_idx)], dtype=np.float64)
        return train_idx, val_idx, test_idx, {
            "strategy": f"chronological_{grouping}_groups_train_val_test",
            "source_file_count": source_count,
            "target_ratios": {
                "train": float(train_ratio),
                "validation": float(val_ratio),
                "test": float(test_ratio),
            },
            "actual_sample_ratios": {
                "train": float(counts[0] / n_samples),
                "validation": float(counts[1] / n_samples),
                "test": float(counts[2] / n_samples),
            },
            "training_groups": group_names[0],
            "validation_groups": group_names[1],
            "test_groups": group_names[2],
            "isolation_seconds": None,
            "random_window_fallback": False,
        }

    gap_seconds = max(
        float(isolation_seconds),
        MIN_SINGLE_FILE_ISOLATION_SECONDS,
        float(data["history_seconds"] + data["future_horizon_seconds"]),
    )
    train_idx, val_idx, test_idx, actual_gaps = continuous_time_three_way_split(
        meta=meta,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        isolation_seconds=gap_seconds,
    )
    counts = np.asarray([len(train_idx), len(val_idx), len(test_idx)], dtype=np.float64)
    source_file = str(meta[0]["source_file"])
    return train_idx, val_idx, test_idx, {
        "strategy": "single_file_continuous_time_blocks",
        "source_file_count": source_count,
        "target_ratios": {
            "train": float(train_ratio),
            "validation": float(val_ratio),
            "test": float(test_ratio),
        },
        "actual_sample_ratios": {
            "train": float(counts[0] / n_samples),
            "validation": float(counts[1] / n_samples),
            "test": float(counts[2] / n_samples),
        },
        "training_groups": [source_file],
        "validation_groups": [source_file],
        "test_groups": [source_file],
        "isolation_seconds": actual_gaps,
        "random_window_fallback": False,
    }


def grouped_chronological_three_way_split(
    groups: dict[str, list[int]],
    meta: list[dict[str, Any]],
    n_samples: int,
    ratios: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[list[str], list[str], list[str]]]:

    ordered_groups = sorted(
        groups,
        key=lambda key: min(
            parse_meta_time(meta[index]["history_start"], index, "history_start")
            for index in groups[key]
        ),
    )
    group_sizes = np.asarray([len(groups[key]) for key in ordered_groups], dtype=np.int64)
    cumulative = np.concatenate([[0], np.cumsum(group_sizes)])
    best: tuple[float, int, int] | None = None
    for train_end in range(1, len(ordered_groups) - 1):
        for val_end in range(train_end + 1, len(ordered_groups)):
            counts = np.asarray(
                [
                    cumulative[train_end],
                    cumulative[val_end] - cumulative[train_end],
                    cumulative[-1] - cumulative[val_end],
                ],
                dtype=np.float64,
            )
            error = float(np.sum((counts / n_samples - ratios) ** 2))
            candidate = (error, train_end, val_end)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("Unable to form three non-empty chronological groups.")

    _, train_end, val_end = best
    train_groups = ordered_groups[:train_end]
    val_groups = ordered_groups[train_end:val_end]
    test_groups = ordered_groups[val_end:]

    def indices_for(keys: list[str]) -> np.ndarray:
        return np.asarray(sorted(index for key in keys for index in groups[key]), dtype=np.int64)

    train_idx = indices_for(train_groups)
    val_idx = indices_for(val_groups)
    test_idx = indices_for(test_groups)
    if min(train_idx.size, val_idx.size, test_idx.size) == 0:
        raise ValueError("Strict grouped split produced an empty train, validation or test set.")
    return train_idx, val_idx, test_idx, (train_groups, val_groups, test_groups)


def continuous_time_three_way_split(
    meta: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    isolation_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:

    ordered = sorted(
        range(len(meta)),
        key=lambda index: parse_meta_time(meta[index]["history_start"], index, "history_start"),
    )
    target_train = max(1, int(round(len(ordered) * train_ratio)))
    target_val = max(1, int(round(len(ordered) * val_ratio)))
    target_test = max(1, int(round(len(ordered) * test_ratio)))
    if target_train + target_val + target_test > len(ordered):
        target_train = len(ordered) - target_val - target_test
    if target_train < 1:
        raise ValueError("Not enough samples for a three-way single-file time split.")

    gap_ns = int(math.ceil(isolation_seconds * 1e9))
    test_idx = np.asarray(ordered[-target_test:], dtype=np.int64)
    test_start = min(
        parse_meta_time(meta[index]["history_start"], index, "history_start")
        for index in test_idx
    )
    val_cutoff = test_start - np.timedelta64(gap_ns, "ns")
    validation_candidates = [
        index
        for index in ordered[:-target_test]
        if parse_meta_time(meta[index]["future_end"], index, "future_end") <= val_cutoff
    ]
    val_idx = np.asarray(validation_candidates[-target_val:], dtype=np.int64)
    if val_idx.size == 0:
        raise ValueError(
            "Single-file data are too short to isolate validation and test time blocks."
        )

    val_start = min(
        parse_meta_time(meta[index]["history_start"], index, "history_start")
        for index in val_idx
    )
    train_cutoff = val_start - np.timedelta64(gap_ns, "ns")
    training_candidates = [
        index
        for index in ordered
        if parse_meta_time(meta[index]["future_end"], index, "future_end") <= train_cutoff
    ]
    train_idx = np.asarray(training_candidates[:target_train], dtype=np.int64)
    if train_idx.size == 0:
        raise ValueError(
            "Single-file data are too short to isolate train, validation and test time blocks."
        )

    train_end = max(
        parse_meta_time(meta[index]["future_end"], index, "future_end") for index in train_idx
    )
    val_end = max(
        parse_meta_time(meta[index]["future_end"], index, "future_end") for index in val_idx
    )
    train_val_gap = float((val_start - train_end) / np.timedelta64(1, "s"))
    val_test_gap = float((test_start - val_end) / np.timedelta64(1, "s"))
    if min(train_val_gap, val_test_gap) + 1e-6 < isolation_seconds:
        raise RuntimeError("Internal split error: requested three-way isolation was not achieved.")
    return (
        np.sort(train_idx),
        np.sort(val_idx),
        np.sort(test_idx),
        {"train_validation": train_val_gap, "validation_test": val_test_gap},
    )


def parse_meta_time(value: Any, index: int, field: str) -> np.datetime64:
    try:
        timestamp = np.datetime64(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} in sample_meta[{index}]: {value}") from exc
    if np.isnat(timestamp):
        raise ValueError(f"Invalid {field} in sample_meta[{index}]: {value}")
    return timestamp.astype("datetime64[ns]")


def build_split_groups(meta: list[dict[str, Any]], grouping: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(meta):
        if grouping == "file":
            key = str(row["source_file"])
        elif grouping == "date":
            key = str(parse_meta_time(row["history_start"], index, "history_start").astype("datetime64[D]"))
        else:
            raise ValueError(f"Unsupported split grouping: {grouping}")
        groups.setdefault(key, []).append(index)
    return groups


def fit_stats(data: dict[str, Any], train_idx: np.ndarray) -> dict[str, np.ndarray]:

    eps = 1e-6
    x_cont = data["x_cont"][train_idx].reshape(-1, data["x_cont"].shape[-1])
    x_init = data["x_init"][train_idx]
    y_reg = data["y_reg"][train_idx]
    tae_valid = data["tae_valid_mask"][train_idx].reshape(-1) > 0.5
    if not np.any(tae_valid):
        raise ValueError(
            "The training split contains no reliable TAE labels."
        )
    beta_valid = data["energy_per_soc_wh"][train_idx].reshape(-1)[tae_valid]
    y_plan_cont = data["y_plan"][train_idx, :PLAN_CONTINUOUS_DIM]
    y_plan_dock = data["y_plan"][train_idx, PLAN_CONTINUOUS_DIM]
    dock_positive_rate = float(np.mean(y_plan_dock > 0.5))
    safe_initial_dock_rate = float(np.clip(dock_positive_rate, 1e-4, 1.0 - 1e-4))
    positives = float(np.sum(y_plan_dock > 0.5))
    negatives = float(len(y_plan_dock) - positives)
    dock_pos_weight = float(np.clip(negatives / positives, 0.25, 20.0)) if positives > 0 else 1.0

    configured_nominal_energy = float(data.get("nominal_energy_wh", float("nan")))
    if np.isfinite(configured_nominal_energy) and configured_nominal_energy > 0:
        baseline_nominal_energy = configured_nominal_energy
        baseline_source = 1.0
    else:
        soc_fraction = x_init[tae_valid, data["init_columns"].index("socFraction")].astype(
            np.float64
        )
        tae = y_reg[tae_valid, data["regression_targets"].index("tae_wh")].astype(
            np.float64
        )
        denominator = float(np.sum(soc_fraction**2))
        baseline_nominal_energy = (
            float(np.sum(soc_fraction * tae) / denominator) if denominator > 1e-12 else 0.0
        )
        baseline_source = 0.0
    target_mean = y_reg.mean(axis=0).astype(np.float32)
    target_std = np.maximum(y_reg.std(axis=0), eps).astype(np.float32)
    tae_index = data["regression_targets"].index("tae_wh")
    target_mean[tae_index] = np.mean(y_reg[tae_valid, tae_index], dtype=np.float64)
    target_std[tae_index] = max(
        float(np.std(y_reg[tae_valid, tae_index], dtype=np.float64)), eps
    )
    return {
        "cont_mean": x_cont.mean(axis=0).astype(np.float32),
        "cont_std": np.maximum(x_cont.std(axis=0), eps).astype(np.float32),
        "init_mean": x_init.mean(axis=0).astype(np.float32),
        "init_std": np.maximum(x_init.std(axis=0), eps).astype(np.float32),
        "target_mean": target_mean,
        "target_std": target_std,
        "energy_per_soc_mean": np.asarray([np.mean(beta_valid)], dtype=np.float32),
        "energy_per_soc_std": np.asarray(
            [max(float(np.std(beta_valid, dtype=np.float64)), eps)], dtype=np.float32
        ),
        "tae_valid_train_count": np.asarray([np.sum(tae_valid)], dtype=np.float32),
        "tae_valid_train_ratio": np.asarray([np.mean(tae_valid)], dtype=np.float32),
        "plan_target_mean": y_plan_cont.mean(axis=0).astype(np.float32),
        "plan_target_std": np.maximum(y_plan_cont.std(axis=0), eps).astype(np.float32),
        "plan_dock_positive_rate": np.asarray([safe_initial_dock_rate], dtype=np.float32),
        "plan_dock_pos_weight": np.asarray([dock_pos_weight], dtype=np.float32),
        "baseline_nominal_energy_wh": np.asarray([baseline_nominal_energy], dtype=np.float32),
        "baseline_nominal_energy_source": np.asarray([baseline_source], dtype=np.float32),
    }


def fit_head_initial_values(data: dict[str, Any], train_idx: np.ndarray) -> dict[str, float]:
    y_reg = data["y_reg"][train_idx]
    anchors = build_anchors(data["x_init"][train_idx], data["init_columns"])
    tae_valid = data["tae_valid_mask"][train_idx].reshape(-1) > 0.5
    beta = data["energy_per_soc_wh"][train_idx].reshape(-1)[tae_valid]
    energy = y_reg[:, data["regression_targets"].index("future_energy_wh")]
    voltage_min = y_reg[:, data["regression_targets"].index("future_min_voltage_v")]
    max_temp = y_reg[:, data["regression_targets"].index("future_max_battery_temp_c")]

    def positive_median(values: np.ndarray, floor: float = 1e-3) -> float:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values) & (values >= 0)]
        return max(float(np.median(values)) if values.size else floor, floor)

    return {
        "initial_energy_per_soc_wh": positive_median(beta),
        "initial_future_energy_wh": positive_median(energy),
        "initial_voltage_drop_v": positive_median(anchors[:, 1] - voltage_min),
        "initial_temperature_rise_c": positive_median(max_temp - anchors[:, 2]),
    }


def build_anchors(x_init_raw: np.ndarray, init_columns: list[str]) -> np.ndarray:
    required = ["socFraction", "voltage", "powerTemp"]
    missing = [column for column in required if column not in init_columns]
    if missing:
        raise ValueError(f"x_init is missing physical anchor columns: {missing}")
    return np.column_stack(
        [x_init_raw[:, init_columns.index(column)] for column in required]
    ).astype(np.float32)


class MultiTaskDataset:
    def __init__(self, data: dict[str, Any], indices: np.ndarray, stats: dict[str, np.ndarray]) -> None:
        torch = require_torch()
        x_cont = data["x_cont"][indices].copy()
        x_init_raw = data["x_init"][indices].copy()
        anchors = build_anchors(x_init_raw, data["init_columns"])
        x_cont = (x_cont - stats["cont_mean"]) / stats["cont_std"]
        x_init = (x_init_raw - stats["init_mean"]) / stats["init_std"]

        self.x_cat = torch.as_tensor(data["x_cat"][indices], dtype=torch.long)
        self.x_cont = torch.as_tensor(x_cont, dtype=torch.float32)
        self.x_init = torch.as_tensor(x_init, dtype=torch.float32)
        self.anchors = torch.as_tensor(anchors, dtype=torch.float32)
        self.y_reg = torch.as_tensor(data["y_reg"][indices], dtype=torch.float32)
        self.y_plan = torch.as_tensor(data["y_plan"][indices], dtype=torch.float32)
        self.soc_drop_fraction = torch.as_tensor(
            data["future_soc_drop_fraction"][indices], dtype=torch.float32
        )
        self.energy_per_soc = torch.as_tensor(
            data["energy_per_soc_wh"][indices], dtype=torch.float32
        )
        self.tae_valid_mask = torch.as_tensor(
            data["tae_valid_mask"][indices], dtype=torch.float32
        )
        self.consistency_mask = torch.as_tensor(
            data["consistency_mask"][indices], dtype=torch.float32
        )

    def __len__(self) -> int:
        return int(self.y_reg.shape[0])

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        return (
            self.x_cat[index],
            self.x_cont[index],
            self.x_init[index],
            self.anchors[index],
            self.y_reg[index],
            self.y_plan[index],
            self.soc_drop_fraction[index],
            self.energy_per_soc[index],
            self.tae_valid_mask[index],
            self.consistency_mask[index],
        )


def train_one_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    loss_fn: Any,
    device: Any,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    total_samples = 0
    for batch in dataloader:
        (
            x_cat,
            x_cont,
            x_init,
            anchors,
            y_reg,
            y_plan,
            soc_drop_fraction,
            energy_per_soc_target,
            tae_valid_mask,
            consistency_mask,
        ) = [value.to(device) for value in batch]
        # y_plan不传入模型，只在模型完成预测后用于监督损失。
        outputs = model(x_cat, x_cont, x_init, anchors)
        losses = loss_fn(
            outputs,
            y_reg,
            y_plan,
            soc_drop_fraction,
            energy_per_soc_target,
            tae_valid_mask,
            consistency_mask,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        if grad_clip > 0:
            require_torch().nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = x_cat.size(0)
        total_samples += batch_size
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_size
    return {key: value / max(total_samples, 1) for key, value in totals.items()}


def evaluate(
    model: Any,
    dataloader: Any,
    loss_fn: Any,
    device: Any,
    target_names: list[str],
    plan_target_names: list[str],
    stats: dict[str, np.ndarray],
) -> tuple[dict[str, float], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    torch = require_torch()
    model.eval()
    totals: dict[str, float] = {}
    total_samples = 0
    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    plan_true_parts: list[np.ndarray] = []
    plan_raw_pred_parts: list[np.ndarray] = []
    attention_parts: list[np.ndarray] = []
    soc_drop_parts: list[np.ndarray] = []
    consistency_mask_parts: list[np.ndarray] = []
    tae_valid_mask_parts: list[np.ndarray] = []
    beta_true_parts: list[np.ndarray] = []
    beta_pred_parts: list[np.ndarray] = []
    anchor_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            (
                x_cat,
                x_cont,
                x_init,
                anchors,
                y_reg,
                y_plan,
                soc_drop_fraction,
                beta_true,
                tae_valid_mask,
                consistency_mask,
            ) = [value.to(device) for value in batch]

            outputs = model(x_cat, x_cont, x_init, anchors)
            losses = loss_fn(
                outputs,
                y_reg,
                y_plan,
                soc_drop_fraction,
                beta_true,
                tae_valid_mask,
                consistency_mask,
            )
            batch_size = x_cat.size(0)
            total_samples += batch_size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_size
            true_parts.append(y_reg.cpu().numpy())
            pred_parts.append(outputs["regression"].cpu().numpy())
            plan_true_parts.append(y_plan.cpu().numpy())
            plan_raw_pred_parts.append(outputs["plan_prediction"].cpu().numpy())
            attention_parts.append(outputs["attention_weights"].cpu().numpy())
            soc_drop_parts.append(soc_drop_fraction.cpu().numpy())
            consistency_mask_parts.append(consistency_mask.cpu().numpy())
            tae_valid_mask_parts.append(tae_valid_mask.cpu().numpy())
            beta_true_parts.append(beta_true.cpu().numpy())
            beta_pred_parts.append(outputs["energy_per_soc_wh"].cpu().numpy())
            anchor_parts.append(anchors.cpu().numpy())

    if not true_parts:
        raise ValueError("Evaluation dataloader produced no samples.")
    y_true = np.concatenate(true_parts, axis=0)
    y_pred = np.concatenate(pred_parts, axis=0)
    plan_true = np.concatenate(plan_true_parts, axis=0)
    plan_raw_pred = np.concatenate(plan_raw_pred_parts, axis=0)
    attention_weights = np.concatenate(attention_parts, axis=0)
    soc_drop = np.concatenate(soc_drop_parts, axis=0)
    consistency_mask = np.concatenate(consistency_mask_parts, axis=0)
    tae_valid_mask = np.concatenate(tae_valid_mask_parts, axis=0)
    beta_true = np.concatenate(beta_true_parts, axis=0)
    beta_pred = np.concatenate(beta_pred_parts, axis=0)
    anchors = np.concatenate(anchor_parts, axis=0)

    plan_pred = plan_raw_pred.copy()
    plan_dock_logit = plan_raw_pred[:, PLAN_CONTINUOUS_DIM].copy()
    plan_dock_probability = sigmoid_numpy(plan_dock_logit)
    plan_pred[:, PLAN_CONTINUOUS_DIM] = plan_dock_probability
    energy_from_beta = beta_pred * soc_drop
    energy_index = target_names.index("future_energy_wh")
    energy_pred = y_pred[:, energy_index : energy_index + 1]
    tae_valid = tae_valid_mask.reshape(-1) > 0.5

    metrics = {key: value / max(total_samples, 1) for key, value in totals.items()}
    for target_index, target_name in enumerate(target_names):
        metric_mask = tae_valid if target_name == "tae_wh" else np.ones(len(y_true), dtype=bool)
        metrics.update(
            prefix_metrics(
                target_name,
                y_true[metric_mask, target_index],
                y_pred[metric_mask, target_index],
            )
        )
    for target_index, target_name in enumerate(plan_target_names):
        metrics.update(
            prefix_metrics(target_name, plan_true[:, target_index], plan_pred[:, target_index])
        )
    metrics.update(
        binary_metrics(
            plan_true[:, PLAN_CONTINUOUS_DIM],
            plan_dock_probability,
            prefix="future_dock_event",
        )
    )
    metrics.update(
        prefix_metrics(
            "energy_per_soc_wh",
            beta_true.reshape(-1)[tae_valid],
            beta_pred.reshape(-1)[tae_valid],
        )
    )
    tae_index = target_names.index("tae_wh")
    nominal_energy_wh = float(np.asarray(stats["baseline_nominal_energy_wh"]).reshape(-1)[0])
    soc_baseline = anchors[:, 0] * nominal_energy_wh
    metrics.update(
        prefix_metrics(
            "soc_energy_baseline_tae_wh",
            y_true[tae_valid, tae_index],
            soc_baseline[tae_valid],
        )
    )
    metrics["tae_rmse_improvement_over_soc_baseline_percent"] = float(
        (metrics["soc_energy_baseline_tae_wh_rmse"] - metrics["tae_wh_rmse"])
        / max(metrics["soc_energy_baseline_tae_wh_rmse"], 1e-12)
        * 100.0
    )
    metrics["soc_baseline_nominal_energy_wh"] = nominal_energy_wh
    metrics["tae_valid_sample_count"] = float(np.sum(tae_valid))
    metrics["tae_valid_sample_ratio"] = float(np.mean(tae_valid))
    active = consistency_mask.reshape(-1) > 0.5
    metrics["energy_consistency_active_ratio"] = float(np.mean(active))
    metrics["energy_consistency_mae_wh"] = (
        float(
            np.mean(
                np.abs(
                    energy_pred.reshape(-1)[active] - energy_from_beta.reshape(-1)[active]
                )
            )
        )
        if np.any(active)
        else float("nan")
    )
    metrics["voltage_constraint_violations"] = float(
        np.sum(y_pred[:, 2] > anchors[:, 1] + 1e-6)
    )
    metrics["temperature_constraint_violations"] = float(
        np.sum(y_pred[:, 3] < anchors[:, 2] - 1e-6)
    )
    return y_metrics_float(metrics), y_true, y_pred, {
        "plan_true": plan_true,
        "plan_pred": plan_pred,
        "plan_raw_pred": plan_raw_pred,
        "plan_dock_logit": plan_dock_logit.reshape(-1, 1),
        "plan_dock_probability": plan_dock_probability.reshape(-1, 1),
        "attention_weights": attention_weights,
        "soc_drop_fraction": soc_drop,
        "tae_valid_mask": tae_valid_mask,
        "consistency_mask": consistency_mask,
        "energy_per_soc_true": beta_true,
        "energy_per_soc_pred": beta_pred,
        "energy_from_beta": energy_from_beta,
        "anchors": anchors,
        "soc_baseline_tae_wh": soc_baseline.reshape(-1, 1),
    }


def sigmoid_numpy(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output.astype(np.float32)


def prefix_metrics(prefix: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return {
            f"{prefix}_mae": float("nan"),
            f"{prefix}_rmse": float("nan"),
            f"{prefix}_r2": float("nan"),
            f"{prefix}_wmape_percent": float("nan"),
            f"{prefix}_sample_count": 0.0,
        }
    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    denominator = float(np.sum(np.abs(y_true)))
    wmape = (
        float(np.sum(np.abs(residual)) / denominator * 100.0)
        if denominator > 1e-12
        else float("nan")
    )
    return {
        f"{prefix}_mae": mae,
        f"{prefix}_rmse": rmse,
        f"{prefix}_r2": r2,
        f"{prefix}_wmape_percent": wmape,
        f"{prefix}_sample_count": float(y_true.size),
    }


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, prefix: str) -> dict[str, float]:
    true = np.asarray(y_true).reshape(-1) > 0.5
    pred = np.asarray(probability).reshape(-1) >= 0.5
    tp = int(np.sum(true & pred))
    fp = int(np.sum(~true & pred))
    fn = int(np.sum(true & ~pred))
    accuracy = float(np.mean(true == pred))
    recall = float(tp / (tp + fn)) if tp + fn > 0 else float("nan")
    precision = float(tp / (tp + fp)) if tp + fp > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if np.isfinite(recall) and precision + recall > 0 else 0.0
    return {
        f"{prefix}_accuracy": accuracy,
        f"{prefix}_recall": recall,
        f"{prefix}_precision": precision,
        f"{prefix}_f1": f1,
        f"{prefix}_positive_count": float(np.sum(true)),
    }


def y_metrics_float(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items()}


def make_checkpoint(
    model_state: dict[str, Any],
    epoch: int,
    args: argparse.Namespace,
    data: dict[str, Any],
    stats: dict[str, np.ndarray],
    head_initial_values: dict[str, float],
    split_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_version": "soc_fraction_tae_predicted_future_plan_v3",
        "epoch": epoch,
        "model_state_dict": model_state,
        "args": vars(args),
        "stats": numpy_stats_to_json(stats),
        "target_columns": data["regression_targets"],
        "plan_target_columns": data["plan_targets"],
        "category_cardinalities": data["category_cardinalities"],
        "continuous_columns": data["continuous_columns"],
        "init_columns": data["init_columns"],
        "anchor_columns": ["socFraction", "voltage", "powerTemp"],
        "head_initial_values": head_initial_values,
        "split_info": split_info,
        "leakage_policy": (
            "The model forward path accepts only x_cat, x_cont, x_init and current anchors. "
            "True y_plan is used only after forward for supervision and metrics."
        ),
    }


def save_outputs(
    output_dir: Path,
    data: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    split_info: dict[str, Any],
    val_metrics: dict[str, float],
    val_y_true: np.ndarray,
    val_y_pred: np.ndarray,
    val_auxiliary: dict[str, np.ndarray],
    test_metrics: dict[str, float],
    test_y_true: np.ndarray,
    test_y_pred: np.ndarray,
    test_auxiliary: dict[str, np.ndarray],
    history: list[dict[str, float]],
    args: argparse.Namespace,
    stats: dict[str, np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "validation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2, ensure_ascii=False)
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    with (output_dir / "training_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_version": "soc_fraction_tae_predicted_future_plan_v3",
                "args": vars(args),
                "data_path": data["path"],
                "num_samples": int(len(data["y_reg"])),
                "num_train_samples": int(len(train_idx)),
                "num_val_samples": int(len(val_idx)),
                "num_test_samples": int(len(test_idx)),
                "target_names": data["regression_targets"],
                "plan_target_names": data["plan_targets"],
                "split": split_info,
                "loss_definition": {
                    "target_weights": parse_target_loss_weights(
                        args.target_loss_weights, len(data["regression_targets"])
                    ).tolist(),
                    "beta_loss_weight": args.beta_loss_weight,
                    "consistency_weight": args.consistency_loss_weight,
                    "plan_loss_weight": args.plan_loss_weight,
                    "physics_regression_losses": {
                        "tae_wh": (
                            "MSE on train-standardized target"
                            if args.regression_loss == "mse"
                            else "SmoothL1/Huber on train-standardized target"
                        ),
                        "future_energy_wh": "MSE on train-standardized target",
                        "future_min_voltage_v": "MSE on train-standardized target",
                        "future_max_battery_temp_c": "MSE on train-standardized target",
                    },
                    "plan_continuous_loss": "Huber on train-standardized targets",
                    "plan_dock_loss": "BCEWithLogits using training-only positive weight",
                    "tae_valid_mask": (
                        "TAE and unit-SOC-energy losses use only samples whose SOC drop "
                        "strictly exceeds the dataset threshold; all other task losses use "
                        "all samples."
                    ),
                },
                "selection_definition": {
                    "metric": args.selection_metric,
                    "normalized_tae_weight": 0.7,
                    "normalized_future_energy_weight": 0.3,
                    "normalization": "each RMSE divided by training target_std",
                    "used_for": [
                        "checkpoint_selection",
                        "lr_scheduler",
                        "early_stopping",
                    ],
                },
                "leakage_policy": {
                    "future_raw_as_input": False,
                    "physics_heads_use": "present context only; plan branch is auxiliary",
                    "true_y_plan_use": "supervision and evaluation metrics only",
                    "test_set_use": (
                        "final evaluation only; never checkpoint selection or early stopping"
                        if int(split_info.get("train_test_overlap_count", 0)) == 0
                        else (
                            "final evaluation only, but intentionally contains exact training-window "
                            "duplicates for a leakage experiment"
                        )
                    ),
                },
                "time_definition": {
                    "sample_rate_hz": data["sample_rate_hz"],
                    "history_seconds": data["history_seconds"],
                    "future_horizon_seconds": data["future_horizon_seconds"],
                    "stride_seconds": data["stride_seconds"],
                },
                "soc_gate_threshold": data["soc_gate_threshold"],
                "soc_baseline_nominal_energy_wh": float(stats["baseline_nominal_energy_wh"][0]),
                "soc_baseline_energy_source": (
                    "battery_specification"
                    if float(stats["baseline_nominal_energy_source"][0]) > 0.5
                    else "least_squares_training_fit"
                ),
                "stats": numpy_stats_to_json(stats),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    save_history_csv(output_dir / "training_history.csv", history)
    np.savez_compressed(
        output_dir / "split_indices.npz",
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
    )
    save_predictions_csv(
        output_dir / "validation_predictions.csv",
        data=data,
        val_idx=val_idx,
        y_true=val_y_true,
        y_pred=val_y_pred,
        auxiliary=val_auxiliary,
    )
    save_predictions_csv(
        output_dir / "test_predictions.csv",
        data=data,
        val_idx=test_idx,
        y_true=test_y_true,
        y_pred=test_y_pred,
        auxiliary=test_auxiliary,
    )
    np.savez_compressed(
        output_dir / "validation_attention_weights.npz",
        original_index=val_idx.astype(np.int64),
        attention_weights=val_auxiliary["attention_weights"].astype(np.float32),
        history_seconds=np.asarray([data["history_seconds"]], dtype=np.float32),
    )
    np.savez_compressed(
        output_dir / "test_attention_weights.npz",
        original_index=test_idx.astype(np.int64),
        attention_weights=test_auxiliary["attention_weights"].astype(np.float32),
        history_seconds=np.asarray([data["history_seconds"]], dtype=np.float32),
    )
    save_soc_condition_comparison(
        output_dir / "soc_condition_comparison.csv",
        data,
        val_idx,
        val_y_true,
        val_y_pred,
        val_auxiliary,
    )
    save_soc_condition_comparison(
        output_dir / "test_soc_condition_comparison.csv",
        data,
        test_idx,
        test_y_true,
        test_y_pred,
        test_auxiliary,
    )
    plot_outputs(
        output_dir,
        data["regression_targets"],
        data["plan_targets"],
        val_y_true,
        val_y_pred,
        val_auxiliary["plan_true"],
        val_auxiliary["plan_pred"],
        val_auxiliary["tae_valid_mask"],
        history,
    )
    test_plot_dir = output_dir / "test_plots"
    test_plot_dir.mkdir(parents=True, exist_ok=True)
    plot_outputs(
        test_plot_dir,
        data["regression_targets"],
        data["plan_targets"],
        test_y_true,
        test_y_pred,
        test_auxiliary["plan_true"],
        test_auxiliary["plan_pred"],
        test_auxiliary["tae_valid_mask"],
        [],
    )


def save_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    fieldnames = list(history[0])
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_predictions_csv(
    path: Path,
    data: dict[str, Any],
    val_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    auxiliary: dict[str, np.ndarray],
) -> None:
    target_names = data["regression_targets"]
    plan_target_names = data["plan_targets"]
    fieldnames = [
        "sample_index",
        "original_index",
        "source_file",
        "history_end",
        "future_end",
        "current_soc_fraction",
        "future_soc_drop_fraction",
        "tae_valid_mask",
        "energy_per_soc_wh_true",
        "energy_per_soc_wh_pred",
        "future_energy_from_beta_wh",
        "soc_energy_baseline_tae_wh",
        "consistency_mask",
    ]
    for target_name in target_names:
        fieldnames.extend(
            [f"{target_name}_true", f"{target_name}_pred", f"{target_name}_residual"]
        )
    for target_name in plan_target_names[:-1]:
        fieldnames.extend(
            [f"{target_name}_true", f"{target_name}_pred", f"{target_name}_residual"]
        )
    fieldnames.extend(
        [
            "future_dock_event_true",
            "future_dock_event_logit",
            "future_dock_event_probability",
            "future_dock_event_predicted_class",
        ]
    )
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample_index, original_index in enumerate(val_idx):
            meta = data["sample_meta"][int(original_index)]
            soc_drop = float(auxiliary["soc_drop_fraction"][sample_index, 0])
            row: dict[str, Any] = {
                "sample_index": sample_index,
                "original_index": int(original_index),
                "source_file": str(meta["source_file"]),
                "history_end": str(meta["history_end"]),
                "future_end": str(meta["future_end"]),
                "current_soc_fraction": float(auxiliary["anchors"][sample_index, 0]),
                "future_soc_drop_fraction": soc_drop,
                "tae_valid_mask": int(
                    auxiliary["tae_valid_mask"][sample_index, 0] > 0.5
                ),
                "energy_per_soc_wh_true": float(
                    auxiliary["energy_per_soc_true"][sample_index, 0]
                ),
                "energy_per_soc_wh_pred": float(
                    auxiliary["energy_per_soc_pred"][sample_index, 0]
                ),
                "future_energy_from_beta_wh": float(
                    auxiliary["energy_from_beta"][sample_index, 0]
                ),
                "soc_energy_baseline_tae_wh": float(
                    auxiliary["soc_baseline_tae_wh"][sample_index, 0]
                ),
                "consistency_mask": int(
                    auxiliary["consistency_mask"][sample_index, 0] > 0.5
                ),
            }
            for target_index, target_name in enumerate(target_names):
                true_value = float(y_true[sample_index, target_index])
                pred_value = float(y_pred[sample_index, target_index])
                row[f"{target_name}_true"] = true_value
                row[f"{target_name}_pred"] = pred_value
                row[f"{target_name}_residual"] = pred_value - true_value
            for target_index, target_name in enumerate(plan_target_names[:-1]):
                true_value = float(auxiliary["plan_true"][sample_index, target_index])
                pred_value = float(auxiliary["plan_pred"][sample_index, target_index])
                row[f"{target_name}_true"] = true_value
                row[f"{target_name}_pred"] = pred_value
                row[f"{target_name}_residual"] = pred_value - true_value
            dock_true = float(auxiliary["plan_true"][sample_index, PLAN_CONTINUOUS_DIM])
            dock_logit = float(auxiliary["plan_dock_logit"][sample_index, 0])
            dock_probability = float(auxiliary["plan_dock_probability"][sample_index, 0])
            row.update(
                {
                    "future_dock_event_true": dock_true,
                    "future_dock_event_logit": dock_logit,
                    "future_dock_event_probability": dock_probability,
                    "future_dock_event_predicted_class": int(dock_probability >= 0.5),
                }
            )
            writer.writerow(row)


def save_soc_condition_comparison(
    path: Path,
    data: dict[str, Any],
    val_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    auxiliary: dict[str, np.ndarray],
) -> None:
    try:
        import pandas as pd
    except ImportError:
        return

    soc = auxiliary["anchors"][:, 0].reshape(-1)
    speed = data["x_init"][val_idx, data["init_columns"].index("carSpeed")]
    battery_temp = data["x_init"][val_idx, data["init_columns"].index("powerTemp")]
    tae_index = data["regression_targets"].index("tae_wh")
    true_tae = y_true[:, tae_index]
    pred_tae = y_pred[:, tae_index]
    baseline_tae = auxiliary["soc_baseline_tae_wh"].reshape(-1)
    tae_valid = auxiliary["tae_valid_mask"].reshape(-1) > 0.5
    soc_bin_lower = np.floor(np.clip(soc, 0.0, 0.999999) * 10.0) / 10.0
    soc_bins = np.asarray([f"[{lower:.1f},{lower + 0.1:.1f})" for lower in soc_bin_lower])
    speed_band = np.select(
        [np.abs(speed) <= 0.05, np.abs(speed) <= 0.2, np.abs(speed) <= 0.5],
        ["stationary", "low", "medium"],
        default="high",
    )
    temp_band = np.select(
        [battery_temp < 35.0, battery_temp < 45.0],
        ["<35C", "35-45C"],
        default=">=45C",
    )

    def decoded_category(column: str) -> np.ndarray:
        cat_index = data["categorical_columns"].index(column)
        ids = data["x_cat"][val_idx, -1, cat_index]
        reverse = {
            int(value): str(key) for key, value in data["mappings"].get(column, {}).items()
        }
        return np.asarray([reverse.get(int(value), str(int(value))) for value in ids])

    conditions = {
        "speed": speed_band,
        "battery_temperature": temp_band,
        "gait": decoded_category("gait"),
        "phase": decoded_category("phase"),
    }
    rows: list[dict[str, Any]] = []
    for condition_type, values in conditions.items():
        for soc_bin in np.unique(soc_bins):
            in_soc_bin = soc_bins == soc_bin
            for condition_value in np.unique(values[in_soc_bin]):
                mask = in_soc_bin & (values == condition_value) & tae_valid
                if not np.any(mask):
                    continue
                rows.append(
                    {
                        "soc_bin": soc_bin,
                        "condition_type": condition_type,
                        "condition_value": str(condition_value),
                        "count": int(np.sum(mask)),
                        "true_tae_mean_wh": float(np.mean(true_tae[mask])),
                        "model_tae_mean_wh": float(np.mean(pred_tae[mask])),
                        "soc_baseline_tae_mean_wh": float(np.mean(baseline_tae[mask])),
                        "model_mae_wh": float(np.mean(np.abs(pred_tae[mask] - true_tae[mask]))),
                        "soc_baseline_mae_wh": float(
                            np.mean(np.abs(baseline_tae[mask] - true_tae[mask]))
                        ),
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def plot_outputs(
    output_dir: Path,
    target_names: list[str],
    plan_target_names: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    plan_true: np.ndarray,
    plan_pred: np.ndarray,
    tae_valid_mask: np.ndarray,
    history: list[dict[str, float]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    def plot_target(name: str, true: np.ndarray, pred: np.ndarray, prefix: str = "") -> None:
        if true.size == 0:
            return
        low = float(min(np.min(true), np.min(pred)))
        high = float(max(np.max(true), np.max(pred)))
        if np.isclose(low, high):
            high = low + 1.0
        plt.figure(figsize=(6, 5))
        plt.scatter(true, pred, s=9, alpha=0.35)
        plt.plot([low, high], [low, high], color="red", linewidth=1)
        plt.xlabel(f"True {name}")
        plt.ylabel(f"Predicted {name}")
        plt.title(f"{name}: prediction vs true")
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}{name}_prediction_scatter.png", dpi=200)
        plt.close()

    for target_index, target_name in enumerate(target_names):
        metric_mask = (
            tae_valid_mask.reshape(-1) > 0.5
            if target_name == "tae_wh"
            else np.ones(len(y_true), dtype=bool)
        )
        plot_target(
            target_name,
            y_true[metric_mask, target_index],
            y_pred[metric_mask, target_index],
        )
    for target_index, target_name in enumerate(plan_target_names[:-1]):
        plot_target(
            target_name,
            plan_true[:, target_index],
            plan_pred[:, target_index],
            prefix="plan_",
        )

    if history:
        plt.figure(figsize=(7, 4))
        plt.plot(
            [row["epoch"] for row in history],
            [row["train_loss"] for row in history],
            label="train",
        )
        plt.plot(
            [row["epoch"] for row in history],
            [row["val_loss"] for row in history],
            label="validation",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Plan + physics multi-task loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "training_history.png", dpi=200)
        plt.close()


def numpy_stats_to_json(stats: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {key: np.asarray(value).astype(float).tolist() for key, value in stats.items()}


def print_summary(
    metrics: dict[str, float],
    target_names: list[str],
    plan_target_names: list[str],
    split_name: str = "Validation",
) -> None:
    print(f"\n{split_name} physics metrics")
    print(
        "TAE valid samples: "
        f"{int(metrics['tae_valid_sample_count'])} "
        f"({metrics['tae_valid_sample_ratio']:.2%}); TAE metrics exclude invalid placeholders."
    )
    for target in target_names:
        print(
            f"{target}: MAE={metrics[f'{target}_mae']:.6f}, "
            f"RMSE={metrics[f'{target}_rmse']:.6f}, "
            f"R2={metrics[f'{target}_r2']:.6f}, "
            f"wMAPE={metrics[f'{target}_wmape_percent']:.3f}%"
        )
    print(f"\n{split_name} future-plan metrics")
    for target in plan_target_names:
        print(
            f"{target}: MAE={metrics[f'{target}_mae']:.6f}, "
            f"RMSE={metrics[f'{target}_rmse']:.6f}, "
            f"R2={metrics[f'{target}_r2']:.6f}"
        )
    print(
        "future_dock_event classification: "
        f"accuracy={metrics['future_dock_event_accuracy']:.4f}, "
        f"recall={metrics['future_dock_event_recall']:.4f}, "
        f"F1={metrics['future_dock_event_f1']:.4f}"
    )
    print(
        "physical consistency: "
        f"MAE={metrics['energy_consistency_mae_wh']:.6f} Wh, "
        f"voltage violations={int(metrics['voltage_constraint_violations'])}, "
        f"temperature violations={int(metrics['temperature_constraint_violations'])}"
    )
    print(
        "SOC-only energy baseline: "
        f"E_nom={metrics['soc_baseline_nominal_energy_wh']:.3f} Wh, "
        f"RMSE={metrics['soc_energy_baseline_tae_wh_rmse']:.6f} Wh, "
        f"model improvement={metrics['tae_rmse_improvement_over_soc_baseline_percent']:.3f}%"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
