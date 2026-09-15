from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent



CATEGORICAL_COLUMNS = ["gait", "pavement", "phase", "model", "controlMode"]


CONTINUOUS_COLUMNS = [
    "socFraction",
    "voltage",
    "current",
    "dischargePower",
    "powerTemp",
    "bodyMaxTemp",
    "carSpeed",
    "odometryDelta",
    "x",
    "y",
    "z",
    "carDirectionSin",
    "carDirectionCos",
    "carDirectionDelta",
    "isCharging",
    "isOnDock",
    "voltageDiff30",
    "powerTempDiff30",
    "dischargePowerMean60",
    "dischargePowerStd60",
    "voltageRollingMin60",
]


INIT_COLUMNS = [
    "socFraction",
    "voltage",
    "current",
    "powerTemp",
    "dischargePowerMean30",
    "carSpeed",
    "carDirectionSin",
    "carDirectionCos",
    "isCharging",
    "isOnDock",
]

REGRESSION_TARGETS = [
    "tae_wh",
    "future_energy_wh",
    "future_min_voltage_v",
    "future_max_battery_temp_c",
]


PLAN_TARGETS = [
    "future_delta_x",
    "future_delta_y",
    "future_delta_z",
    "future_path_distance",
    "future_mean_abs_speed",
    "future_moving_ratio",
    "future_absolute_yaw_change",
    "future_model4_ratio",
    "future_dock_event",
]

RAW_REQUIRED_COLUMNS = {
    "timestamp",
    "gait",
    "pavement",
    "controlMode",
    "chargeState",
    "onDockState",
    "returnState",
    "model",
    "current",
    "carSpeed",
    "voltage",
    "power",
    "powerTemp",
    "odometry",
    "x",
    "y",
    "z",
    "carDirection",
}

BODY_TEMP_COLUMNS = [
    "boxTemp",
    "leftBackHip",
    "leftBackKnee",
    "leftBackSway",
    "leftFrontHip",
    "leftFrontKnee",
    "leftFrontSway",
    "rightBackHip",
    "rightBackSway",
    "rightFrontHip",
    "rightFrontKnee",
    "rightFrontSway",
    "flTire",
    "frTire",
    "rlTire",
    "rrTire",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build the time-defined TAE dataset.")
    parser.add_argument("--raw-dir", type=str, default=None)


    parser.add_argument("--pattern", type=str, default="ros_msg.*.csv")
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(SCRIPT_DIR / "data" / "robot.npz"),
    )
    parser.add_argument("--sample-rate-hz", type=float, default=1.0)
    parser.add_argument("--history-seconds", type=float, default=90.0)
    parser.add_argument("--future-horizon-seconds", type=float, default=300.0)
    parser.add_argument("--stride-seconds", type=float, default=10.0)
    parser.add_argument("--max-raw-gap-seconds", type=float, default=5.0)
    parser.add_argument("--max-files", type=int, default=None)
    movement_group = parser.add_mutually_exclusive_group()
    movement_group.add_argument(
        "--moving-only",
        dest="moving_only",
        action="store_true",
        help="Retain only windows whose historical mean absolute speed exceeds the threshold (default).",
    )
    movement_group.add_argument(
        "--include-stationary",
        dest="moving_only",
        action="store_false",
        help="Retain stationary history windows as well as moving windows.",
    )
    parser.set_defaults(moving_only=True)
    parser.add_argument("--speed-threshold", type=float, default=1e-3)
    parser.add_argument("--current-discharge-sign", type=float, default=-1.0)
    parser.add_argument(
        "--min-soc-drop-fraction",
        type=float,
        default=0.015,
        help=(
            "Minimum delta_s required for reliable TAE/beta supervision."
        ),
    )
    parser.add_argument(
        "--soc-noise-std-fraction",
        type=float,
        default=0.005,
        help="SOC measurement noise sigma on the [0,1] scale.",
    )
    parser.add_argument(
        "--soc-resolution-fraction",
        type=float,
        default=0.01,
        help="SOC resolution on the [0,1] scale; integer-percent SOC has resolution 0.01.",
    )
    parser.add_argument(
        "--nominal-energy-wh",
        type=float,
        default=None,
        help="Optional rated battery energy.",
    )
    stop_group = parser.add_mutually_exclusive_group()
    stop_group.add_argument(
        "--include-operational-stop-windows",
        dest="include_operational_stop_windows",
        action="store_true",
        help="future_dock_event。",
    )
    stop_group.add_argument(
        "--exclude-operational-stop-windows",
        dest="include_operational_stop_windows",
        action="store_false",
        help="dock。",
    )
    parser.set_defaults(include_operational_stop_windows=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir) if args.raw_dir else find_default_raw_dir(args.pattern)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_dataset(
        raw_dir=raw_dir,
        pattern=args.pattern,
        output_path=output_path,
        sample_rate_hz=args.sample_rate_hz,
        history_seconds=args.history_seconds,
        future_horizon_seconds=args.future_horizon_seconds,
        stride_seconds=args.stride_seconds,
        max_raw_gap_seconds=args.max_raw_gap_seconds,
        max_files=args.max_files,
        moving_only=args.moving_only,
        speed_threshold=args.speed_threshold,
        current_discharge_sign=args.current_discharge_sign,
        min_soc_drop_fraction=args.min_soc_drop_fraction,
        soc_noise_std_fraction=args.soc_noise_std_fraction,
        soc_resolution_fraction=args.soc_resolution_fraction,
        nominal_energy_wh=args.nominal_energy_wh,
        include_operational_stop_windows=args.include_operational_stop_windows,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def find_default_raw_dir(pattern: str) -> Path:
    candidates: list[Path] = []
    for directory_name in ["2026-01-04-CSV", "2026-01-05-CSV"]:
        candidates.extend(sorted(Path("D:/").glob(f"*/{directory_name}")))
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob(pattern)):
            return candidate
    matches = sorted(Path("D:/").glob(f"*/**/{pattern}"))
    if matches:
        return matches[0].parent
    raise FileNotFoundError("Could not find raw logs. Pass --raw-dir explicitly.")


def samples_for_seconds(seconds: float, sample_rate_hz: float, name: str) -> int:
    value = int(round(seconds * sample_rate_hz))
    if value < 1 or not np.isclose(value, seconds * sample_rate_hz, atol=1e-6):
        raise ValueError(f"{name} * sample_rate_hz must be a positive integer, got {seconds * sample_rate_hz}.")
    return value


def build_dataset(
    raw_dir: Path,
    pattern: str,
    output_path: Path,
    sample_rate_hz: float,
    history_seconds: float,
    future_horizon_seconds: float,
    stride_seconds: float,
    max_raw_gap_seconds: float,
    max_files: int | None,
    moving_only: bool,
    speed_threshold: float,
    current_discharge_sign: float,
    min_soc_drop_fraction: float,
    soc_noise_std_fraction: float,
    soc_resolution_fraction: float,
    nominal_energy_wh: float | None,
    include_operational_stop_windows: bool,
) -> dict[str, Any]:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    if max_raw_gap_seconds <= 0:
        raise ValueError("max_raw_gap_seconds must be positive.")
    if min_soc_drop_fraction < 0 or soc_noise_std_fraction < 0 or soc_resolution_fraction < 0:
        raise ValueError("SOC thresholds must be non-negative fractions.")
    if nominal_energy_wh is not None and nominal_energy_wh <= 0:
        raise ValueError("nominal_energy_wh must be positive when provided.")

    history_len = samples_for_seconds(history_seconds, sample_rate_hz, "history_seconds")
    future_len = samples_for_seconds(future_horizon_seconds, sample_rate_hz, "future_horizon_seconds")
    stride_len = samples_for_seconds(stride_seconds, sample_rate_hz, "stride_seconds")
    soc_gate_threshold = max(3.0 * soc_noise_std_fraction, soc_resolution_fraction)

    files = sorted(raw_dir.glob(pattern))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No files matched: {raw_dir / pattern}")

    x_cat_samples: list[np.ndarray] = []
    x_cont_samples: list[np.ndarray] = []
    x_init_samples: list[np.ndarray] = []
    y_reg_samples: list[list[float]] = []
    y_plan_samples: list[list[float]] = []
    soc_drop_samples: list[float] = []
    energy_per_soc_samples: list[float] = []
    tae_valid_mask_samples: list[float] = []
    consistency_mask_samples: list[float] = []
    sample_meta: list[dict[str, Any]] = []
    mappings: dict[str, dict[str, int]] = {column: {} for column in CATEGORICAL_COLUMNS}
    tae_invalid_count = 0
    skipped_operational_stop = 0
    skipped_invalid_plan = 0
    num_continuous_segments = 0
    min_rows = history_len + future_len

    for file_index, file_path in enumerate(files, start=1):
        print(f"[{file_index}/{len(files)}] Reading and resampling {file_path.name}")
        segments = read_and_resample_log(
            file_path=file_path,
            sample_rate_hz=sample_rate_hz,
            max_raw_gap_seconds=max_raw_gap_seconds,
            current_discharge_sign=current_discharge_sign,
        )
        num_continuous_segments += len(segments)
        for segment_index, df in enumerate(segments):
            if len(df) < min_rows:
                continue
            for column in CATEGORICAL_COLUMNS:
                df[f"{column}Id"] = map_category_series(df[column], mappings[column])
            cat_array = df[[f"{column}Id" for column in CATEGORICAL_COLUMNS]].to_numpy(dtype=np.int64)
            cont_array = df[CONTINUOUS_COLUMNS].to_numpy(dtype=np.float32)
            init_array = df[INIT_COLUMNS].to_numpy(dtype=np.float32)
            timestamps_ns = df["timestamp"].astype("int64").to_numpy()
            timestamps = df["timestamp"].to_numpy()

            for start in range(0, len(df) - min_rows + 1, stride_len):
                history_end = start + history_len
                future_end = history_end + future_len
                anchor_index = history_end - 1
                history_slice = slice(start, history_end)
                future_slice = slice(history_end, future_end)
                if moving_only:
                    history_speed = np.abs(df["carSpeed"].iloc[history_slice].to_numpy(dtype=float))
                    if float(np.nanmean(history_speed)) <= speed_threshold:
                        continue

                future = df.iloc[future_slice]
                if not include_operational_stop_windows and has_operational_stop(future):
                    skipped_operational_stop += 1
                    continue

                current_row = df.iloc[anchor_index]
                target = compute_targets(
                    future=future,
                    future_timestamps_ns=timestamps_ns[future_slice],
                    current_timestamp_ns=int(timestamps_ns[anchor_index]),
                    current_soc_fraction=float(current_row["socFraction"]),
                    current_voltage_v=float(current_row["voltage"]),
                    current_battery_temp_c=float(current_row["powerTemp"]),
                    current_discharge_power_w=float(current_row["dischargePower"]),
                    min_soc_drop_fraction=min_soc_drop_fraction,
                    soc_gate_threshold=soc_gate_threshold,
                    nominal_energy_wh=nominal_energy_wh,
                )
                if target is None:


                    continue
                plan_target = compute_plan_targets(current_row=current_row, future=future)
                if plan_target is None:
                    skipped_invalid_plan += 1
                    continue

                x_cat_samples.append(cat_array[history_slice])
                x_cont_samples.append(cont_array[history_slice])
                x_init_samples.append(init_array[anchor_index])
                y_reg_samples.append([target[name] for name in REGRESSION_TARGETS])
                y_plan_samples.append([plan_target[name] for name in PLAN_TARGETS])
                soc_drop_samples.append(target["future_soc_drop_fraction"])
                energy_per_soc_samples.append(target["energy_per_soc_wh"])
                tae_valid_mask_samples.append(target["tae_valid_mask"])
                consistency_mask_samples.append(target["consistency_mask"])
                tae_invalid_count += int(target["tae_valid_mask"] < 0.5)
                meta_row: dict[str, Any] = {
                    "source_file": file_path.name,
                    "continuous_segment": int(segment_index),
                    "history_start": str(timestamps[start]),
                    "history_end": str(timestamps[anchor_index]),
                    "future_start": str(timestamps[history_end]),
                    "future_end": str(timestamps[future_end - 1]),
                    "current_soc_fraction": target["current_soc_fraction"],
                    "future_end_soc_fraction": target["future_end_soc_fraction"],
                    "future_soc_drop_fraction": target["future_soc_drop_fraction"],
                    "energy_per_soc_wh": target["energy_per_soc_wh"],
                    "tae_valid_mask": int(target["tae_valid_mask"]),
                    "consistency_mask": int(target["consistency_mask"]),
                    "future_dock_event": int(plan_target["future_dock_event"]),
                }
                if nominal_energy_wh is not None:
                    meta_row["soc_energy_wh"] = target["soc_energy_wh"]
                    meta_row["task_correction_factor"] = target["task_correction_factor"]
                sample_meta.append(meta_row)

    if not x_cat_samples:
        raise ValueError(
            "No samples were created. Check continuous-run duration, motion/operational-stop "
            "filters, and whether the always-valid future labels are finite."
        )

    x_cat = np.stack(x_cat_samples).astype(np.int64)
    x_cont = np.stack(x_cont_samples).astype(np.float32)
    x_init = np.stack(x_init_samples).astype(np.float32)
    y_reg = np.asarray(y_reg_samples, dtype=np.float32)
    y_plan = np.asarray(y_plan_samples, dtype=np.float32)
    future_soc_drop_fraction = np.asarray(soc_drop_samples, dtype=np.float32).reshape(-1, 1)
    energy_per_soc_wh = np.asarray(energy_per_soc_samples, dtype=np.float32).reshape(-1, 1)
    tae_valid_mask = np.asarray(tae_valid_mask_samples, dtype=np.float32).reshape(-1, 1)
    consistency_mask = np.asarray(consistency_mask_samples, dtype=np.float32).reshape(-1, 1)

    validate_built_arrays(
        x_cat=x_cat,
        x_cont=x_cont,
        x_init=x_init,
        y_reg=y_reg,
        y_plan=y_plan,
        future_soc_drop_fraction=future_soc_drop_fraction,
        energy_per_soc_wh=energy_per_soc_wh,
        tae_valid_mask=tae_valid_mask,
        consistency_mask=consistency_mask,
    )

    np.savez_compressed(
        output_path,
        x_cat=x_cat,
        x_cont=x_cont,
        x_init=x_init,
        y_reg=y_reg,
        y_plan=y_plan,
        future_soc_drop_fraction=future_soc_drop_fraction,
        energy_per_soc_wh=energy_per_soc_wh,
        tae_valid_mask=tae_valid_mask,
        consistency_mask=consistency_mask,
        soc_gate_threshold=np.asarray([soc_gate_threshold], dtype=np.float32),
        sample_rate_hz=np.asarray([sample_rate_hz], dtype=np.float32),
        history_seconds=np.asarray([history_seconds], dtype=np.float32),
        future_horizon_seconds=np.asarray([future_horizon_seconds], dtype=np.float32),
        stride_seconds=np.asarray([stride_seconds], dtype=np.float32),
        nominal_energy_wh=np.asarray(
            [np.nan if nominal_energy_wh is None else nominal_energy_wh], dtype=np.float32
        ),
        categorical_columns=np.asarray(CATEGORICAL_COLUMNS),
        continuous_columns=np.asarray(CONTINUOUS_COLUMNS),
        init_columns=np.asarray(INIT_COLUMNS),
        regression_targets=np.asarray(REGRESSION_TARGETS),
        plan_targets=np.asarray(PLAN_TARGETS),
        mappings=json.dumps(mappings, ensure_ascii=False),
        sample_meta=json.dumps(sample_meta, ensure_ascii=False),
    )

    summary = {
        "definition_version": "soc_fraction_time_resampled_tae_plan_v5_masked_tae",
        "output_path": str(output_path),
        "raw_dir": str(raw_dir),
        "num_files": len(files),
        "num_continuous_segments": num_continuous_segments,
        "num_samples": int(x_cat.shape[0]),
        "x_cat_shape": list(x_cat.shape),
        "x_cont_shape": list(x_cont.shape),
        "x_init_shape": list(x_init.shape),
        "y_reg_shape": list(y_reg.shape),
        "y_plan_shape": list(y_plan.shape),
        "sample_rate_hz": sample_rate_hz,
        "history_seconds": history_seconds,
        "future_horizon_seconds": future_horizon_seconds,
        "stride_seconds": stride_seconds,
        "history_samples": history_len,
        "future_samples": future_len,
        "nominal_energy_wh": nominal_energy_wh,
        "soc_scale": "fraction_[0,1]",
        "min_soc_drop_fraction": min_soc_drop_fraction,
        "soc_noise_std_fraction": soc_noise_std_fraction,
        "soc_resolution_fraction": soc_resolution_fraction,
        "soc_consistency_gate_threshold": soc_gate_threshold,
        "tae_valid_sample_count": int(np.sum(tae_valid_mask > 0.5)),
        "tae_invalid_sample_count": int(tae_invalid_count),
        "tae_valid_sample_ratio": float(np.mean(tae_valid_mask > 0.5)),
        "consistency_active_ratio": float(consistency_mask.mean()),
        "skipped_operational_stop": skipped_operational_stop,
        "skipped_invalid_plan": skipped_invalid_plan,
        "label_policy": {
            "future_energy_wh": "Integral of discharge power on the exact [t,t+H] time interval.",
            "future_soc_drop_fraction": "s(t)-s(t+H), with s=SOC/100.",
            "energy_per_soc_wh": "future_energy_wh / future_soc_drop_fraction.",
            "tae_wh": "s(t) * energy_per_soc_wh.",
            "tae_valid_mask": (
                "1 only when future SOC drop is finite and strictly exceeds "
                "min_soc_drop_fraction; invalid windows are retained with zero placeholders "
                "for tae_wh and energy_per_soc_wh."
            ),
            "future_min_voltage_v": "Minimum voltage on [t,t+H], including current time t.",
            "future_max_battery_temp_c": "Maximum powerTemp on [t,t+H], including t.",
            "include_operational_stop_windows": include_operational_stop_windows,
            "future_plan": (
                "Future 300 s raw data are used only to construct y_plan supervision; "
                "they are never stored as model inputs."
            ),
        },
        "target_stats": {
            target: array_stats(
                y_reg[:, idx],
                mask=(tae_valid_mask.reshape(-1) > 0.5) if target == "tae_wh" else None,
            )
            for idx, target in enumerate(REGRESSION_TARGETS)
        },
        "auxiliary_stats": {
            "future_soc_drop_fraction": array_stats(future_soc_drop_fraction),
            "energy_per_soc_wh": array_stats(
                energy_per_soc_wh, mask=tae_valid_mask.reshape(-1) > 0.5
            ),
            "tae_valid_mask": array_stats(tae_valid_mask),
        },
        "plan_target_stats": {
            target: array_stats(y_plan[:, idx]) for idx, target in enumerate(PLAN_TARGETS)
        },
        "future_dock_positive_count": int(np.sum(y_plan[:, -1] > 0.5)),
        "future_dock_positive_ratio": float(np.mean(y_plan[:, -1] > 0.5)),
        "mappings": mappings,
    }
    with output_path.with_suffix(".metadata.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def array_stats(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.shape != values.shape:
            raise ValueError("array_stats mask shape does not match values.")
        values = values[mask]
    if values.size == 0:
        return {"count": 0, "min": float("nan"), "mean": float("nan"), "max": float("nan")}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def validate_built_arrays(**arrays: np.ndarray) -> None:
    """在写入NPZ前做统一形状、样本数和数值检查。"""
    sample_counts = {name: int(value.shape[0]) for name, value in arrays.items()}
    if len(set(sample_counts.values())) != 1:
        raise ValueError(f"Dataset arrays have inconsistent sample counts: {sample_counts}")
    for name, value in arrays.items():
        if name == "x_cat":
            if value.ndim != 3 or np.any(value < 0):
                raise ValueError(f"{name} must be non-negative [N,L,C] category ids; got {value.shape}.")
        elif not np.all(np.isfinite(value)):
            bad_count = int(np.size(value) - np.count_nonzero(np.isfinite(value)))
            raise ValueError(f"{name} contains {bad_count} NaN/Inf values.")
    if arrays["y_reg"].ndim != 2 or arrays["y_reg"].shape[1] != len(REGRESSION_TARGETS):
        raise ValueError(f"y_reg must have shape [N,{len(REGRESSION_TARGETS)}].")
    if arrays["y_plan"].ndim != 2 or arrays["y_plan"].shape[1] != len(PLAN_TARGETS):
        raise ValueError(f"y_plan must have shape [N,{len(PLAN_TARGETS)}].")
    dock = arrays["y_plan"][:, PLAN_TARGETS.index("future_dock_event")]
    if not np.all(np.isin(dock, [0.0, 1.0])):
        raise ValueError("future_dock_event must be binary 0/1.")
    for mask_name in ["tae_valid_mask", "consistency_mask"]:
        if not np.all(np.isin(arrays[mask_name], [0.0, 1.0])):
            raise ValueError(f"{mask_name} must be binary 0/1.")
    if np.any(arrays["consistency_mask"] > arrays["tae_valid_mask"]):
        raise ValueError("consistency_mask cannot activate an invalid TAE sample.")


def read_raw_log(file_path: Path) -> pd.DataFrame:
    header_columns = set(pd.read_csv(file_path, nrows=0).columns)
    missing = sorted(RAW_REQUIRED_COLUMNS.difference(header_columns))
    if missing:
        raise ValueError(f"{file_path.name}缺少必要CSV字段: {missing}")
    if not header_columns.intersection(BODY_TEMP_COLUMNS):
        raise ValueError(f"{file_path.name}缺少用于bodyMaxTemp的机体温度字段。")
    use_columns = {*RAW_REQUIRED_COLUMNS, *BODY_TEMP_COLUMNS}
    df = pd.read_csv(file_path, usecols=lambda column: column in use_columns, low_memory=False)
    for column in use_columns:
        if column not in df.columns:
            df[column] = np.nan
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    numeric_columns = [
        "current",
        "carSpeed",
        "voltage",
        "power",
        "powerTemp",
        "odometry",
        "x",
        "y",
        "z",
        "carDirection",
        *BODY_TEMP_COLUMNS,
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["bodyMaxTemp"] = df[BODY_TEMP_COLUMNS].max(axis=1, skipna=True)
    df["phase"] = df["returnState"].fillna("unknown")
    df = df.dropna(
        subset=[
            "timestamp",
            "voltage",
            "power",
            "powerTemp",
            "current",
            "carSpeed",
            "odometry",
            "x",
            "y",
            "z",
            "carDirection",
            "bodyMaxTemp",
        ]
    )
    return (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def read_and_resample_log(
    file_path: Path,
    sample_rate_hz: float,
    max_raw_gap_seconds: float,
    current_discharge_sign: float,
) -> list[pd.DataFrame]:
    raw = read_raw_log(file_path)
    if raw.empty:
        return []
    dt = raw["timestamp"].diff().dt.total_seconds()
    segment_id = (dt.isna() | (dt <= 0) | (dt > max_raw_gap_seconds)).cumsum()
    segments: list[pd.DataFrame] = []
    for _, segment in raw.groupby(segment_id, sort=True):
        resampled = resample_continuous_segment(segment, sample_rate_hz, current_discharge_sign)
        if not resampled.empty:
            segments.append(resampled)
    return segments


def resample_continuous_segment(
    segment: pd.DataFrame,
    sample_rate_hz: float,
    current_discharge_sign: float,
) -> pd.DataFrame:
    period = pd.to_timedelta(1.0 / sample_rate_hz, unit="s")
    start = segment["timestamp"].iloc[0]
    end = segment["timestamp"].iloc[-1]
    grid = pd.date_range(start=start, end=end, freq=period)
    if len(grid) < 2:
        return pd.DataFrame()
    source = segment.set_index("timestamp")
    union = source.index.union(grid).sort_values()
    numeric_columns = [
        "current",
        "carSpeed",
        "voltage",
        "power",
        "powerTemp",
        "odometry",
        "x",
        "y",
        "z",
        "carDirection",
        "bodyMaxTemp",
    ]
    categorical_columns = [
        "gait",
        "pavement",
        "phase",
        "model",
        "controlMode",
        "chargeState",
        "onDockState",
    ]
    numeric = (
        source[numeric_columns]
        .reindex(union)
        .interpolate(method="time", limit_area="inside")
        .reindex(grid)
    )
    categorical = source[categorical_columns].reindex(union).ffill().bfill().reindex(grid)
    frame = pd.concat([numeric, categorical], axis=1).dropna(subset=numeric_columns)
    frame.index.name = "timestamp"
    frame = frame.reset_index()

    frame["socFraction"] = np.clip(frame["power"].astype(float) / 100.0, 0.0, 1.0)
    frame["dischargeCurrent"] = np.maximum(
        frame["current"].astype(float) * current_discharge_sign, 0.0
    )
    frame["dischargePower"] = frame["voltage"].astype(float) * frame["dischargeCurrent"]
    frame["odometryDelta"] = frame["odometry"].astype(float).diff().fillna(0.0)
    yaw = frame["carDirection"].to_numpy(dtype=np.float64)
    yaw_is_radian = float(np.max(np.abs(yaw))) <= 2.0 * np.pi + 0.5
    yaw_radian = yaw if yaw_is_radian else np.deg2rad(yaw)
    yaw_period = 2.0 * np.pi if yaw_is_radian else 360.0
    yaw_delta = np.diff(yaw)
    yaw_delta = (yaw_delta + yaw_period / 2.0) % yaw_period - yaw_period / 2.0
    if not yaw_is_radian:
        yaw_delta = np.deg2rad(yaw_delta)
    frame["carDirectionSin"] = np.sin(yaw_radian)
    frame["carDirectionCos"] = np.cos(yaw_radian)
    frame["carDirectionDelta"] = np.concatenate([[0.0], yaw_delta])
    frame["isCharging"] = is_active_charging_state(frame["chargeState"]).astype(float)
    frame["isOnDock"] = is_active_dock_state(frame["onDockState"]).astype(float)
    lag30 = max(1, int(round(30.0 * sample_rate_hz)))
    window30 = lag30
    window60 = max(1, int(round(60.0 * sample_rate_hz)))
    frame["voltageDiff30"] = frame["voltage"].astype(float).diff(lag30).fillna(0.0)
    frame["powerTempDiff30"] = frame["powerTemp"].astype(float).diff(lag30).fillna(0.0)
    frame["dischargePowerMean30"] = (
        frame["dischargePower"].rolling(window30, min_periods=1).mean()
    )
    frame["dischargePowerMean60"] = (
        frame["dischargePower"].rolling(window60, min_periods=1).mean()
    )
    frame["dischargePowerStd60"] = (
        frame["dischargePower"].rolling(window60, min_periods=1).std(ddof=0).fillna(0.0)
    )
    frame["voltageRollingMin60"] = (
        frame["voltage"].rolling(window60, min_periods=1).min()
    )
    required = [
        "timestamp",
        *CATEGORICAL_COLUMNS,
        "chargeState",
        "onDockState",
        "x",
        "y",
        "z",
        "odometry",
        "carDirection",
        *CONTINUOUS_COLUMNS,
        *INIT_COLUMNS,
    ]
    return frame[list(dict.fromkeys(required))].copy()


def compute_plan_targets(
    current_row: pd.Series,
    future: pd.DataFrame,
) -> dict[str, float] | None:


    if future.empty:
        return None

    current_xyz = np.asarray(
        [current_row["x"], current_row["y"], current_row["z"]], dtype=np.float64
    )
    final_xyz = future[["x", "y", "z"]].iloc[-1].to_numpy(dtype=np.float64)


    odometry = future["odometry"].to_numpy(dtype=np.float64)
    odometry_increment = np.diff(odometry)
    path_distance = float(np.sum(np.clip(odometry_increment, 0.0, None)))

    speed = np.abs(future["carSpeed"].to_numpy(dtype=np.float64))
    yaw = future["carDirection"].to_numpy(dtype=np.float64)
    yaw_change = wrapped_absolute_angle_change(yaw)

    model_numeric = pd.to_numeric(future["model"], errors="coerce").to_numpy(dtype=np.float64)
    model4_ratio = float(np.mean(np.isclose(model_numeric, 4.0, equal_nan=False)))
    values = {
        "future_delta_x": float(final_xyz[0] - current_xyz[0]),
        "future_delta_y": float(final_xyz[1] - current_xyz[1]),
        "future_delta_z": float(final_xyz[2] - current_xyz[2]),
        "future_path_distance": path_distance,
        "future_mean_abs_speed": float(np.mean(speed)),
        "future_moving_ratio": float(np.mean(speed > 0.03)),
        "future_absolute_yaw_change": yaw_change,
        "future_model4_ratio": model4_ratio,
        "future_dock_event": float(has_operational_stop(future)),
    }
    if not all(np.isfinite(value) for value in values.values()):
        return None
    return values


def wrapped_absolute_angle_change(angles: np.ndarray) -> float:

    values = np.asarray(angles, dtype=np.float64)
    if values.size < 2 or not np.all(np.isfinite(values)):
        return 0.0 if values.size < 2 else float("nan")
    period = 2.0 * np.pi if float(np.max(np.abs(values))) <= 2.0 * np.pi + 0.5 else 360.0
    delta = np.diff(values)
    wrapped = (delta + period / 2.0) % period - period / 2.0
    return float(np.sum(np.abs(wrapped)))


def compute_targets(
    future: pd.DataFrame,
    future_timestamps_ns: np.ndarray,
    current_timestamp_ns: int,
    current_soc_fraction: float,
    current_voltage_v: float,
    current_battery_temp_c: float,
    current_discharge_power_w: float,
    min_soc_drop_fraction: float,
    soc_gate_threshold: float,
    nominal_energy_wh: float | None = None,
) -> dict[str, float] | None:
    if len(future) < 1:
        return None
    future_end_soc_fraction = float(future["socFraction"].iloc[-1])
    observed_soc_drop_fraction = float(current_soc_fraction - future_end_soc_fraction)
    tae_valid = bool(
        np.isfinite(observed_soc_drop_fraction)
        and observed_soc_drop_fraction > min_soc_drop_fraction
    )


    future_soc_drop_fraction = (
        observed_soc_drop_fraction if np.isfinite(observed_soc_drop_fraction) else 0.0
    )

    power_w = np.concatenate(
        [[current_discharge_power_w], future["dischargePower"].to_numpy(dtype=float)]
    )
    timestamps_ns = np.concatenate(
        [[current_timestamp_ns], np.asarray(future_timestamps_ns, dtype=np.int64)]
    )
    future_energy_wh = integrate_energy_wh(power_w, timestamps_ns)
    if tae_valid:
        energy_per_soc_wh = future_energy_wh / future_soc_drop_fraction
        tae_wh = current_soc_fraction * energy_per_soc_wh
    else:
        energy_per_soc_wh = 0.0
        tae_wh = 0.0
    future_min_voltage_v = float(
        min(current_voltage_v, np.nanmin(future["voltage"].to_numpy(dtype=float)))
    )
    future_max_battery_temp_c = float(
        max(current_battery_temp_c, np.nanmax(future["powerTemp"].to_numpy(dtype=float)))
    )

    if nominal_energy_wh is not None and tae_valid:
        soc_energy_wh = nominal_energy_wh * current_soc_fraction
        task_correction_factor = future_energy_wh / (
            nominal_energy_wh * future_soc_drop_fraction
        )
    elif nominal_energy_wh is not None:
        soc_energy_wh = nominal_energy_wh * current_soc_fraction
        task_correction_factor = 0.0
    else:
        soc_energy_wh = float("nan")
        task_correction_factor = float("nan")
    required = [
        current_soc_fraction,
        future_end_soc_fraction,
        future_soc_drop_fraction,
        future_energy_wh,
        energy_per_soc_wh,
        tae_wh,
        future_min_voltage_v,
        future_max_battery_temp_c,
    ]
    if not all(np.isfinite(value) for value in required):
        return None
    return {
        "tae_wh": float(tae_wh),
        "future_energy_wh": float(future_energy_wh),
        "future_min_voltage_v": future_min_voltage_v,
        "future_max_battery_temp_c": future_max_battery_temp_c,
        "future_soc_drop_fraction": future_soc_drop_fraction,
        "energy_per_soc_wh": float(energy_per_soc_wh),
        "current_soc_fraction": float(current_soc_fraction),
        "future_end_soc_fraction": future_end_soc_fraction,
        "tae_valid_mask": float(tae_valid),


        "consistency_mask": float(
            tae_valid and future_soc_drop_fraction > soc_gate_threshold + 1e-9
        ),
        "soc_energy_wh": float(soc_energy_wh),
        "task_correction_factor": float(task_correction_factor),
    }


def integrate_energy_wh(power_w: np.ndarray, timestamps_ns: np.ndarray) -> float:
    if len(power_w) < 2:
        return 0.0
    dt_seconds = np.diff(timestamps_ns).astype(np.float64) / 1e9
    interval_power = 0.5 * (power_w[:-1] + power_w[1:])
    valid = np.isfinite(dt_seconds) & np.isfinite(interval_power) & (dt_seconds > 0)
    if not np.any(valid):
        return 0.0
    return float(np.sum(interval_power[valid] * dt_seconds[valid]) / 3600.0)


def has_operational_stop(future: pd.DataFrame) -> bool:
    active_charging = is_active_charging_state(future["chargeState"]).to_numpy()
    on_dock = is_active_dock_state(future["onDockState"]).to_numpy()
    return bool(np.any(active_charging) or np.any(on_dock))


def is_active_charging_state(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    return text.ne("") & ~text.str.startswith("0") & ~text.str.lower().isin(["nan", "none", "<na>"])


def is_active_dock_state(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    return text.ne("") & ~text.str.startswith("0") & ~text.str.lower().isin(["nan", "none", "<na>"])


def map_category_series(series: pd.Series, mapping: dict[str, int]) -> pd.Series:
    def map_one(value: Any) -> int:
        key = str(value)
        if key not in mapping:
            mapping[key] = len(mapping)
        return mapping[key]

    return series.map(map_one).astype(int)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
