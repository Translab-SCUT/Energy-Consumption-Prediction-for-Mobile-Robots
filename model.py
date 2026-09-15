from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


REGRESSION_OUTPUTS = [
    "tae_wh",
    "future_energy_wh",
    "future_min_voltage_v",
    "future_max_battery_temp_c",
]

PLAN_OUTPUTS = [
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

PLAN_CONTINUOUS_DIM = len(PLAN_OUTPUTS) - 1


def _positive_head(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:


    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        return -10.0
    if value > 20.0:
        return float(value)
    return float(torch.log(torch.expm1(torch.tensor(value))).item())


def _initialise_positive_head(head: nn.Sequential, physical_value: float) -> None:
    final_layer = head[-1]
    if isinstance(final_layer, nn.Linear):
        nn.init.constant_(final_layer.bias, _inverse_softplus(physical_value))


class MultiTaskLSTMAttentionModel(nn.Module):


    def __init__(
        self,
        category_cardinalities: list[int],
        continuous_dim: int,
        init_dim: int,
        category_embedding_dim: int = 4,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        shared_hidden_dim: int = 128,
        dropout: float = 0.2,
        num_regression_targets: int = 4,
        attention_hidden_dim: int | None = None,
        initial_energy_per_soc_wh: float = 1.0,
        initial_future_energy_wh: float = 1.0,
        initial_voltage_drop_v: float = 1.0,
        initial_temperature_rise_c: float = 1.0,
        plan_target_mean: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        plan_target_std: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        initial_dock_probability: float = 0.01,
    ) -> None:
        super().__init__()
        if not category_cardinalities:
            raise ValueError("category_cardinalities must not be empty.")
        if num_regression_targets != len(REGRESSION_OUTPUTS):
            raise ValueError(
                f"The physics-structured model has exactly {len(REGRESSION_OUTPUTS)} outputs; "
                f"got num_regression_targets={num_regression_targets}."
            )

        plan_mean = torch.zeros(PLAN_CONTINUOUS_DIM, dtype=torch.float32)
        plan_std = torch.ones(PLAN_CONTINUOUS_DIM, dtype=torch.float32)
        if plan_target_mean is not None:
            plan_mean = torch.as_tensor(plan_target_mean, dtype=torch.float32).reshape(-1)
        if plan_target_std is not None:
            plan_std = torch.as_tensor(plan_target_std, dtype=torch.float32).reshape(-1)
        if plan_mean.numel() != PLAN_CONTINUOUS_DIM or plan_std.numel() != PLAN_CONTINUOUS_DIM:
            raise ValueError(
                f"plan_target_mean/std must each contain {PLAN_CONTINUOUS_DIM} continuous values."
            )
        if not torch.isfinite(plan_mean).all() or not torch.isfinite(plan_std).all():
            raise ValueError("plan_target_mean/std contain NaN or Inf.")
        if torch.any(plan_std <= 0):
            raise ValueError("plan_target_std must be strictly positive.")
        self.register_buffer("plan_target_mean", plan_mean.view(1, -1))
        self.register_buffer("plan_target_std", plan_std.view(1, -1))

        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, category_embedding_dim) for cardinality in category_cardinalities]
        )
        lstm_input_dim = len(category_cardinalities) * category_embedding_dim + continuous_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )
        attention_hidden_dim = attention_hidden_dim or lstm_hidden_dim
        self.attention_pooling = nn.Sequential(
            nn.Linear(lstm_hidden_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
        )
        context_dim = lstm_hidden_dim + init_dim
        plan_hidden_dim = max(shared_hidden_dim, 64)
        self.plan_head = nn.Sequential(
            nn.Linear(context_dim, plan_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(plan_hidden_dim, len(PLAN_OUTPUTS)),
        )
        if not 0.0 < initial_dock_probability < 1.0:
            raise ValueError("initial_dock_probability must be between 0 and 1.")
        plan_final = self.plan_head[-1]
        if isinstance(plan_final, nn.Linear):
            with torch.no_grad():
                plan_final.bias[:PLAN_CONTINUOUS_DIM].zero_()
                plan_final.bias[PLAN_CONTINUOUS_DIM] = torch.logit(
                    torch.tensor(float(initial_dock_probability)).clamp(1e-5, 1.0 - 1e-5)
                )
        self.shared = nn.Sequential(
            nn.Linear(context_dim, shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_hidden_dim, shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        head_hidden_dim = max(shared_hidden_dim // 2, 32)
        self.energy_per_soc_head = _positive_head(shared_hidden_dim, head_hidden_dim, dropout)
        self.future_energy_head = _positive_head(shared_hidden_dim, head_hidden_dim, dropout)
        self.voltage_drop_head = _positive_head(shared_hidden_dim, head_hidden_dim, dropout)
        self.temperature_rise_head = _positive_head(shared_hidden_dim, head_hidden_dim, dropout)
        _initialise_positive_head(self.energy_per_soc_head, initial_energy_per_soc_wh)
        _initialise_positive_head(self.future_energy_head, initial_future_energy_wh)
        _initialise_positive_head(self.voltage_drop_head, initial_voltage_drop_v)
        _initialise_positive_head(self.temperature_rise_head, initial_temperature_rise_c)

    def encode(
        self,
        x_cat: torch.Tensor,
        x_cont: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = [
            embedding(x_cat[:, :, index].long())
            for index, embedding in enumerate(self.embeddings)
        ]
        seq_input = torch.cat([*embedded, x_cont.float()], dim=-1)
        lstm_output, _ = self.lstm(seq_input)
        attention_logits = self.attention_pooling(lstm_output).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1)
        seq_feature = torch.sum(lstm_output * attention_weights.unsqueeze(-1), dim=1)
        return seq_feature, attention_weights

    def forward(
        self,
        x_cat: torch.Tensor,
        x_cont: torch.Tensor,
        x_init: torch.Tensor,
        anchors: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x_cat.ndim != 3 or x_cat.shape[-1] != len(self.embeddings):
            raise ValueError(
                f"x_cat must have shape [batch,time,{len(self.embeddings)}]; got {tuple(x_cat.shape)}."
            )
        if x_cont.ndim != 3 or x_cont.shape[:2] != x_cat.shape[:2]:
            raise ValueError("x_cont must be [batch,time,features] and align with x_cat.")
        if x_init.ndim != 2 or x_init.shape[0] != x_cat.shape[0]:
            raise ValueError("x_init must be [batch,features] and align with x_cat.")
        if anchors.ndim != 2 or anchors.shape[-1] != 3:
            raise ValueError(
                "anchors must have shape [batch, 3] for [SOC fraction, voltage, battery temperature]."
            )
        if anchors.shape[0] != x_cat.shape[0]:
            raise ValueError("anchors batch size must match x_cat.")
        if not torch.isfinite(x_cont).all() or not torch.isfinite(x_init).all() or not torch.isfinite(anchors).all():
            raise ValueError("Model inputs contain NaN or Inf.")

        history_feature, attention_weights = self.encode(x_cat, x_cont)
        present_context = torch.cat([history_feature, x_init.float()], dim=-1)



        plan_raw = self.plan_head(present_context)
        plan_continuous_norm = plan_raw[:, :PLAN_CONTINUOUS_DIM]
        dock_logit = plan_raw[:, PLAN_CONTINUOUS_DIM : PLAN_CONTINUOUS_DIM + 1]
        plan_continuous = (
            plan_continuous_norm * self.plan_target_std + self.plan_target_mean
        )
        plan_prediction = torch.cat([plan_continuous, dock_logit], dim=-1)


        shared_feature = self.shared(present_context)
        energy_per_soc = F.softplus(self.energy_per_soc_head(shared_feature))
        future_energy = F.softplus(self.future_energy_head(shared_feature))
        voltage_drop = F.softplus(self.voltage_drop_head(shared_feature))
        temperature_rise = F.softplus(self.temperature_rise_head(shared_feature))

        current_soc = anchors[:, 0:1].float().clamp(min=0.0, max=1.0)
        current_voltage = anchors[:, 1:2].float()
        current_battery_temp = anchors[:, 2:3].float()

        tae = current_soc * energy_per_soc
        future_min_voltage = current_voltage - voltage_drop
        future_max_temp = current_battery_temp + temperature_rise
        regression = torch.cat(
            [tae, future_energy, future_min_voltage, future_max_temp],
            dim=-1,
        )
        return {
            "regression": regression,
            "plan_prediction": plan_prediction,
            "tae_wh": tae,
            "energy_per_soc_wh": energy_per_soc,
            "future_energy_wh": future_energy,
            "voltage_drop_v": voltage_drop,
            "future_min_voltage_v": future_min_voltage,
            "temperature_rise_c": temperature_rise,
            "future_max_battery_temp_c": future_max_temp,
            "attention_weights": attention_weights,
        }
