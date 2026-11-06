#!/usr/bin/env python3
"""
Glass-Box MLP Plotly Animation
--------------------------------
Creates a standalone HTML dashboard that explains how a tiny 1-2-1 MLP
learns from data through forward pass, ReLU, loss, backward pass, and updates.

Run:
    python glass_box_mlp_plotly.py --output glass_box_mlp.html --epochs 5 --lr 0.08
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import webbrowser
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import plotly.graph_objects as go


# ============================================================
# Data containers
# ============================================================

@dataclass
class Neuron:
    name: str
    x: float
    y: float
    value: Optional[float] = None
    z: Optional[float] = None
    active: bool = False


@dataclass
class Layer:
    name: str
    neurons: List[Neuron]


@dataclass
class ForwardCache:
    x: float
    y: float
    z: np.ndarray
    h: np.ndarray
    t: np.ndarray
    y_hat: float
    loss: float


@dataclass
class Gradients:
    dw_h: np.ndarray
    db_h: np.ndarray
    dw_o: np.ndarray
    db_o: float
    dz: np.ndarray
    dh: np.ndarray
    d_y_hat: float


@dataclass
class StepRecord:
    epoch: int
    sample_index: int
    x: float
    y: float
    params_before: Dict[str, np.ndarray | float]
    cache: ForwardCache
    gradients: Gradients
    params_after_output: Dict[str, np.ndarray | float]
    params_after: Dict[str, np.ndarray | float]


# ============================================================
# Model
# ============================================================

class GlassBoxMLP:
    """Small 1-2-1 MLP with ReLU hidden layer and squared loss."""

    def __init__(self, lr: float = 0.08) -> None:
        self.lr = lr
        self.w_h = np.array([0.85, -0.70], dtype=float)
        self.b_h = np.array([-0.15, 0.20], dtype=float)
        self.w_o = np.array([0.70, -0.45], dtype=float)
        self.b_o = 0.05

    def snapshot(self) -> Dict[str, np.ndarray | float]:
        return {
            "w_h": self.w_h.copy(),
            "b_h": self.b_h.copy(),
            "w_o": self.w_o.copy(),
            "b_o": float(self.b_o),
        }

    @staticmethod
    def with_params(params: Dict[str, np.ndarray | float], x_grid: np.ndarray) -> np.ndarray:
        w_h = np.asarray(params["w_h"], dtype=float)
        b_h = np.asarray(params["b_h"], dtype=float)
        w_o = np.asarray(params["w_o"], dtype=float)
        b_o = float(params["b_o"])
        z = x_grid[:, None] * w_h[None, :] + b_h[None, :]
        h = np.maximum(0.0, z)
        return h @ w_o + b_o

    def forward(self, x: float, y: float) -> ForwardCache:
        z = self.w_h * x + self.b_h
        h = np.maximum(0.0, z)
        t = self.w_o * h
        y_hat = float(np.sum(t) + self.b_o)
        loss = float((y_hat - y) ** 2)
        return ForwardCache(x=x, y=y, z=z, h=h, t=t, y_hat=y_hat, loss=loss)

    def backward(self, cache: ForwardCache) -> Gradients:
        d_y_hat = 2.0 * (cache.y_hat - cache.y)
        dw_o = d_y_hat * cache.h
        db_o = float(d_y_hat)
        dh = d_y_hat * self.w_o
        relu_mask = (cache.z > 0).astype(float)
        dz = dh * relu_mask
        dw_h = dz * cache.x
        db_h = dz
        return Gradients(
            dw_h=dw_h,
            db_h=db_h,
            dw_o=dw_o,
            db_o=db_o,
            dz=dz,
            dh=dh,
            d_y_hat=float(d_y_hat),
        )

    @staticmethod
    def output_updated_params(
        params_before: Dict[str, np.ndarray | float], gradients: Gradients, lr: float
    ) -> Dict[str, np.ndarray | float]:
        out = {
            "w_h": np.asarray(params_before["w_h"], dtype=float).copy(),
            "b_h": np.asarray(params_before["b_h"], dtype=float).copy(),
            "w_o": np.asarray(params_before["w_o"], dtype=float).copy(),
            "b_o": float(params_before["b_o"]),
        }
        out["w_o"] = np.asarray(out["w_o"], dtype=float) - lr * gradients.dw_o
        out["b_o"] = float(out["b_o"]) - lr * gradients.db_o
        return out

    @staticmethod
    def all_updated_params(
        params_before: Dict[str, np.ndarray | float], gradients: Gradients, lr: float
    ) -> Dict[str, np.ndarray | float]:
        out = GlassBoxMLP.output_updated_params(params_before, gradients, lr)
        out["w_h"] = np.asarray(out["w_h"], dtype=float) - lr * gradients.dw_h
        out["b_h"] = np.asarray(out["b_h"], dtype=float) - lr * gradients.db_h
        return out

    def apply_gradients(self, gradients: Gradients) -> None:
        self.w_o = self.w_o - self.lr * gradients.dw_o
        self.b_o = self.b_o - self.lr * gradients.db_o
        self.w_h = self.w_h - self.lr * gradients.dw_h
        self.b_h = self.b_h - self.lr * gradients.db_h

    def train_records(self, x_data: np.ndarray, y_data: np.ndarray, epochs: int) -> List[StepRecord]:
        records: List[StepRecord] = []
        for epoch in range(1, epochs + 1):
            for sample_index, (x, y) in enumerate(zip(x_data, y_data), start=1):
                params_before = self.snapshot()
                cache = self.forward(float(x), float(y))
                gradients = self.backward(cache)
                params_after_output = self.output_updated_params(params_before, gradients, self.lr)
                params_after = self.all_updated_params(params_before, gradients, self.lr)
                records.append(
                    StepRecord(
                        epoch=epoch,
                        sample_index=sample_index,
                        x=float(x),
                        y=float(y),
                        params_before=params_before,
                        cache=cache,
                        gradients=gradients,
                        params_after_output=params_after_output,
                        params_after=params_after,
                    )
                )
                self.apply_gradients(gradients)
        return records


# ============================================================
# Plot builder
# ============================================================

class GlassBoxPlotBuilder:
    PHASES = [
        "load",
        "forward_hidden",
        "relu",
        "forward_output",
        "loss",
        "backward_output",
        "update_output",
        "backward_hidden",
        "update_hidden",
    ]

    PHASE_TITLES = {
        "load": "Nạp mẫu dữ liệu",
        "forward_hidden": "Tính z ở lớp ẩn",
        "relu": "Áp dụng ReLU",
        "forward_output": "Tính đầu ra ŷ",
        "loss": "Tính mất mát",
        "backward_output": "Lan truyền ngược từ đầu ra",
        "update_output": "Cập nhật tham số đầu ra",
        "backward_hidden": "Lan truyền ngược về lớp ẩn",
        "update_hidden": "Cập nhật tham số lớp ẩn",
    }

    def __init__(
        self,
        records: List[StepRecord],
        x_data: np.ndarray,
        y_data: np.ndarray,
        epochs: int,
        lr: float,
        output: str,
        auto_open: bool = True,
    ) -> None:
        self.records = records
        self.x_data = x_data
        self.y_data = y_data
        self.epochs = epochs
        self.lr = lr
        self.output = output
        self.auto_open = auto_open
        self.x_grid = np.linspace(min(x_data) - 0.4, max(x_data) + 0.4, 220)
        self.y_curve_all = [GlassBoxMLP.with_params(r.params_before, self.x_grid) for r in records]
        self.y_min = float(min(np.min(y_data), min(np.min(v) for v in self.y_curve_all)) - 0.25)
        self.y_max = float(max(np.max(y_data), max(np.max(v) for v in self.y_curve_all)) + 0.25)

        self.input_neuron = Neuron("x", 10, 62)
        self.hidden_layer = Layer("hidden", [Neuron("h<sub>1</sub>", 32, 72), Neuron("h<sub>2</sub>", 32, 52)])
        self.output_neuron = Neuron("ŷ", 56, 62)
        self.relu_chart_boxes = self.activation_chart_boxes()

    # ---------- Formatting helpers ----------

    @staticmethod
    def fmt(v: float, digits: int = 3) -> str:
        return f"{v:.{digits}f}"

    @staticmethod
    def signed(v: float, digits: int = 3) -> str:
        return f"{v:+.{digits}f}"

    @staticmethod
    def html_signed(v: float, digits: int = 3) -> str:
        if v >= 0:
            return f"+{v:.{digits}f}"
        return f"{v:.{digits}f}"

    @staticmethod
    def clone_params(params: Dict[str, np.ndarray | float]) -> Dict[str, np.ndarray | float]:
        return {
            "w_h": np.asarray(params["w_h"], dtype=float).copy(),
            "b_h": np.asarray(params["b_h"], dtype=float).copy(),
            "w_o": np.asarray(params["w_o"], dtype=float).copy(),
            "b_o": float(params["b_o"]),
        }

    @staticmethod
    def neuron_fill(value: Optional[float], vmin: float = -0.5, vmax: float = 2.0) -> str:
        if value is None:
            return "#EAF4FF"
        t = (float(value) - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, t))
        # Dark blue -> bright cyan, same blue/cyan family.
        r = int(18 + 70 * t)
        g = int(72 + 170 * t)
        b = int(145 + 95 * t)
        return f"rgb({r},{g},{b})"

    @staticmethod
    def edge_width(w: float) -> float:
        return 2.0 + min(5.0, abs(float(w)) * 4.2)

    @staticmethod
    def edge_opacity(w: float) -> float:
        return 0.25 + min(0.65, abs(float(w)) * 0.55)

    @staticmethod
    def param_size(v: float) -> float:
        return 17 + min(18, abs(float(v)) * 16)

    @staticmethod
    def rgba(hex_color: str, opacity: float) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{opacity:.3f})"

    def activation_chart_boxes(self) -> List[Tuple[float, float, float, float]]:
        """Return compact ReLU chart boxes placed closer to each hidden neuron."""
        old_w, old_h = 10.0, 7.0
        new_w, new_h = old_w * 0.7, old_h * 0.7
        old_boxes = [(38.5, 80.0, old_w, old_h), (38.5, 39.0, old_w, old_h)]
        neurons = [(32.0, 72.0), (32.0, 52.0)]
        boxes: List[Tuple[float, float, float, float]] = []
        for (old_x0, old_y0, w, h), (nx, ny) in zip(old_boxes, neurons):
            old_cx = old_x0 + w / 2
            old_cy = old_y0 + h / 2
            new_cx = nx + (old_cx - nx) / 2
            new_cy = ny + (old_cy - ny) / 2
            boxes.append((new_cx - new_w / 2, new_cy - new_h / 2, new_w, new_h))
        return boxes

    # ---------- Dynamic state ----------

    def params_for_phase(self, record: StepRecord, phase: str) -> Dict[str, np.ndarray | float]:
        if phase == "update_output":
            return self.clone_params(record.params_after_output)
        if phase == "update_hidden":
            return self.clone_params(record.params_after)
        return self.clone_params(record.params_before)

    def visible_values(self, record: StepRecord, phase: str) -> Dict[str, Optional[float]]:
        cache = record.cache
        values: Dict[str, Optional[float]] = {
            "x": record.x,
            "h1": None,
            "h2": None,
            "yhat": None,
        }
        if phase in {"relu", "forward_output", "loss", "backward_output", "update_output", "backward_hidden", "update_hidden"}:
            values["h1"] = float(cache.h[0])
            values["h2"] = float(cache.h[1])
        if phase in {"forward_output", "loss", "backward_output", "update_output", "backward_hidden", "update_hidden"}:
            values["yhat"] = float(cache.y_hat)
        return values

    # ---------- Base traces ----------

    def make_traces(self, record: StepRecord, phase: str) -> List[go.BaseTraceType]:
        params = self.params_for_phase(record, phase)
        visible = self.visible_values(record, phase)
        traces: List[go.BaseTraceType] = []

        traces.extend(self.edge_traces(params, phase))
        traces.append(self.neuron_trace(record, phase, visible))
        traces.append(self.parameter_bubble_trace(params))
        traces.append(self.forward_pulse_trace(phase))
        traces.append(self.backward_pulse_trace(phase))
        traces.append(self.update_star_trace(record, phase))
        traces.extend(self.relu_traces(record, phase))
        traces.extend(self.prediction_traces(record, phase, params))
        return traces

    def edge_traces(self, params: Dict[str, np.ndarray | float], phase: str) -> List[go.Scatter]:
        w_h = np.asarray(params["w_h"], dtype=float)
        w_o = np.asarray(params["w_o"], dtype=float)
        edge_defs = [
            ("x → h₁", (10, 62), (32, 72), w_h[0]),
            ("x → h₂", (10, 62), (32, 52), w_h[1]),
            ("h₁ → ŷ", (32, 72), (56, 62), w_o[0]),
            ("h₂ → ŷ", (32, 52), (56, 62), w_o[1]),
        ]
        active_names = set()
        if phase == "forward_hidden":
            active_names.update(["x → h₁", "x → h₂"])
        elif phase == "forward_output":
            active_names.update(["h₁ → ŷ", "h₂ → ŷ"])
        elif phase == "backward_output":
            active_names.update(["h₁ → ŷ", "h₂ → ŷ"])
        elif phase == "backward_hidden":
            active_names.update(["x → h₁", "x → h₂"])

        traces: List[go.Scatter] = []
        for name, p0, p1, w in edge_defs:
            is_active = name in active_names
            base = "#1B5FA7"
            if phase.startswith("backward") and is_active:
                base = "#E14667"
            elif is_active:
                base = "#F4A11A"
            traces.append(
                go.Scatter(
                    x=[p0[0], p1[0]],
                    y=[p0[1], p1[1]],
                    mode="lines",
                    line=dict(color=self.rgba(base, self.edge_opacity(float(w)) + (0.18 if is_active else 0)), width=self.edge_width(float(w))),
                    hoverinfo="skip",
                    showlegend=False,
                    xaxis="x",
                    yaxis="y",
                )
            )
        return traces

    def neuron_trace(self, record: StepRecord, phase: str, visible: Dict[str, Optional[float]]) -> go.Scatter:
        cache = record.cache
        labels = [
            f"x<br><b>{self.fmt(record.x, 2)}</b>",
            "h<sub>1</sub>" if visible["h1"] is None else f"h<sub>1</sub><br><b>{self.fmt(float(visible['h1']))}</b>",
            "h<sub>2</sub>" if visible["h2"] is None else f"h<sub>2</sub><br><b>{self.fmt(float(visible['h2']))}</b>",
            "ŷ" if visible["yhat"] is None else f"ŷ<br><b>{self.fmt(float(visible['yhat']))}</b>",
        ]
        values = [record.x, visible["h1"], visible["h2"], visible["yhat"]]
        colors = [self.neuron_fill(v) for v in values]

        line_colors = ["#276AA3", "#276AA3", "#276AA3", "#276AA3"]
        line_widths = [2, 2, 2, 2]
        if phase in {"relu", "forward_output", "loss", "backward_output", "update_output", "backward_hidden", "update_hidden"}:
            for idx, z in enumerate(cache.z, start=1):
                if z > 0:
                    line_colors[idx] = "#FFD43B"
                    line_widths[idx] = 7
                else:
                    line_colors[idx] = "#9EA8B3"
                    line_widths[idx] = 2

        return go.Scatter(
            x=[10, 32, 32, 56],
            y=[62, 72, 52, 62],
            mode="markers+text",
            marker=dict(
                size=[58, 62, 62, 62],
                color=colors,
                line=dict(color=line_colors, width=line_widths),
                opacity=1,
            ),
            text=labels,
            textposition="middle center",
            textfont=dict(size=15, color="#0B1E33", family="Arial"),
            hoverinfo="skip",
            showlegend=False,
            xaxis="x",
            yaxis="y",
        )

    def parameter_bubble_trace(self, params: Dict[str, np.ndarray | float]) -> go.Scatter:
        w_h = np.asarray(params["w_h"], dtype=float)
        b_h = np.asarray(params["b_h"], dtype=float)
        w_o = np.asarray(params["w_o"], dtype=float)
        b_o = float(params["b_o"])
        vals = [w_h[0], w_h[1], b_h[0], b_h[1], w_o[0], w_o[1], b_o]
        xs = [21, 21, 32, 32, 45, 45, 56]
        ys = [68.5, 55.5, 82, 42, 69, 55, 74]
        return go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(
                size=[self.param_size(v) for v in vals],
                color="#FFFFFF",
                line=dict(color="#2B78B8", width=1.5),
                opacity=0.95,
            ),
            hoverinfo="skip",
            showlegend=False,
            xaxis="x",
            yaxis="y",
        )

    def interpolate(self, p0: Tuple[float, float], p1: Tuple[float, float], t: float) -> Tuple[float, float]:
        return (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)

    def forward_pulse_trace(self, phase: str) -> go.Scatter:
        xs: List[float] = []
        ys: List[float] = []
        if phase == "forward_hidden":
            for p1 in [(32, 72), (32, 52)]:
                x, y = self.interpolate((10, 62), p1, 0.68)
                xs.append(x); ys.append(y)
        elif phase == "forward_output":
            for p0 in [(32, 72), (32, 52)]:
                x, y = self.interpolate(p0, (56, 62), 0.68)
                xs.append(x); ys.append(y)
        return go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=18, color="#FFB000", line=dict(color="#FFF6D1", width=3), symbol="circle"),
            hoverinfo="skip",
            showlegend=False,
            xaxis="x",
            yaxis="y",
        )

    def backward_pulse_trace(self, phase: str) -> go.Scatter:
        xs: List[float] = []
        ys: List[float] = []
        if phase == "backward_output":
            for p1 in [(32, 72), (32, 52)]:
                x, y = self.interpolate((56, 62), p1, 0.62)
                xs.append(x); ys.append(y)
        elif phase == "backward_hidden":
            for p0 in [(32, 72), (32, 52)]:
                x, y = self.interpolate(p0, (10, 62), 0.62)
                xs.append(x); ys.append(y)
        return go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=18, color="#F05478", line=dict(color="#FFE0E8", width=3), symbol="circle"),
            hoverinfo="skip",
            showlegend=False,
            xaxis="x",
            yaxis="y",
        )

    def update_star_trace(self, record: StepRecord, phase: str) -> go.Scatter:
        xs: List[float] = []
        ys: List[float] = []
        if phase == "update_output":
            xs = [45, 45, 56]
            ys = [69, 55, 74]
        elif phase == "update_hidden":
            xs = [21, 21, 32, 32]
            ys = [68.5, 55.5, 82, 42]
        return go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=22, color="#FFD23C", line=dict(color="#9A6B00", width=1.5), symbol="star"),
            hoverinfo="skip",
            showlegend=False,
            xaxis="x",
            yaxis="y",
        )

    def relu_traces(self, record: StepRecord, phase: str) -> List[go.Scatter]:
        # Two mini charts: curve and moving dot. Dot appears after z is calculated.
        traces: List[go.Scatter] = []
        chart_boxes = self.relu_chart_boxes
        # ReLU polyline in normalized mini-chart coords: z=-1..1, relu=0..1.
        z_points = np.array([-1.0, 0.0, 1.0])
        relu_points = np.maximum(0, z_points)
        for idx, (x0, y0, w, h) in enumerate(chart_boxes):
            xs = x0 + (z_points + 1) / 2 * w
            ys = y0 + relu_points * h
            traces.append(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color="#268BD2", width=2.5),
                    hoverinfo="skip",
                    showlegend=False,
                    xaxis="x",
                    yaxis="y",
                )
            )
            dot_xs: List[float] = []
            dot_ys: List[float] = []
            if phase in {"forward_hidden", "relu", "forward_output", "loss", "backward_output", "update_output", "backward_hidden", "update_hidden"}:
                z = float(record.cache.z[idx])
                z_clip = max(-1.0, min(1.0, z))
                dot_xs = [x0 + (z_clip + 1) / 2 * w]
                dot_ys = [y0 + max(0.0, z_clip) * h]
            traces.append(
                go.Scatter(
                    x=dot_xs,
                    y=dot_ys,
                    mode="markers",
                    marker=dict(size=10, color="#FFD43B" if record.cache.z[idx] > 0 else "#9EA8B3", line=dict(color="#263238", width=1)),
                    hoverinfo="skip",
                    showlegend=False,
                    xaxis="x",
                    yaxis="y",
                )
            )
        return traces

    def prediction_traces(
        self, record: StepRecord, phase: str, params: Dict[str, np.ndarray | float]
    ) -> List[go.BaseTraceType]:
        y_curve = GlassBoxMLP.with_params(params, self.x_grid)
        y_current_model = float(GlassBoxMLP.with_params(params, np.array([record.x]))[0])
        show_prediction = phase in {"forward_output", "loss", "backward_output", "update_output", "backward_hidden", "update_hidden"}
        show_error = phase in {"loss", "backward_output", "update_output", "backward_hidden", "update_hidden"}

        traces: List[go.BaseTraceType] = [
            go.Scatter(
                x=self.x_grid,
                y=y_curve,
                mode="lines",
                line=dict(color="#F28E2B", width=3),
                name="Hàm mạng hiện tại",
                xaxis="x2",
                yaxis="y2",
                hovertemplate="x=%{x:.2f}<br>ŷ=%{y:.3f}<extra></extra>",
            ),
            go.Scatter(
                x=self.x_data,
                y=self.y_data,
                mode="markers",
                marker=dict(size=8, color="#245D8A", line=dict(color="#FFFFFF", width=1)),
                name="Dữ liệu",
                xaxis="x2",
                yaxis="y2",
                hovertemplate="x=%{x:.2f}<br>y=%{y:.3f}<extra></extra>",
            ),
            go.Scatter(
                x=[record.x],
                y=[record.y],
                mode="markers",
                marker=dict(size=14, color="#00A1C9", line=dict(color="#0B1E33", width=2), symbol="circle"),
                name="Mẫu đang học",
                xaxis="x2",
                yaxis="y2",
                hovertemplate="x=%{x:.2f}<br>y=%{y:.3f}<extra></extra>",
            ),
            go.Scatter(
                x=[record.x] if show_prediction else [],
                y=[y_current_model] if show_prediction else [],
                mode="markers",
                marker=dict(size=13, color="#FFB000", line=dict(color="#7A4A00", width=2), symbol="diamond"),
                name="Dự đoán ŷ",
                xaxis="x2",
                yaxis="y2",
                hovertemplate="x=%{x:.2f}<br>ŷ=%{y:.3f}<extra></extra>",
            ),
            go.Scatter(
                x=[record.x, record.x] if show_error else [],
                y=[record.y, y_current_model] if show_error else [],
                mode="lines",
                line=dict(color="#D62728", width=3, dash="dot"),
                name="Sai số",
                xaxis="x2",
                yaxis="y2",
                hoverinfo="skip",
            ),
        ]
        return traces

    # ---------- Shapes and annotations ----------

    def static_panel_shapes(self) -> List[dict]:
        card = dict(type="rect", xref="x", yref="y", line=dict(color="#D5E0EA", width=1.2), fillcolor="#F8FBFF", layer="below")
        return [
            {**card, "x0": 2, "y0": 91, "x1": 98, "y1": 99, "fillcolor": "#F3F8FF"},
            {**card, "x0": 2, "y0": 35, "x1": 62, "y1": 89, "fillcolor": "#FBFDFF"},
            {**card, "x0": 65, "y0": 45, "x1": 98, "y1": 89, "fillcolor": "#FBFDFF"},
            {**card, "x0": 65, "y0": 12, "x1": 98, "y1": 42, "fillcolor": "#FBFDFF"},
            {**card, "x0": 2, "y0": 2, "x1": 62, "y1": 32, "fillcolor": "#FBFDFF"},
        ]

    def relu_chart_shapes(self) -> List[dict]:
        shapes: List[dict] = []
        for x0, y0, w, h in self.relu_chart_boxes:
            shapes.extend([
                dict(type="rect", xref="x", yref="y", x0=x0 - 0.7, y0=y0 - 0.7, x1=x0 + w + 0.8, y1=y0 + h + 0.8,
                     line=dict(color="#D4DEE9", width=1), fillcolor="#FFFFFF", layer="below"),
                dict(type="line", xref="x", yref="y", x0=x0, y0=y0, x1=x0 + w, y1=y0,
                     line=dict(color="#9AA8B6", width=1)),
                dict(type="line", xref="x", yref="y", x0=x0 + w / 2, y0=y0, x1=x0 + w / 2, y1=y0 + h,
                     line=dict(color="#9AA8B6", width=1)),
            ])
        return shapes

    def gradient_shapes(self, record: StepRecord, phase: str) -> Tuple[List[dict], List[dict]]:
        # Returns shapes and annotations for gradient panel.
        shapes: List[dict] = []
        ann: List[dict] = []
        center = 83.8
        x_left = 73.8
        x_right = 96.2
        bar_h = 1.15
        row_y = [37.2, 34.0, 30.8, 27.6, 24.4, 21.2, 18.0]
        labels = [
            "∂L/∂w<sub>o,1</sub>",
            "∂L/∂w<sub>o,2</sub>",
            "∂L/∂b<sub>o</sub>",
            "∂L/∂w<sub>h,1</sub>",
            "∂L/∂b<sub>h,1</sub>",
            "∂L/∂w<sub>h,2</sub>",
            "∂L/∂b<sub>h,2</sub>",
        ]
        grads = np.array([
            record.gradients.dw_o[0],
            record.gradients.dw_o[1],
            record.gradients.db_o,
            record.gradients.dw_h[0],
            record.gradients.db_h[0],
            record.gradients.dw_h[1],
            record.gradients.db_h[1],
        ], dtype=float)

        # Only show relevant groups for backward/update phases.
        mask = np.zeros_like(grads)
        if phase in {"backward_output", "update_output"}:
            mask[:3] = 1
        elif phase in {"backward_hidden", "update_hidden"}:
            mask[:] = 1
        # Keep tiny grey ghost bars before gradients are introduced.
        shown = grads * mask
        max_abs = max(0.04, float(np.max(np.abs(grads))))
        scale = 9.0 / max_abs

        # center line and faint row tracks
        shapes.append(dict(type="line", xref="x", yref="y", x0=center, x1=center, y0=16.7, y1=38.4,
                           line=dict(color="#6B7785", width=1.4)))
        for y in row_y:
            shapes.append(dict(type="line", xref="x", yref="y", x0=x_left, x1=x_right, y0=y, y1=y,
                               line=dict(color="#E5EBF1", width=1)))
        for y, lab, g, raw_g in zip(row_y, labels, shown, grads):
            if abs(g) < 1e-9:
                width = 0.28 if mask[list(row_y).index(y)] == 1 and abs(raw_g) < 1e-9 else 0.0
                color = "#AEB8C2"
            else:
                width = min(9.0, abs(float(g)) * scale)
                color = "#2F80ED" if g > 0 else "#E24A5A"
            x0 = center
            x1 = center + math.copysign(width, g if abs(g) > 1e-12 else 1)
            if width > 0:
                shapes.append(dict(
                    type="rect", xref="x", yref="y",
                    x0=min(x0, x1), x1=max(x0, x1), y0=y - bar_h / 2, y1=y + bar_h / 2,
                    line=dict(color=color, width=0), fillcolor=color,
                ))
            ann.append(dict(
                x=66.8, y=y, xref="x", yref="y", text=lab, showarrow=False,
                font=dict(size=10, color="#26384D"), align="left", xanchor="left"
            ))
            val_text = "0" if abs(g) < 1e-9 else self.signed(float(g), 3)
            if mask[list(row_y).index(y)] == 0:
                val_text = "—"
            ann.append(dict(
                x=96.6, y=y, xref="x", yref="y", text=val_text, showarrow=False,
                font=dict(size=10, color="#26384D"), align="right", xanchor="right"
            ))
        return shapes, ann

    def formula_annotations(self, record: StepRecord, phase: str) -> List[dict]:
        p = self.params_for_phase(record, phase)
        w_h = np.asarray(p["w_h"], dtype=float)
        b_h = np.asarray(p["b_h"], dtype=float)
        w_o = np.asarray(p["w_o"], dtype=float)
        b_o = float(p["b_o"])
        c = record.cache

        z1_symbolic = "z<sub>1</sub> = w<sub>h,1</sub>x + b<sub>h,1</sub>"
        z2_symbolic = "z<sub>2</sub> = w<sub>h,2</sub>x + b<sub>h,2</sub>"
        if phase in {"forward_hidden", "relu", "forward_output", "loss", "backward_output", "update_output", "backward_hidden", "update_hidden"}:
            z1 = f"z<sub>1</sub> = {self.fmt(record.params_before['w_h'][0])} · {self.fmt(c.x, 2)} + ({self.fmt(record.params_before['b_h'][0])}) = <b>{self.fmt(c.z[0])}</b>"
            z2 = f"z<sub>2</sub> = {self.fmt(record.params_before['w_h'][1])} · {self.fmt(c.x, 2)} + ({self.fmt(record.params_before['b_h'][1])}) = <b>{self.fmt(c.z[1])}</b>"
        else:
            z1, z2 = z1_symbolic, z2_symbolic

        if phase in {"forward_output", "loss", "backward_output", "update_output", "backward_hidden", "update_hidden"}:
            t1 = f"t<sub>1</sub> = {self.fmt(record.params_before['w_o'][0])} · {self.fmt(c.h[0])} = <b>{self.fmt(c.t[0])}</b>"
            t2 = f"t<sub>2</sub> = {self.fmt(record.params_before['w_o'][1])} · {self.fmt(c.h[1])} = <b>{self.fmt(c.t[1])}</b>"
        else:
            t1 = "t<sub>1</sub> = w<sub>o,1</sub>h<sub>1</sub>"
            t2 = "t<sub>2</sub> = w<sub>o,2</sub>h<sub>2</sub>"

        anns: List[dict] = []
        box = dict(showarrow=False, font=dict(size=12, color="#172B4D"), bgcolor="#FFFFFF", bordercolor="#D4DEE9", borderwidth=1, borderpad=4)
        anns.extend([
            dict(x=15.5, y=84.5, xref="x", yref="y", text=z1, width=330, **box),
            dict(x=15.5, y=39.2, xref="x", yref="y", text=z2, width=330, **box),
            dict(x=45.2, y=83.5, xref="x", yref="y", text=t1, width=210, **box),
            dict(x=45.2, y=38.2, xref="x", yref="y", text=t2, width=210, **box),
        ])

        main_formula = ""
        if phase == "relu":
            main_formula = (
                f"h<sub>1</sub> = ReLU({self.fmt(c.z[0])}) = <b>{self.fmt(c.h[0])}</b> &nbsp;&nbsp; | &nbsp;&nbsp; "
                f"h<sub>2</sub> = ReLU({self.fmt(c.z[1])}) = <b>{self.fmt(c.h[1])}</b>"
            )
        elif phase == "forward_output":
            main_formula = (
                f"ŷ = t<sub>1</sub> + t<sub>2</sub> + b<sub>o</sub> = "
                f"{self.fmt(c.t[0])} + {self.fmt(c.t[1])} + {self.fmt(record.params_before['b_o'])} = <b>{self.fmt(c.y_hat)}</b>"
            )
        elif phase == "loss":
            main_formula = (
                f"L = (ŷ − y)<sup>2</sup> = ({self.fmt(c.y_hat)} − {self.fmt(c.y)})<sup>2</sup> = <b>{self.fmt(c.loss, 4)}</b>"
            )
        elif phase == "backward_output":
            main_formula = (
                "∂L/∂w<sub>o,i</sub> = 2(ŷ − y)h<sub>i</sub>, &nbsp; "
                f"∂L/∂b<sub>o</sub> = 2(ŷ − y) = <b>{self.fmt(record.gradients.db_o)}</b>"
            )
        elif phase == "update_output":
            main_formula = "w<sub>o</sub> ← w<sub>o</sub> − η∂L/∂w<sub>o</sub>, &nbsp; b<sub>o</sub> ← b<sub>o</sub> − η∂L/∂b<sub>o</sub>"
        elif phase == "backward_hidden":
            blocked_note = "h<sub>2</sub> bị ReLU chặn ⇒ gradient bằng 0" if c.z[1] <= 0 else "cả hai neuron đều cho gradient đi qua"
            main_formula = (
                "∂L/∂w<sub>h,i</sub> = 2(ŷ − y)w<sub>o,i</sub>1[z<sub>i</sub> &gt; 0]x. &nbsp; "
                f"{blocked_note}."
            )
        elif phase == "update_hidden":
            main_formula = "w<sub>h</sub> ← w<sub>h</sub> − η∂L/∂w<sub>h</sub>, &nbsp; b<sub>h</sub> ← b<sub>h</sub> − η∂L/∂b<sub>h</sub>"
        else:
            main_formula = "Các giá trị h<sub>1</sub>, h<sub>2</sub>, ŷ sẽ chỉ hiện sau khi phép tính tương ứng hoàn tất."

        # The main phase formula is shown in the explanation/formula card below,
        # leaving the network panel uncluttered and preventing overlap with lower edges.
        return anns

    def parameter_annotations(self, params: Dict[str, np.ndarray | float]) -> List[dict]:
        w_h = np.asarray(params["w_h"], dtype=float)
        b_h = np.asarray(params["b_h"], dtype=float)
        w_o = np.asarray(params["w_o"], dtype=float)
        b_o = float(params["b_o"])
        items = [
            (21, 66.0, f"w<sub>h,1</sub><br>{self.fmt(w_h[0], 2)}"),
            (21, 58.1, f"w<sub>h,2</sub><br>{self.fmt(w_h[1], 2)}"),
            (32, 85.8, f"b<sub>h,1</sub><br>{self.fmt(b_h[0], 2)}"),
            (32, 45.8, f"b<sub>h,2</sub><br>{self.fmt(b_h[1], 2)}"),
            (45, 66.5, f"w<sub>o,1</sub><br>{self.fmt(w_o[0], 2)}"),
            (45, 57.7, f"w<sub>o,2</sub><br>{self.fmt(w_o[1], 2)}"),
            (56, 77.8, f"b<sub>o</sub><br>{self.fmt(b_o, 2)}"),
        ]
        return [
            dict(x=x, y=y, xref="x", yref="y", text=text, showarrow=False,
                 width=60, font=dict(size=10.5, color="#24405E"), bgcolor="rgba(255,255,255,0.88)", bordercolor="rgba(213,224,234,0.9)", borderwidth=0.7, borderpad=2)
            for x, y, text in items
        ]

    def header_annotations(self, record: StepRecord, phase: str) -> List[dict]:
        return [
            dict(x=4, y=96.8, xref="x", yref="y", text="<b>Mạng thần kinh nhân tạo: nhìn xuyên qua quá trình học của mạng 1‑2‑1</b>",
                 showarrow=False, xanchor="left", font=dict(size=19, color="#0B1E33")),
            dict(x=4, y=93.2, xref="x", yref="y",
                 text=f"<b>Bước:</b> {self.PHASE_TITLES[phase]} &nbsp; | &nbsp; <b>Epoch:</b> {record.epoch}/{self.epochs} &nbsp; | &nbsp; <b>Mẫu:</b> {record.sample_index}/5 &nbsp; | &nbsp; <b>x:</b> {self.fmt(record.x, 2)} &nbsp; | &nbsp; <b>y:</b> {self.fmt(record.y, 3)} &nbsp; | &nbsp; <b>η:</b> {self.fmt(self.lr, 2)}",
                 showarrow=False, xanchor="left", font=dict(size=13, color="#26384D")),
            dict(x=60.5, y=87.3, xref="x", yref="y", text="<b>Panel mạng MLP</b>", showarrow=False,
                 xanchor="right", font=dict(size=13, color="#24405E")),
            dict(x=66.2, y=87.3, xref="x", yref="y", text="<b>Dự đoán và mục tiêu</b>", showarrow=False,
                 xanchor="left", font=dict(size=13, color="#24405E")),
            dict(x=66.2, y=41.0, xref="x", yref="y", text="<b>Theo dõi gradient</b>", showarrow=False,
                 xanchor="left", font=dict(size=13, color="#24405E")),
            dict(x=4, y=30.4, xref="x", yref="y", text="<b>Giải thích bước hiện tại</b>", showarrow=False,
                 xanchor="left", font=dict(size=13, color="#24405E")),
        ]

    def explanation_for_phase(self, record: StepRecord, phase: str) -> str:
        c = record.cache
        phase_text = {
            "load": f"Epoch {record.epoch}/{self.epochs}, mẫu {record.sample_index}/5. Nạp điểm dữ liệu. Lúc này chỉ có x và y là đã biết; h<sub>1</sub>, h<sub>2</sub>, ŷ chưa hiện số vì chưa được tính.",
            "forward_hidden": "Truyền tới từ x sang lớp ẩn. Mỗi neuron ẩn tính z<sub>i</sub> = w<sub>h,i</sub>x + b<sub>h,i</sub>;<br>số chỉ xuất hiện khi phép tính hoàn tất.",
            "relu": "Áp dụng ReLU. Nếu z<sub>i</sub> &gt; 0, neuron sáng viền vàng và h<sub>i</sub> = z<sub>i</sub>;<br>nếu z<sub>i</sub> ≤ 0, tín hiệu bị chặn và h<sub>i</sub> = 0.",
            "forward_output": "Tín hiệu từ hai neuron ẩn đi tới đầu ra. Mạng cộng hai đóng góp t<sub>1</sub>,<br>t<sub>2</sub> và độ lệch b<sub>o</sub> để tạo dự đoán ŷ.",
            "loss": "Tính mất mát L = (ŷ − y)<sup>2</sup>. Trong đồ thị bên phải, đường cam là hàm hiện tại của mạng,<br>còn đoạn đỏ là sai số tại mẫu đang học.",
            "backward_output": "Lan truyền ngược từ đầu ra. Gradient cho biết cần tăng hay giảm tham số;<br>thanh xanh lệch phải là dương, thanh đỏ lệch trái là âm.",
            "update_output": "Cập nhật tham số đầu ra bằng w ← w − η∂L/∂w và b ← b − η∂L/∂b.<br>Đường cam đổi hình vì hàm của mạng đã thay đổi sau cập nhật.",
            "backward_hidden": "Gradient tiếp tục đi ngược về lớp ẩn. Nếu một ReLU bị chặn, hệ số 1[z<sub>i</sub> &gt; 0] bằng 0<br> nên gradient của neuron đó không đi tiếp.",
            "update_hidden": "Cập nhật trọng số và độ lệch của lớp ẩn. Sau bước này, mạng sẵn sàng học mẫu kế tiếp trong epoch hiện tại.",
        }
        active_state = (
            f"Hiện tại: z<sub>1</sub> = {self.fmt(c.z[0])} ({'mở' if c.z[0] > 0 else 'chặn'}), "
            f"z<sub>2</sub> = {self.fmt(c.z[1])} ({'mở' if c.z[1] > 0 else 'chặn'}), "
            f"ŷ = {self.fmt(c.y_hat)}, L = {self.fmt(c.loss, 4)}."
        )
        if phase in {"load", "forward_hidden", "relu"}:
            active_state = ""
        return phase_text[phase] + ("<br>" + active_state if active_state else "")

    def legend_text(self) -> str:
        return (
            "<b>Chú giải nhanh:</b><br>"
            "• Neuron dùng cùng họ màu xanh; giá trị lớn hơn thì sáng hơn.<br>"
            "• Viền vàng dày: ReLU mở; viền xám mỏng: ReLU bị chặn.<br>"
            "• Đường nối/bong bóng lớn hơn: |w| hoặc |b| lớn hơn."
        )

    def gradient_formula_text(self, record: StepRecord, phase: str) -> str:
        c = record.cache
        if phase == "relu":
            return (
                "<b>Công thức ReLU:</b><br>"
                f"h<sub>1</sub> = ReLU({self.fmt(c.z[0])}) = <b>{self.fmt(c.h[0])}</b><br>"
                f"h<sub>2</sub> = ReLU({self.fmt(c.z[1])}) = <b>{self.fmt(c.h[1])}</b>"
            )
        if phase == "forward_output":
            return (
                "<b>Công thức đầu ra:</b><br>"
                f"ŷ = t<sub>1</sub> + t<sub>2</sub> + b<sub>o</sub><br>"
                f"ŷ = {self.fmt(c.t[0])} + {self.fmt(c.t[1])} + {self.fmt(record.params_before['b_o'])} = <b>{self.fmt(c.y_hat)}</b>"
            )
        if phase == "loss":
            return (
                "<b>Công thức mất mát:</b><br>"
                f"L = (ŷ − y)<sup>2</sup><br>"
                f"L = ({self.fmt(c.y_hat)} − {self.fmt(c.y)})<sup>2</sup> = <b>{self.fmt(c.loss, 4)}</b>"
            )
        if phase == "update_output":
            return (
                "<b>Quy tắc cập nhật đầu ra:</b><br>"
                "w<sub>o</sub> ← w<sub>o</sub> − η∂L/∂w<sub>o</sub><br>"
                "b<sub>o</sub> ← b<sub>o</sub> − η∂L/∂b<sub>o</sub>"
            )
        if phase == "update_hidden":
            return (
                "<b>Quy tắc cập nhật lớp ẩn:</b><br>"
                "w<sub>h</sub> ← w<sub>h</sub> − η∂L/∂w<sub>h</sub><br>"
                "b<sub>h</sub> ← b<sub>h</sub> − η∂L/∂b<sub>h</sub>"
            )
        if phase == "backward_output":
            return (
                "<b>Công thức gradient đầu ra:</b><br>"
                "∂L/∂w<sub>o,i</sub> = 2(ŷ − y)h<sub>i</sub><br>"
                f"∂L/∂b<sub>o</sub> = 2(ŷ − y) = {self.fmt(record.gradients.db_o)}"
            )
        if phase == "backward_hidden":
            gate = "0" if c.z[1] <= 0 else "1"
            return (
                "<b>Công thức gradient lớp ẩn:</b><br>"
                "∂L/∂w<sub>h,i</sub> = 2(ŷ − y)w<sub>o,i</sub>1[z<sub>i</sub> &gt; 0]x<br>"
                f"Với h<sub>2</sub>, 1[z<sub>2</sub> &gt; 0] = {gate}."
            )
        return "<b>Gợi ý:</b><br>Dùng Play để xem từng pha.<br>Kéo thanh trượt để chuyển nhanh giữa các mẫu E1‑M1 đến E5‑M5."

    def explanation_annotations(self, record: StepRecord, phase: str) -> List[dict]:
        return [
            dict(x=4, y=23.4, xref="x", yref="y", text=self.explanation_for_phase(record, phase),
                 showarrow=False, xanchor="left", align="left", width=780, font=dict(size=12, color="#172B4D"),
                 bgcolor="#FFFFFF", bordercolor="#D4DEE9", borderwidth=1, borderpad=4),
            dict(x=4, y=11.8, xref="x", yref="y", text=self.legend_text(),
                 showarrow=False, xanchor="left", align="left", width=520, font=dict(size=12, color="#26384D"),
                 bgcolor="#F4F9FF", bordercolor="#D4DEE9", borderwidth=1, borderpad=8),
            dict(x=65.5, y=11.8, xref="x", yref="y", text=self.gradient_formula_text(record, phase),
                 showarrow=False, xanchor="left", align="left", width=430, font=dict(size=12, color="#26384D"),
                 bgcolor="#FFFDF4", bordercolor="#F2C94C", borderwidth=1, borderpad=8),
        ]

    def update_card_annotations(self, record: StepRecord, phase: str) -> List[dict]:
        if phase not in {"update_output", "update_hidden"}:
            return []
        pb = record.params_before
        if phase == "update_output":
            pa = record.params_after_output
            title = "Đang cập nhật w<sub>o,1</sub>, w<sub>o,2</sub>, b<sub>o</sub>"
            lines = [
                f"w<sub>o,1</sub>: {self.fmt(pb['w_o'][0])} → <b>{self.fmt(pa['w_o'][0])}</b>",
                f"w<sub>o,2</sub>: {self.fmt(pb['w_o'][1])} → <b>{self.fmt(pa['w_o'][1])}</b>",
                f"b<sub>o</sub>: {self.fmt(pb['b_o'])} → <b>{self.fmt(pa['b_o'])}</b>",
            ]
            x, y = 50.3, 47.5
        else:
            pa = record.params_after
            title = "Đang cập nhật w<sub>h</sub>, b<sub>h</sub>"
            lines = [
                f"w<sub>h,1</sub>: {self.fmt(pb['w_h'][0])} → <b>{self.fmt(pa['w_h'][0])}</b>",
                f"b<sub>h,1</sub>: {self.fmt(pb['b_h'][0])} → <b>{self.fmt(pa['b_h'][0])}</b>",
                f"w<sub>h,2</sub>: {self.fmt(pb['w_h'][1])} → <b>{self.fmt(pa['w_h'][1])}</b>",
                f"b<sub>h,2</sub>: {self.fmt(pb['b_h'][1])} → <b>{self.fmt(pa['b_h'][1])}</b>",
            ]
            x, y = 12.5, 49.5
        text = f"<b>{title}</b><br><span style='font-size:11px'>Delta shadow</span><br>" + "<br>".join(lines)
        return [
            dict(x=x, y=y, xref="x", yref="y", text=text, showarrow=False,
                 xanchor="center", align="left", width=185, font=dict(size=11.5, color="#172B4D"),
                 bgcolor="#FFF8D6", bordercolor="#F2C94C", borderwidth=1.4, borderpad=8)
        ]

    def layout_for_frame(self, record: StepRecord, phase: str) -> dict:
        shapes = self.static_panel_shapes() + self.relu_chart_shapes()
        gradient_shapes, gradient_annotations = self.gradient_shapes(record, phase)
        shapes.extend(gradient_shapes)
        annotations = []
        annotations.extend(self.header_annotations(record, phase))
        annotations.extend(self.formula_annotations(record, phase))
        annotations.extend(self.parameter_annotations(self.params_for_phase(record, phase)))
        annotations.extend(gradient_annotations)
        annotations.extend(self.explanation_annotations(record, phase))
        annotations.extend(self.update_card_annotations(record, phase))
        annotations.extend([
        ])
        return dict(shapes=shapes, annotations=annotations)

    # ---------- Build and save ----------

    def frame_name(self, record: StepRecord, phase: str) -> str:
        return f"E{record.epoch}_M{record.sample_index}_{phase}"

    def build(self) -> go.Figure:
        first_record = self.records[0]
        first_phase = self.PHASES[0]
        fig = go.Figure(data=self.make_traces(first_record, first_phase))

        frames: List[go.Frame] = []
        for record in self.records:
            for phase in self.PHASES:
                frames.append(
                    go.Frame(
                        name=self.frame_name(record, phase),
                        data=self.make_traces(record, phase),
                        layout=self.layout_for_frame(record, phase),
                    )
                )
        fig.frames = frames

        slider_steps = []
        for record in self.records:
            name = self.frame_name(record, "load")
            slider_steps.append(
                dict(
                    method="animate",
                    label=f"E{record.epoch}-M{record.sample_index}",
                    args=[[name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
                )
            )

        fig.update_layout(
            width=1500,
            height=920,
            margin=dict(l=18, r=18, t=18, b=76),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#EDF4FA",
            showlegend=True,
            legend=dict(
                x=0.682, y=0.555, xanchor="left", yanchor="top",
                orientation="v", bgcolor="rgba(255,255,255,0.85)", bordercolor="#D4DEE9", borderwidth=1,
                font=dict(size=10)
            ),
            xaxis=dict(range=[0, 100], visible=False, fixedrange=True, domain=[0, 1]),
            yaxis=dict(range=[0, 100], visible=False, fixedrange=True, domain=[0, 1]),
            xaxis2=dict(
                domain=[0.682, 0.965], anchor="y2", title=dict(text="x", font=dict(size=11)), range=[min(self.x_grid), max(self.x_grid)],
                showgrid=True, gridcolor="#E7EEF5", zeroline=False, fixedrange=True,
                tickfont=dict(size=10), linecolor="#9AA8B6", mirror=True
            ),
            yaxis2=dict(
                domain=[0.585, 0.835], anchor="x2", title=dict(text="y / ŷ", font=dict(size=11)), range=[self.y_min, self.y_max],
                showgrid=True, gridcolor="#E7EEF5", zeroline=False, fixedrange=True,
                tickfont=dict(size=10), linecolor="#9AA8B6", mirror=True
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.02,
                    y=-0.055,
                    xanchor="left",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 780, "redraw": True}, "fromcurrent": True, "transition": {"duration": 180}, "mode": "immediate"}],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                        ),
                    ],
                    bgcolor="#FFFFFF",
                    bordercolor="#C9D6E2",
                    borderwidth=1,
                    pad=dict(r=8, t=2),
                )
            ],
            sliders=[
                dict(
                    active=0,
                    currentvalue=dict(prefix="Mẫu: ", font=dict(size=12, color="#172B4D"), xanchor="right"),
                    pad=dict(t=44, b=8, l=120, r=30),
                    x=0.12,
                    y=-0.055,
                    len=0.84,
                    bgcolor="#FFFFFF",
                    bordercolor="#C9D6E2",
                    borderwidth=1,
                    ticklen=3,
                    steps=slider_steps,
                    font=dict(size=9),
                )
            ],
            **self.layout_for_frame(first_record, first_phase),
        )
        return fig

    def save(self) -> None:
        fig = self.build()
        fig.write_html(
            self.output,
            include_plotlyjs=True,
            full_html=True,
            config={"displayModeBar": False, "responsive": True},
            auto_play=False,
        )
        if self.auto_open:
            try:
                webbrowser.open("file://" + os.path.abspath(self.output))
            except Exception:
                pass


# ============================================================
# CLI
# ============================================================

def make_dataset() -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([1.20, -1.30, -0.40, 0.55, 1.85], dtype=float)
    noise = np.array([0.020, -0.025, 0.010, -0.015, 0.030], dtype=float)
    y = 0.45 * x ** 2 - 0.10 * x + 0.18 + noise
    return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a standalone Plotly Glass-Box MLP animation.")
    parser.add_argument("--output", default="glass_box_mlp.html", help="Output HTML file path.")
    parser.add_argument("--lr", type=float, default=0.08, help="Learning rate η. Must be positive.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs. Must be positive.")
    parser.add_argument("--no-open", action="store_true", help="Do not automatically open the generated HTML.")
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.lr <= 0:
        raise ValueError("learning rate must be positive")
    return args


def console_safe(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding)


def main() -> None:
    args = parse_args()
    x_data, y_data = make_dataset()
    model = GlassBoxMLP(lr=args.lr)

    # Sanity check required by the specification.
    first_cache = model.forward(float(x_data[0]), float(y_data[0]))
    if not (first_cache.z[0] > 0 and first_cache.z[1] <= 0):
        raise RuntimeError("Initialization does not satisfy z1 > 0 and z2 <= 0 for x = 1.20")

    records = model.train_records(x_data, y_data, args.epochs)
    builder = GlassBoxPlotBuilder(
        records=records,
        x_data=x_data,
        y_data=y_data,
        epochs=args.epochs,
        lr=args.lr,
        output=args.output,
        auto_open=not args.no_open,
    )
    builder.save()
    print(f"Saved interactive animation to: {console_safe(os.path.abspath(args.output))}")
    print(f"Total training updates: {len(records)} = {args.epochs} epochs x {len(x_data)} samples")


if __name__ == "__main__":
    main()
    
    
    
