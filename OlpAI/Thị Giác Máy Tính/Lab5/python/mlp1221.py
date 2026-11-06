#!/usr/bin/env python3
"""
Glass-Box 1-2-2-1 MLP Plotly Animation
--------------------------------------
Creates a standalone HTML dashboard that explains how a tiny 1-2-2-1 MLP
learns from data through forward pass, two ReLU layers, loss, backpropagation,
and parameter updates.

Run:
    python mlp1221.py --output glass_box_mlp_1221.html --epochs 5 --lr 0.08
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import webbrowser
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go


@dataclass
class Neuron:
    name: str
    x: float
    y: float
    value: Optional[float] = None


@dataclass
class ForwardCache:
    x: float
    y: float
    z1: np.ndarray
    h1: np.ndarray
    z2: np.ndarray
    h2: np.ndarray
    t: np.ndarray
    y_hat: float
    loss: float


@dataclass
class Gradients:
    dw1: np.ndarray
    db1: np.ndarray
    dW2: np.ndarray
    db2: np.ndarray
    dw3: np.ndarray
    db3: float
    dz1: np.ndarray
    dz2: np.ndarray
    dh1: np.ndarray
    dh2: np.ndarray
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
    params_after_hidden2: Dict[str, np.ndarray | float]
    params_after: Dict[str, np.ndarray | float]


class GlassBoxMLP1221:
    """Small 1-2-2-1 MLP with two ReLU hidden layers and squared loss."""

    def __init__(self, lr: float = 0.08) -> None:
        self.lr = lr
        self.w1 = np.array([0.85, -0.70], dtype=float)
        self.b1 = np.array([-0.15, 0.20], dtype=float)
        self.W2 = np.array([[0.65, -0.35], [0.40, 0.75]], dtype=float)
        self.b2 = np.array([0.05, -0.10], dtype=float)
        self.w3 = np.array([0.70, -0.45], dtype=float)
        self.b3 = 0.05

    def snapshot(self) -> Dict[str, np.ndarray | float]:
        return {
            "w1": self.w1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
            "w3": self.w3.copy(),
            "b3": float(self.b3),
        }

    @staticmethod
    def with_params(params: Dict[str, np.ndarray | float], x_grid: np.ndarray) -> np.ndarray:
        w1 = np.asarray(params["w1"], dtype=float)
        b1 = np.asarray(params["b1"], dtype=float)
        W2 = np.asarray(params["W2"], dtype=float)
        b2 = np.asarray(params["b2"], dtype=float)
        w3 = np.asarray(params["w3"], dtype=float)
        b3 = float(params["b3"])
        z1 = x_grid[:, None] * w1[None, :] + b1[None, :]
        h1 = np.maximum(0.0, z1)
        z2 = h1 @ W2 + b2
        h2 = np.maximum(0.0, z2)
        return h2 @ w3 + b3

    def forward(self, x: float, y: float) -> ForwardCache:
        z1 = self.w1 * x + self.b1
        h1 = np.maximum(0.0, z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(0.0, z2)
        t = self.w3 * h2
        y_hat = float(np.sum(t) + self.b3)
        loss = float((y_hat - y) ** 2)
        return ForwardCache(x=x, y=y, z1=z1, h1=h1, z2=z2, h2=h2, t=t, y_hat=y_hat, loss=loss)

    def backward(self, cache: ForwardCache) -> Gradients:
        d_y_hat = 2.0 * (cache.y_hat - cache.y)
        dw3 = d_y_hat * cache.h2
        db3 = float(d_y_hat)
        dh2 = d_y_hat * self.w3
        dz2 = dh2 * (cache.z2 > 0).astype(float)
        dW2 = np.outer(cache.h1, dz2)
        db2 = dz2
        dh1 = self.W2 @ dz2
        dz1 = dh1 * (cache.z1 > 0).astype(float)
        dw1 = dz1 * cache.x
        db1 = dz1
        return Gradients(
            dw1=dw1,
            db1=db1,
            dW2=dW2,
            db2=db2,
            dw3=dw3,
            db3=db3,
            dz1=dz1,
            dz2=dz2,
            dh1=dh1,
            dh2=dh2,
            d_y_hat=float(d_y_hat),
        )

    @staticmethod
    def clone_params(params: Dict[str, np.ndarray | float]) -> Dict[str, np.ndarray | float]:
        return {
            "w1": np.asarray(params["w1"], dtype=float).copy(),
            "b1": np.asarray(params["b1"], dtype=float).copy(),
            "W2": np.asarray(params["W2"], dtype=float).copy(),
            "b2": np.asarray(params["b2"], dtype=float).copy(),
            "w3": np.asarray(params["w3"], dtype=float).copy(),
            "b3": float(params["b3"]),
        }

    @classmethod
    def output_updated_params(
        cls, params_before: Dict[str, np.ndarray | float], gradients: Gradients, lr: float
    ) -> Dict[str, np.ndarray | float]:
        out = cls.clone_params(params_before)
        out["w3"] = np.asarray(out["w3"], dtype=float) - lr * gradients.dw3
        out["b3"] = float(out["b3"]) - lr * gradients.db3
        return out

    @classmethod
    def hidden2_updated_params(
        cls, params_before: Dict[str, np.ndarray | float], gradients: Gradients, lr: float
    ) -> Dict[str, np.ndarray | float]:
        out = cls.output_updated_params(params_before, gradients, lr)
        out["W2"] = np.asarray(out["W2"], dtype=float) - lr * gradients.dW2
        out["b2"] = np.asarray(out["b2"], dtype=float) - lr * gradients.db2
        return out

    @classmethod
    def all_updated_params(
        cls, params_before: Dict[str, np.ndarray | float], gradients: Gradients, lr: float
    ) -> Dict[str, np.ndarray | float]:
        out = cls.hidden2_updated_params(params_before, gradients, lr)
        out["w1"] = np.asarray(out["w1"], dtype=float) - lr * gradients.dw1
        out["b1"] = np.asarray(out["b1"], dtype=float) - lr * gradients.db1
        return out

    def apply_gradients(self, gradients: Gradients) -> None:
        self.w3 = self.w3 - self.lr * gradients.dw3
        self.b3 = self.b3 - self.lr * gradients.db3
        self.W2 = self.W2 - self.lr * gradients.dW2
        self.b2 = self.b2 - self.lr * gradients.db2
        self.w1 = self.w1 - self.lr * gradients.dw1
        self.b1 = self.b1 - self.lr * gradients.db1

    def train_records(self, x_data: np.ndarray, y_data: np.ndarray, epochs: int) -> List[StepRecord]:
        records: List[StepRecord] = []
        for epoch in range(1, epochs + 1):
            for sample_index, (x, y) in enumerate(zip(x_data, y_data), start=1):
                params_before = self.snapshot()
                cache = self.forward(float(x), float(y))
                gradients = self.backward(cache)
                records.append(
                    StepRecord(
                        epoch=epoch,
                        sample_index=sample_index,
                        x=float(x),
                        y=float(y),
                        params_before=params_before,
                        cache=cache,
                        gradients=gradients,
                        params_after_output=self.output_updated_params(params_before, gradients, self.lr),
                        params_after_hidden2=self.hidden2_updated_params(params_before, gradients, self.lr),
                        params_after=self.all_updated_params(params_before, gradients, self.lr),
                    )
                )
                self.apply_gradients(gradients)
        return records


class GlassBoxPlotBuilder1221:
    PHASES = [
        "load",
        "forward_hidden1",
        "relu1",
        "forward_hidden2",
        "relu2",
        "forward_output",
        "loss",
        "backward_output",
        "update_output",
        "backward_hidden2",
        "update_hidden2",
        "backward_hidden1",
        "update_hidden1",
    ]

    PHASE_TITLES = {
        "load": "Nạp mẫu dữ liệu",
        "forward_hidden1": "Tính z ở lớp ẩn 1",
        "relu1": "Áp dụng ReLU ở lớp ẩn 1",
        "forward_hidden2": "Tính z ở lớp ẩn 2",
        "relu2": "Áp dụng ReLU ở lớp ẩn 2",
        "forward_output": "Tính đầu ra ŷ",
        "loss": "Tính mất mát",
        "backward_output": "Lan truyền ngược từ đầu ra",
        "update_output": "Cập nhật tham số đầu ra",
        "backward_hidden2": "Lan truyền ngược về lớp ẩn 2",
        "update_hidden2": "Cập nhật lớp ẩn 2",
        "backward_hidden1": "Lan truyền ngược về lớp ẩn 1",
        "update_hidden1": "Cập nhật lớp ẩn 1",
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
        self.y_curve_all = [GlassBoxMLP1221.with_params(r.params_before, self.x_grid) for r in records]
        self.y_min = float(min(np.min(y_data), min(np.min(v) for v in self.y_curve_all)) - 0.25)
        self.y_max = float(max(np.max(y_data), max(np.max(v) for v in self.y_curve_all)) + 0.25)
        self.input_neuron = Neuron("x", 8, 62)
        self.hidden1 = [Neuron("h1_1", 28, 72), Neuron("h1_2", 28, 52)]
        self.hidden2 = [Neuron("h2_1", 48, 72), Neuron("h2_2", 48, 52)]
        self.output_neuron = Neuron("y_hat", 68, 62)
        self.relu_chart_boxes = self.activation_chart_boxes()

    @staticmethod
    def fmt(v: float, digits: int = 3) -> str:
        return f"{v:.{digits}f}"

    @staticmethod
    def signed(v: float, digits: int = 3) -> str:
        return f"{v:+.{digits}f}"

    @staticmethod
    def rgba(hex_color: str, opacity: float) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{opacity:.3f})"

    @staticmethod
    def neuron_fill(value: Optional[float], vmin: float = -0.5, vmax: float = 2.0) -> str:
        if value is None:
            return "#EAF4FF"
        t = max(0.0, min(1.0, (float(value) - vmin) / (vmax - vmin)))
        return f"rgb({int(18 + 70 * t)},{int(72 + 170 * t)},{int(145 + 95 * t)})"

    @staticmethod
    def edge_width(w: float) -> float:
        return 2.0 + min(4.5, abs(float(w)) * 3.6)

    @staticmethod
    def edge_opacity(w: float) -> float:
        return 0.24 + min(0.62, abs(float(w)) * 0.50)

    @staticmethod
    def param_size(v: float) -> float:
        return 13 + min(14, abs(float(v)) * 13)

    def activation_chart_boxes(self) -> List[Tuple[float, float, float, float]]:
        old_w, old_h = 10.0, 7.0
        new_w, new_h = old_w * 0.7, old_h * 0.7
        old_boxes = [
            (34.0, 80.0, old_w, old_h),
            (34.0, 39.0, old_w, old_h),
            (54.0, 80.0, old_w, old_h),
            (54.0, 39.0, old_w, old_h),
        ]
        neurons = [(28.0, 72.0), (28.0, 52.0), (48.0, 72.0), (48.0, 52.0)]
        boxes: List[Tuple[float, float, float, float]] = []
        for (old_x0, old_y0, w, h), (nx, ny) in zip(old_boxes, neurons):
            old_cx = old_x0 + w / 2
            old_cy = old_y0 + h / 2
            new_cx = nx + (old_cx - nx) / 2
            new_cy = ny + (old_cy - ny) / 2
            boxes.append((new_cx - new_w / 2, new_cy - new_h / 2, new_w, new_h))
        return boxes

    def clone_params(self, params: Dict[str, np.ndarray | float]) -> Dict[str, np.ndarray | float]:
        return GlassBoxMLP1221.clone_params(params)

    def params_for_phase(self, record: StepRecord, phase: str) -> Dict[str, np.ndarray | float]:
        if phase == "update_output":
            return self.clone_params(record.params_after_output)
        if phase == "update_hidden2":
            return self.clone_params(record.params_after_hidden2)
        if phase == "update_hidden1":
            return self.clone_params(record.params_after)
        return self.clone_params(record.params_before)

    def visible_values(self, record: StepRecord, phase: str) -> Dict[str, Optional[float]]:
        c = record.cache
        values: Dict[str, Optional[float]] = {
            "x": record.x,
            "h11": None,
            "h12": None,
            "h21": None,
            "h22": None,
            "yhat": None,
        }
        if phase in {
            "relu1",
            "forward_hidden2",
            "relu2",
            "forward_output",
            "loss",
            "backward_output",
            "update_output",
            "backward_hidden2",
            "update_hidden2",
            "backward_hidden1",
            "update_hidden1",
        }:
            values["h11"] = float(c.h1[0])
            values["h12"] = float(c.h1[1])
        if phase in {
            "relu2",
            "forward_output",
            "loss",
            "backward_output",
            "update_output",
            "backward_hidden2",
            "update_hidden2",
            "backward_hidden1",
            "update_hidden1",
        }:
            values["h21"] = float(c.h2[0])
            values["h22"] = float(c.h2[1])
        if phase in {
            "forward_output",
            "loss",
            "backward_output",
            "update_output",
            "backward_hidden2",
            "update_hidden2",
            "backward_hidden1",
            "update_hidden1",
        }:
            values["yhat"] = float(c.y_hat)
        return values

    def make_traces(self, record: StepRecord, phase: str) -> List[go.BaseTraceType]:
        params = self.params_for_phase(record, phase)
        traces: List[go.BaseTraceType] = []
        traces.extend(self.edge_traces(params, phase))
        traces.append(self.neuron_trace(record, phase, self.visible_values(record, phase)))
        traces.append(self.parameter_bubble_trace(params))
        traces.append(self.forward_pulse_trace(phase))
        traces.append(self.backward_pulse_trace(phase))
        traces.append(self.update_star_trace(phase))
        traces.extend(self.relu_traces(record, phase))
        traces.extend(self.prediction_traces(record, phase, params))
        return traces

    def edge_traces(self, params: Dict[str, np.ndarray | float], phase: str) -> List[go.Scatter]:
        w1 = np.asarray(params["w1"], dtype=float)
        W2 = np.asarray(params["W2"], dtype=float)
        w3 = np.asarray(params["w3"], dtype=float)
        edge_defs = [
            ("x-h11", (8, 62), (28, 72), w1[0]),
            ("x-h12", (8, 62), (28, 52), w1[1]),
            ("h11-h21", (28, 72), (48, 72), W2[0, 0]),
            ("h11-h22", (28, 72), (48, 52), W2[0, 1]),
            ("h12-h21", (28, 52), (48, 72), W2[1, 0]),
            ("h12-h22", (28, 52), (48, 52), W2[1, 1]),
            ("h21-y", (48, 72), (68, 62), w3[0]),
            ("h22-y", (48, 52), (68, 62), w3[1]),
        ]
        active_names: set[str] = set()
        if phase == "forward_hidden1":
            active_names.update(["x-h11", "x-h12"])
        elif phase == "forward_hidden2":
            active_names.update(["h11-h21", "h11-h22", "h12-h21", "h12-h22"])
        elif phase == "forward_output":
            active_names.update(["h21-y", "h22-y"])
        elif phase == "backward_output":
            active_names.update(["h21-y", "h22-y"])
        elif phase == "backward_hidden2":
            active_names.update(["h11-h21", "h11-h22", "h12-h21", "h12-h22"])
        elif phase == "backward_hidden1":
            active_names.update(["x-h11", "x-h12"])

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
                    line=dict(
                        color=self.rgba(base, self.edge_opacity(float(w)) + (0.16 if is_active else 0.0)),
                        width=self.edge_width(float(w)),
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                    xaxis="x",
                    yaxis="y",
                )
            )
        return traces

    def neuron_trace(self, record: StepRecord, phase: str, visible: Dict[str, Optional[float]]) -> go.Scatter:
        c = record.cache
        labels = [
            f"x<br><b>{self.fmt(record.x, 2)}</b>",
            "h<sub>1,1</sub>" if visible["h11"] is None else f"h<sub>1,1</sub><br><b>{self.fmt(float(visible['h11']))}</b>",
            "h<sub>1,2</sub>" if visible["h12"] is None else f"h<sub>1,2</sub><br><b>{self.fmt(float(visible['h12']))}</b>",
            "h<sub>2,1</sub>" if visible["h21"] is None else f"h<sub>2,1</sub><br><b>{self.fmt(float(visible['h21']))}</b>",
            "h<sub>2,2</sub>" if visible["h22"] is None else f"h<sub>2,2</sub><br><b>{self.fmt(float(visible['h22']))}</b>",
            "ŷ" if visible["yhat"] is None else f"ŷ<br><b>{self.fmt(float(visible['yhat']))}</b>",
        ]
        values = [record.x, visible["h11"], visible["h12"], visible["h21"], visible["h22"], visible["yhat"]]
        colors = [self.neuron_fill(v) for v in values]
        line_colors = ["#276AA3"] * 6
        line_widths = [2] * 6
        if phase in {
            "relu1",
            "forward_hidden2",
            "relu2",
            "forward_output",
            "loss",
            "backward_output",
            "update_output",
            "backward_hidden2",
            "update_hidden2",
            "backward_hidden1",
            "update_hidden1",
        }:
            for idx, z in enumerate(c.z1, start=1):
                line_colors[idx] = "#FFD43B" if z > 0 else "#9EA8B3"
                line_widths[idx] = 7 if z > 0 else 2
        if phase in {
            "relu2",
            "forward_output",
            "loss",
            "backward_output",
            "update_output",
            "backward_hidden2",
            "update_hidden2",
            "backward_hidden1",
            "update_hidden1",
        }:
            for idx, z in enumerate(c.z2, start=3):
                line_colors[idx] = "#FFD43B" if z > 0 else "#9EA8B3"
                line_widths[idx] = 7 if z > 0 else 2
        return go.Scatter(
            x=[8, 28, 28, 48, 48, 68],
            y=[62, 72, 52, 72, 52, 62],
            mode="markers+text",
            marker=dict(size=[54, 58, 58, 58, 58, 58], color=colors, line=dict(color=line_colors, width=line_widths)),
            text=labels,
            textposition="middle center",
            textfont=dict(size=13, color="#0B1E33", family="Arial"),
            hoverinfo="skip",
            showlegend=False,
            xaxis="x",
            yaxis="y",
        )

    def parameter_bubble_trace(self, params: Dict[str, np.ndarray | float]) -> go.Scatter:
        w1 = np.asarray(params["w1"], dtype=float)
        b1 = np.asarray(params["b1"], dtype=float)
        W2 = np.asarray(params["W2"], dtype=float)
        b2 = np.asarray(params["b2"], dtype=float)
        w3 = np.asarray(params["w3"], dtype=float)
        b3 = float(params["b3"])
        vals = [w1[0], w1[1], b1[0], b1[1], W2[0, 0], W2[0, 1], W2[1, 0], W2[1, 1], b2[0], b2[1], w3[0], w3[1], b3]
        xs = [18, 18, 28, 28, 38, 38, 38, 38, 48, 48, 58, 58, 68]
        ys = [68, 56, 82, 42, 75, 68, 56, 49, 82, 42, 68, 56, 75]
        return go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=[self.param_size(v) for v in vals], color="#FFFFFF", line=dict(color="#2B78B8", width=1.2), opacity=0.92),
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
        if phase == "forward_hidden1":
            for p1 in [(28, 72), (28, 52)]:
                x, y = self.interpolate((8, 62), p1, 0.68)
                xs.append(x)
                ys.append(y)
        elif phase == "forward_hidden2":
            for p0 in [(28, 72), (28, 52)]:
                for p1 in [(48, 72), (48, 52)]:
                    x, y = self.interpolate(p0, p1, 0.68)
                    xs.append(x)
                    ys.append(y)
        elif phase == "forward_output":
            for p0 in [(48, 72), (48, 52)]:
                x, y = self.interpolate(p0, (68, 62), 0.68)
                xs.append(x)
                ys.append(y)
        return go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=16, color="#FFB000", line=dict(color="#FFF6D1", width=3), symbol="circle"),
            hoverinfo="skip",
            showlegend=False,
            xaxis="x",
            yaxis="y",
        )

    def backward_pulse_trace(self, phase: str) -> go.Scatter:
        xs: List[float] = []
        ys: List[float] = []
        if phase == "backward_output":
            for p1 in [(48, 72), (48, 52)]:
                x, y = self.interpolate((68, 62), p1, 0.62)
                xs.append(x)
                ys.append(y)
        elif phase == "backward_hidden2":
            for p0 in [(48, 72), (48, 52)]:
                for p1 in [(28, 72), (28, 52)]:
                    x, y = self.interpolate(p0, p1, 0.62)
                    xs.append(x)
                    ys.append(y)
        elif phase == "backward_hidden1":
            for p0 in [(28, 72), (28, 52)]:
                x, y = self.interpolate(p0, (8, 62), 0.62)
                xs.append(x)
                ys.append(y)
        return go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=16, color="#F05478", line=dict(color="#FFE0E8", width=3), symbol="circle"),
            hoverinfo="skip",
            showlegend=False,
            xaxis="x",
            yaxis="y",
        )

    def update_star_trace(self, phase: str) -> go.Scatter:
        xs: List[float] = []
        ys: List[float] = []
        if phase == "update_output":
            xs, ys = [58, 58, 68], [68, 56, 75]
        elif phase == "update_hidden2":
            xs, ys = [38, 38, 38, 38, 48, 48], [75, 68, 56, 49, 82, 42]
        elif phase == "update_hidden1":
            xs, ys = [18, 18, 28, 28], [68, 56, 82, 42]
        return go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=20, color="#FFD23C", line=dict(color="#9A6B00", width=1.4), symbol="star"),
            hoverinfo="skip",
            showlegend=False,
            xaxis="x",
            yaxis="y",
        )

    def relu_traces(self, record: StepRecord, phase: str) -> List[go.Scatter]:
        traces: List[go.Scatter] = []
        z_points = np.array([-1.0, 0.0, 1.0])
        relu_points = np.maximum(0, z_points)
        z_values = [record.cache.z1[0], record.cache.z1[1], record.cache.z2[0], record.cache.z2[1]]
        visible = [
            phase in {
                "forward_hidden1",
                "relu1",
                "forward_hidden2",
                "relu2",
                "forward_output",
                "loss",
                "backward_output",
                "update_output",
                "backward_hidden2",
                "update_hidden2",
                "backward_hidden1",
                "update_hidden1",
            },
            phase in {
                "forward_hidden1",
                "relu1",
                "forward_hidden2",
                "relu2",
                "forward_output",
                "loss",
                "backward_output",
                "update_output",
                "backward_hidden2",
                "update_hidden2",
                "backward_hidden1",
                "update_hidden1",
            },
            phase in {
                "forward_hidden2",
                "relu2",
                "forward_output",
                "loss",
                "backward_output",
                "update_output",
                "backward_hidden2",
                "update_hidden2",
                "backward_hidden1",
                "update_hidden1",
            },
            phase in {
                "forward_hidden2",
                "relu2",
                "forward_output",
                "loss",
                "backward_output",
                "update_output",
                "backward_hidden2",
                "update_hidden2",
                "backward_hidden1",
                "update_hidden1",
            },
        ]
        for idx, (x0, y0, w, h) in enumerate(self.relu_chart_boxes):
            xs = x0 + (z_points + 1) / 2 * w
            ys = y0 + relu_points * h
            traces.append(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color="#268BD2", width=2.1),
                    hoverinfo="skip",
                    showlegend=False,
                    xaxis="x",
                    yaxis="y",
                )
            )
            dot_xs: List[float] = []
            dot_ys: List[float] = []
            if visible[idx]:
                z = float(z_values[idx])
                z_clip = max(-1.0, min(1.0, z))
                dot_xs = [x0 + (z_clip + 1) / 2 * w]
                dot_ys = [y0 + max(0.0, z_clip) * h]
            traces.append(
                go.Scatter(
                    x=dot_xs,
                    y=dot_ys,
                    mode="markers",
                    marker=dict(size=8, color="#FFD43B" if z_values[idx] > 0 else "#9EA8B3", line=dict(color="#263238", width=1)),
                    hoverinfo="skip",
                    showlegend=False,
                    xaxis="x",
                    yaxis="y",
                )
            )
        return traces

    def prediction_traces(self, record: StepRecord, phase: str, params: Dict[str, np.ndarray | float]) -> List[go.BaseTraceType]:
        y_curve = GlassBoxMLP1221.with_params(params, self.x_grid)
        y_current_model = float(GlassBoxMLP1221.with_params(params, np.array([record.x]))[0])
        show_prediction = phase in {
            "forward_output",
            "loss",
            "backward_output",
            "update_output",
            "backward_hidden2",
            "update_hidden2",
            "backward_hidden1",
            "update_hidden1",
        }
        show_error = phase in {
            "loss",
            "backward_output",
            "update_output",
            "backward_hidden2",
            "update_hidden2",
            "backward_hidden1",
            "update_hidden1",
        }
        return [
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
                name="Dự đoán",
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

    def static_panel_shapes(self) -> List[dict]:
        card = dict(type="rect", xref="x", yref="y", line=dict(color="#D5E0EA", width=1.2), fillcolor="#F8FBFF", layer="below")
        return [
            {**card, "x0": 2, "y0": 91, "x1": 98, "y1": 99, "fillcolor": "#F3F8FF"},
            {**card, "x0": 2, "y0": 35, "x1": 72, "y1": 89, "fillcolor": "#FBFDFF"},
            {**card, "x0": 74, "y0": 45, "x1": 98, "y1": 89, "fillcolor": "#FBFDFF"},
            {**card, "x0": 74, "y0": 12, "x1": 98, "y1": 42, "fillcolor": "#FBFDFF"},
            {**card, "x0": 2, "y0": 2, "x1": 72, "y1": 32, "fillcolor": "#FBFDFF"},
        ]

    def relu_chart_shapes(self) -> List[dict]:
        shapes: List[dict] = []
        for x0, y0, w, h in self.relu_chart_boxes:
            shapes.extend(
                [
                    dict(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=x0 - 0.55,
                        y0=y0 - 0.55,
                        x1=x0 + w + 0.65,
                        y1=y0 + h + 0.65,
                        line=dict(color="#D4DEE9", width=1),
                        fillcolor="#FFFFFF",
                        layer="below",
                    ),
                    dict(type="line", xref="x", yref="y", x0=x0, y0=y0, x1=x0 + w, y1=y0, line=dict(color="#9AA8B6", width=1)),
                    dict(
                        type="line",
                        xref="x",
                        yref="y",
                        x0=x0 + w / 2,
                        y0=y0,
                        x1=x0 + w / 2,
                        y1=y0 + h,
                        line=dict(color="#9AA8B6", width=1),
                    ),
                ]
            )
        return shapes

    def gradient_shapes(self, record: StepRecord, phase: str) -> Tuple[List[dict], List[dict]]:
        shapes: List[dict] = []
        ann: List[dict] = []
        center = 88.0
        x_left = 78.5
        x_right = 97.0
        bar_h = 0.86
        row_y = [38.0, 35.9, 33.8, 31.7, 29.6, 27.5, 25.4, 23.3, 21.2, 19.1, 17.0, 14.9]
        labels = [
            "∂L/∂w<sub>3,1</sub>",
            "∂L/∂w<sub>3,2</sub>",
            "∂L/∂b<sub>3</sub>",
            "∂L/∂W<sub>2,11</sub>",
            "∂L/∂W<sub>2,12</sub>",
            "∂L/∂W<sub>2,21</sub>",
            "∂L/∂W<sub>2,22</sub>",
            "∂L/∂b<sub>2,1</sub>",
            "∂L/∂b<sub>2,2</sub>",
            "∂L/∂w<sub>1,1</sub>",
            "∂L/∂w<sub>1,2</sub>",
            "∂L/∂b<sub>1</sub>",
        ]
        g = record.gradients
        grads = np.array(
            [
                g.dw3[0],
                g.dw3[1],
                g.db3,
                g.dW2[0, 0],
                g.dW2[0, 1],
                g.dW2[1, 0],
                g.dW2[1, 1],
                g.db2[0],
                g.db2[1],
                g.dw1[0],
                g.dw1[1],
                max(abs(g.db1[0]), abs(g.db1[1])) * (1 if np.sum(g.db1) >= 0 else -1),
            ],
            dtype=float,
        )
        mask = np.zeros_like(grads)
        if phase in {"backward_output", "update_output"}:
            mask[:3] = 1
        elif phase in {"backward_hidden2", "update_hidden2"}:
            mask[:9] = 1
        elif phase in {"backward_hidden1", "update_hidden1"}:
            mask[:] = 1
        shown = grads * mask
        max_abs = max(0.04, float(np.max(np.abs(grads))))
        scale = 8.0 / max_abs
        shapes.append(dict(type="line", xref="x", yref="y", x0=center, x1=center, y0=13.7, y1=39.1, line=dict(color="#6B7785", width=1.3)))
        for y in row_y:
            shapes.append(dict(type="line", xref="x", yref="y", x0=x_left, x1=x_right, y0=y, y1=y, line=dict(color="#E5EBF1", width=1)))
        for i, (y, lab, val) in enumerate(zip(row_y, labels, shown)):
            width = 0.0 if abs(val) < 1e-9 else min(8.0, abs(float(val)) * scale)
            color = "#2F80ED" if val > 0 else "#E24A5A"
            if width > 0:
                x1 = center + math.copysign(width, val)
                shapes.append(dict(type="rect", xref="x", yref="y", x0=min(center, x1), x1=max(center, x1), y0=y - bar_h / 2, y1=y + bar_h / 2, line=dict(color=color, width=0), fillcolor=color))
            ann.append(dict(x=75.2, y=y, xref="x", yref="y", text=lab, showarrow=False, font=dict(size=8.5, color="#26384D"), xanchor="left"))
            val_text = "--" if mask[i] == 0 else ("0" if abs(val) < 1e-9 else self.signed(float(val), 2))
            ann.append(dict(x=97.3, y=y, xref="x", yref="y", text=val_text, showarrow=False, font=dict(size=8.5, color="#26384D"), xanchor="right"))
        return shapes, ann

    def parameter_annotations(self, params: Dict[str, np.ndarray | float]) -> List[dict]:
        w1 = np.asarray(params["w1"], dtype=float)
        b1 = np.asarray(params["b1"], dtype=float)
        W2 = np.asarray(params["W2"], dtype=float)
        b2 = np.asarray(params["b2"], dtype=float)
        w3 = np.asarray(params["w3"], dtype=float)
        b3 = float(params["b3"])
        items = [
            (18, 65.7, f"w<sub>1,1</sub><br>{self.fmt(w1[0], 2)}"),
            (18, 58.1, f"w<sub>1,2</sub><br>{self.fmt(w1[1], 2)}"),
            (28, 85.5, f"b<sub>1,1</sub><br>{self.fmt(b1[0], 2)}"),
            (28, 45.5, f"b<sub>1,2</sub><br>{self.fmt(b1[1], 2)}"),
            (38, 77.5, f"W<sub>2,11</sub><br>{self.fmt(W2[0, 0], 2)}"),
            (38, 69.8, f"W<sub>2,12</sub><br>{self.fmt(W2[0, 1], 2)}"),
            (38, 53.6, f"W<sub>2,21</sub><br>{self.fmt(W2[1, 0], 2)}"),
            (38, 46.1, f"W<sub>2,22</sub><br>{self.fmt(W2[1, 1], 2)}"),
            (48, 85.5, f"b<sub>2,1</sub><br>{self.fmt(b2[0], 2)}"),
            (48, 45.5, f"b<sub>2,2</sub><br>{self.fmt(b2[1], 2)}"),
            (58, 65.7, f"w<sub>3,1</sub><br>{self.fmt(w3[0], 2)}"),
            (58, 58.1, f"w<sub>3,2</sub><br>{self.fmt(w3[1], 2)}"),
            (68, 78.5, f"b<sub>3</sub><br>{self.fmt(b3, 2)}"),
        ]
        return [
            dict(
                x=x,
                y=y,
                xref="x",
                yref="y",
                text=text,
                showarrow=False,
                width=72,
                font=dict(size=9, color="#24405E"),
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="rgba(213,224,234,0.9)",
                borderwidth=0.7,
                borderpad=2,
            )
            for x, y, text in items
        ]

    def formula_annotations(self, record: StepRecord, phase: str) -> List[dict]:
        c = record.cache
        p = record.params_before
        box = dict(showarrow=False, font=dict(size=10.5, color="#172B4D"), bgcolor="#FFFFFF", bordercolor="#D4DEE9", borderwidth=1, borderpad=3)
        z11 = "z<sub>1,1</sub> = w<sub>1,1</sub>x + b<sub>1,1</sub>"
        z12 = "z<sub>1,2</sub> = w<sub>1,2</sub>x + b<sub>1,2</sub>"
        z21 = "z<sub>2,1</sub> = h<sub>1</sub> · W<sub>2,:,1</sub> + b<sub>2,1</sub>"
        z22 = "z<sub>2,2</sub> = h<sub>1</sub> · W<sub>2,:,2</sub> + b<sub>2,2</sub>"
        if phase in set(self.PHASES[self.PHASES.index("forward_hidden1") :]):
            z11 = f"z<sub>1,1</sub> = {self.fmt(p['w1'][0])}·{self.fmt(c.x, 2)}+({self.fmt(p['b1'][0])}) = <b>{self.fmt(c.z1[0])}</b>"
            z12 = f"z<sub>1,2</sub> = {self.fmt(p['w1'][1])}·{self.fmt(c.x, 2)}+({self.fmt(p['b1'][1])}) = <b>{self.fmt(c.z1[1])}</b>"
        if phase in set(self.PHASES[self.PHASES.index("forward_hidden2") :]):
            z21 = f"z<sub>2,1</sub> = h<sub>1</sub> · W<sub>2,:,1</sub> + b<sub>2,1</sub> = <b>{self.fmt(c.z2[0])}</b>"
            z22 = f"z<sub>2,2</sub> = h<sub>1</sub> · W<sub>2,:,2</sub> + b<sub>2,2</sub> = <b>{self.fmt(c.z2[1])}</b>"
        return [
            dict(x=16.7, y=84.5, xref="x", yref="y", text=z11, width=250, **box),
            dict(x=16.7, y=39.2, xref="x", yref="y", text=z12, width=250, **box),
            dict(x=58.3, y=84.5, xref="x", yref="y", text=z21, width=275, **box),
            dict(x=58.3, y=39.2, xref="x", yref="y", text=z22, width=275, **box),
        ]

    def header_annotations(self, record: StepRecord, phase: str) -> List[dict]:
        return [
            dict(x=4, y=96.8, xref="x", yref="y", text="<b>Glass-Box MLP: nhìn xuyên qua quá trình học của mạng 1-2-2-1</b>", showarrow=False, xanchor="left", font=dict(size=19, color="#0B1E33")),
            dict(
                x=4,
                y=93.2,
                xref="x",
                yref="y",
                text=f"<b>Bước:</b> {self.PHASE_TITLES[phase]} &nbsp; | &nbsp; <b>Epoch:</b> {record.epoch}/{self.epochs} &nbsp; | &nbsp; <b>Mẫu:</b> {record.sample_index}/5 &nbsp; | &nbsp; <b>x:</b> {self.fmt(record.x, 2)} &nbsp; | &nbsp; <b>y:</b> {self.fmt(record.y, 3)} &nbsp; | &nbsp; <b>η:</b> {self.fmt(self.lr, 2)}",
                showarrow=False,
                xanchor="left",
                font=dict(size=13, color="#26384D"),
            ),
            dict(x=70.5, y=87.3, xref="x", yref="y", text="<b>Panel mạng MLP</b>", showarrow=False, xanchor="right", font=dict(size=13, color="#24405E")),
            dict(x=75.2, y=87.3, xref="x", yref="y", text="<b>Dự đoán và mục tiêu</b>", showarrow=False, xanchor="left", font=dict(size=13, color="#24405E")),
            dict(x=75.2, y=41.0, xref="x", yref="y", text="<b>Theo dõi gradient</b>", showarrow=False, xanchor="left", font=dict(size=13, color="#24405E")),
            dict(x=4, y=30.4, xref="x", yref="y", text="<b>Giải thích bước hiện tại</b>", showarrow=False, xanchor="left", font=dict(size=13, color="#24405E")),
        ]

    def explanation_for_phase(self, record: StepRecord, phase: str) -> str:
        c = record.cache
        phase_text = {
            "load": "Nạp một điểm dữ liệu. Lúc này chỉ có x và y là đã biết; các kích hoạt ẩn chưa được tính.",
            "forward_hidden1": "Tín hiệu x đi tới lớp ẩn 1. Mỗi neuron tính z<sub>1,i</sub> = w<sub>1,i</sub>x + b<sub>1,i</sub>.",
            "relu1": "Lớp ẩn 1 áp dụng ReLU: z dương được truyền qua, còn z âm trở thành 0.",
            "forward_hidden2": "Hai kích hoạt h1 đi qua ma trận trọng số 2x2 để tới lớp ẩn 2.",
            "relu2": "Lớp ẩn 2 áp dụng ReLU và tạo ra h<sub>2,1</sub>, h<sub>2,2</sub>.",
            "forward_output": "Neuron đầu ra kết hợp h<sub>2,1</sub>, h<sub>2,2</sub> và b<sub>3</sub> để tạo dự đoán ŷ.",
            "loss": "Mất mát bình phương L = (ŷ - y)<sup>2</sup> đo sai số của mẫu hiện tại.",
            "backward_output": "Tính gradient cho w<sub>3</sub> và b<sub>3</sub> từ sai số ở đầu ra.",
            "update_output": "Cập nhật tham số đầu ra bằng quy tắc tham số ← tham số − η·gradient.",
            "backward_hidden2": "Gradient đi qua w<sub>3</sub> và cổng ReLU thứ hai để tới W<sub>2</sub> và b<sub>2</sub>.",
            "update_hidden2": "Cập nhật các trọng số W<sub>2</sub> và độ lệch b<sub>2</sub> của lớp giữa.",
            "backward_hidden1": "Gradient tiếp tục đi qua W<sub>2</sub> và cổng ReLU thứ nhất để tới w<sub>1</sub> và b<sub>1</sub>.",
            "update_hidden1": "Cập nhật tham số lớp đầu tiên; mạng sẵn sàng học mẫu kế tiếp.",
        }
        active_state = (
            f"z<sub>1</sub>=({self.fmt(c.z1[0])}, {self.fmt(c.z1[1])}), "
            f"z<sub>2</sub>=({self.fmt(c.z2[0])}, {self.fmt(c.z2[1])}), "
            f"ŷ={self.fmt(c.y_hat)}, L={self.fmt(c.loss, 4)}."
        )
        if phase in {"load", "forward_hidden1", "relu1", "forward_hidden2", "relu2"}:
            active_state = ""
        return phase_text[phase] + ("<br>" + active_state if active_state else "")

    def side_formula_text(self, record: StepRecord, phase: str) -> str:
        c = record.cache
        if phase == "relu1":
            return f"<b>ReLU lớp ẩn 1:</b><br>h<sub>1,1</sub> = max(0,{self.fmt(c.z1[0])}) = {self.fmt(c.h1[0])}<br>h<sub>1,2</sub> = max(0,{self.fmt(c.z1[1])}) = {self.fmt(c.h1[1])}"
        if phase == "relu2":
            return f"<b>ReLU lớp ẩn 2:</b><br>h<sub>2,1</sub> = max(0,{self.fmt(c.z2[0])}) = {self.fmt(c.h2[0])}<br>h<sub>2,2</sub> = max(0,{self.fmt(c.z2[1])}) = {self.fmt(c.h2[1])}"
        if phase == "forward_output":
            return f"<b>Công thức đầu ra:</b><br>ŷ = w<sub>3</sub> · h<sub>2</sub> + b<sub>3</sub><br>{self.fmt(c.t[0])} + {self.fmt(c.t[1])} + {self.fmt(record.params_before['b3'])} = <b>{self.fmt(c.y_hat)}</b>"
        if phase == "loss":
            return f"<b>Công thức mất mát:</b><br>L = (ŷ - y)<sup>2</sup><br>({self.fmt(c.y_hat)} - {self.fmt(c.y)})<sup>2</sup> = <b>{self.fmt(c.loss, 4)}</b>"
        if phase == "backward_output":
            return f"<b>Gradient đầu ra:</b><br>∂L/∂w<sub>3,i</sub> = 2(ŷ-y)h<sub>2,i</sub><br>∂L/∂b<sub>3</sub> = {self.fmt(record.gradients.db3)}"
        if phase == "backward_hidden2":
            return "<b>Gradient lớp ẩn 2:</b><br>∂L/∂W<sub>2</sub> = tích ngoài(h<sub>1</sub>, dz<sub>2</sub>)<br>∂L/∂b<sub>2</sub> = dz<sub>2</sub>"
        if phase == "backward_hidden1":
            return "<b>Gradient lớp ẩn 1:</b><br>∂L/∂w<sub>1</sub> = dz<sub>1</sub>x<br>∂L/∂b<sub>1</sub> = dz<sub>1</sub>"
        if phase.startswith("update"):
            return "<b>Quy tắc cập nhật:</b><br>tham số ← tham số - η·gradient"
        return "<b>Gợi ý:</b><br>Dùng Chạy để xem từng pha.<br>Kéo thanh trượt để chuyển nhanh giữa các mẫu học."

    def explanation_annotations(self, record: StepRecord, phase: str) -> List[dict]:
        return [
            dict(x=4, y=23.4, xref="x", yref="y", text=self.explanation_for_phase(record, phase), showarrow=False, xanchor="left", align="left", width=940, font=dict(size=13, color="#172B4D"), bgcolor="#FFFFFF", bordercolor="#D4DEE9", borderwidth=1, borderpad=8),
            dict(
                x=4,
                y=11.8,
                xref="x",
                yref="y",
                text="<b>Chú giải nhanh:</b><br>Neuron xanh sáng hơn khi giá trị lớn hơn.<br>Viền vàng: ReLU mở; viền xám: ReLU bị chặn.<br>Đường nối/bong bóng lớn hơn: |w| hoặc |b| lớn hơn.",
                showarrow=False,
                xanchor="left",
                align="left",
                width=500,
                font=dict(size=12, color="#26384D"),
                bgcolor="#F4F9FF",
                bordercolor="#D4DEE9",
                borderwidth=1,
                borderpad=8,
            ),
            dict(x=64.0, y=9.8, xref="x", yref="y", text=self.side_formula_text(record, phase), showarrow=False, xanchor="left", align="left", width=450, font=dict(size=12, color="#26384D"), bgcolor="#FFFDF4", bordercolor="#F2C94C", borderwidth=1, borderpad=8),
        ]

    def update_card_annotations(self, record: StepRecord, phase: str) -> List[dict]:
        if phase not in {"update_output", "update_hidden2", "update_hidden1"}:
            return []
        if phase == "update_output":
            x, y = 61.5, 47.5
            title = "Đang cập nhật lớp đầu ra"
            params = ["w<sub>3,1</sub>", "w<sub>3,2</sub>", "b<sub>3</sub>"]
        elif phase == "update_hidden2":
            x, y = 43.0, 47.5
            title = "Đang cập nhật lớp ẩn 2"
            params = ["W<sub>2</sub>", "b<sub>2</sub>"]
        else:
            x, y = 18.0, 47.5
            title = "Đang cập nhật lớp ẩn 1"
            params = ["w<sub>1</sub>", "b<sub>1</sub>"]
        text = f"<b>{title}</b><br><span style='font-size:11px'>Tham số liên quan: {', '.join(params)}</span>"
        return [dict(x=x, y=y, xref="x", yref="y", text=text, showarrow=False, xanchor="center", align="left", width=260, font=dict(size=11.5, color="#172B4D"), bgcolor="#FFF8D6", bordercolor="#F2C94C", borderwidth=1.4, borderpad=8)]

    def layout_for_frame(self, record: StepRecord, phase: str) -> dict:
        shapes = self.static_panel_shapes() + self.relu_chart_shapes()
        gradient_shapes, gradient_annotations = self.gradient_shapes(record, phase)
        shapes.extend(gradient_shapes)
        annotations: List[dict] = []
        annotations.extend(self.header_annotations(record, phase))
        annotations.extend(self.formula_annotations(record, phase))
        annotations.extend(self.parameter_annotations(self.params_for_phase(record, phase)))
        annotations.extend(gradient_annotations)
        annotations.extend(self.explanation_annotations(record, phase))
        annotations.extend(self.update_card_annotations(record, phase))
        return dict(shapes=shapes, annotations=annotations)

    def frame_name(self, record: StepRecord, phase: str) -> str:
        return f"E{record.epoch}_M{record.sample_index}_{phase}"

    def build(self) -> go.Figure:
        first_record = self.records[0]
        first_phase = self.PHASES[0]
        fig = go.Figure(data=self.make_traces(first_record, first_phase))
        frames: List[go.Frame] = []
        for record in self.records:
            for phase in self.PHASES:
                frames.append(go.Frame(name=self.frame_name(record, phase), data=self.make_traces(record, phase), layout=self.layout_for_frame(record, phase)))
        fig.frames = frames
        slider_steps = []
        for record in self.records:
            name = self.frame_name(record, "load")
            slider_steps.append(dict(method="animate", label=f"E{record.epoch}-M{record.sample_index}", args=[[name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]))
        fig.update_layout(
            width=1500,
            height=920,
            margin=dict(l=18, r=18, t=18, b=76),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#EDF4FA",
            showlegend=True,
            legend=dict(x=0.758, y=0.555, xanchor="left", yanchor="top", orientation="v", bgcolor="rgba(255,255,255,0.85)", bordercolor="#D4DEE9", borderwidth=1, font=dict(size=10)),
            xaxis=dict(range=[0, 100], visible=False, fixedrange=True, domain=[0, 1]),
            yaxis=dict(range=[0, 100], visible=False, fixedrange=True, domain=[0, 1]),
            xaxis2=dict(domain=[0.758, 0.965], anchor="y2", title=dict(text="x", font=dict(size=11)), range=[min(self.x_grid), max(self.x_grid)], showgrid=True, gridcolor="#E7EEF5", zeroline=False, fixedrange=True, tickfont=dict(size=10), linecolor="#9AA8B6", mirror=True),
            yaxis2=dict(domain=[0.585, 0.835], anchor="x2", title=dict(text="y / ŷ", font=dict(size=11)), range=[self.y_min, self.y_max], showgrid=True, gridcolor="#E7EEF5", zeroline=False, fixedrange=True, tickfont=dict(size=10), linecolor="#9AA8B6", mirror=True),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.02,
                    y=-0.055,
                    xanchor="left",
                    yanchor="top",
                    buttons=[
                        dict(label="▶ Chạy", method="animate", args=[None, {"frame": {"duration": 720, "redraw": True}, "fromcurrent": True, "transition": {"duration": 160}, "mode": "immediate"}]),
                        dict(label="⏸ Tạm dừng", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]),
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
        fig.write_html(self.output, include_plotlyjs=True, full_html=True, config={"displayModeBar": False, "responsive": True}, auto_play=False)
        if self.auto_open:
            try:
                webbrowser.open("file://" + os.path.abspath(self.output))
            except Exception:
                pass


def make_dataset() -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([1.20, -1.30, -0.40, 0.55, 1.85], dtype=float)
    noise = np.array([0.020, -0.025, 0.010, -0.015, 0.030], dtype=float)
    y = 0.45 * x**2 - 0.10 * x + 0.18 + noise
    return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tạo animation Plotly Glass-Box cho mạng MLP 1-2-2-1.")
    parser.add_argument("--output", default="glass_box_mlp_1221.html", help="Đường dẫn file HTML đầu ra.")
    parser.add_argument("--lr", type=float, default=0.08, help="Tốc độ học. Phải là số dương.")
    parser.add_argument("--epochs", type=int, default=5, help="Số epoch. Phải là số dương.")
    parser.add_argument("--no-open", action="store_true", help="Không tự động mở file HTML đã tạo.")
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs phải là số dương")
    if args.lr <= 0:
        raise ValueError("learning rate phải là số dương")
    return args


def console_safe(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding)


def main() -> None:
    args = parse_args()
    x_data, y_data = make_dataset()
    model = GlassBoxMLP1221(lr=args.lr)

    first_cache = model.forward(float(x_data[0]), float(y_data[0]))
    if not (first_cache.z1[0] > 0 and first_cache.z1[1] <= 0):
        raise RuntimeError("Khởi tạo không thỏa z1_1 > 0 và z1_2 <= 0 với x = 1.20")

    records = model.train_records(x_data, y_data, args.epochs)
    builder = GlassBoxPlotBuilder1221(
        records=records,
        x_data=x_data,
        y_data=y_data,
        epochs=args.epochs,
        lr=args.lr,
        output=args.output,
        auto_open=not args.no_open,
    )
    builder.save()
    print(console_safe(f"Đã lưu animation tương tác vào: {os.path.abspath(args.output)}"))
    print(console_safe(f"Tổng số lần cập nhật: {len(records)} = {args.epochs} epochs x {len(x_data)} mẫu"))


if __name__ == "__main__":
    main()
