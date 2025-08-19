"""
Risk Assessment Visualization System — Streamlit (minimal, interactive)
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st


# --------------------------- Core model ---------------------------

@dataclass
class DomainConfig:
    x_min: float = -10.0
    x_max: float = 0.0
    y_min: float = 0.5
    y_max: float = 2.0
    nx: int = 500   # slightly smaller for faster interaction
    ny: int = 400


@dataclass
class RiskParameters:
    gamma_deficit: float = 2.0
    gamma_normal: float = 5.0
    gamma_surplus: float = 12.0
    cutoff1: float = 0.90
    cutoff2: float = 1.30
    aggressiveness: float = 1.0

    def get_gammas(self) -> np.ndarray:
        return np.array([self.gamma_deficit, self.gamma_normal, self.gamma_surplus], dtype=np.float64)


class RiskMappingTable:
    def __init__(self):
        self.inputs = np.array(
            [1.0, 1.1, 1.2, 1.7, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0,
             7.0, 8.0, 10.0, 12.0, 13.0, 15.0, 16.0, 18.0, 20.0],
            dtype=np.float64
        )
        self.outputs = np.array(
            [1.0000, 0.9207, 0.9206, 0.7779, 0.6541, 0.5264, 0.4378, 0.3234,
             0.2645, 0.2193, 0.1878, 0.1588, 0.1270, 0.1111, 0.0953,
             0.0828, 0.0794, 0.0693, 0.0635],
            dtype=np.float64
        )

    def map(self, gamma: np.ndarray) -> np.ndarray:
        return np.interp(gamma, self.inputs, self.outputs,
                         left=self.outputs[0], right=self.outputs[-1])


class ParabolicMask:
    """Log-y parabolic mask: (log y - l0)^2 <= k_log (x - x_min)"""
    def __init__(self, domain: DomainConfig):
        self.domain = domain
        self.l0 = 0.0
        half_span_log = np.log(domain.y_max) - np.log(1.0)
        self.k_log = (half_span_log ** 2) / (domain.x_max - domain.x_min)

    def compute_mask(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        L = np.log(Y)
        return (L - self.l0) ** 2 <= self.k_log * (X - self.domain.x_min)

    def get_boundaries(self, xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        span_log = np.sqrt(np.maximum(0.0, self.k_log * (xs - self.domain.x_min)))
        y_upper = np.clip(np.exp(self.l0 + span_log), self.domain.y_min, self.domain.y_max)
        y_lower = np.clip(np.exp(self.l0 - span_log), self.domain.y_min, self.domain.y_max)
        return y_upper, y_lower


class RiskFieldComputer:
    def __init__(self, domain: DomainConfig, mapping_table: RiskMappingTable):
        self.domain = domain
        self.mapping = mapping_table
        self.eps = 1e-6

    def sanitize_cutoffs(self, c1: float, c2: float) -> Tuple[float, float]:
        c1 = float(np.clip(c1, self.domain.y_min, self.domain.y_max))
        c2 = float(np.clip(c2, self.domain.y_min, self.domain.y_max))
        if c1 > c2:
            c1, c2 = c2, c1
        if c2 - c1 < self.eps:
            mid = 0.5 * (c1 + c2)
            c1 = max(self.domain.y_min, mid - self.eps / 2)
            c2 = min(self.domain.y_max, mid + self.eps / 2)
        return c1, c2

    def compute_anchors(self, c1: float, c2: float) -> np.ndarray:
        a1 = 0.5 * (self.domain.y_min + c1)
        a2 = 0.5 * (c1 + c2)
        a3 = 0.5 * (c2 + self.domain.y_max)
        return np.array([a1, a2, a3], dtype=np.float64)

    def compute_field(self, params: RiskParameters) -> np.ndarray:
        c1, c2 = self.sanitize_cutoffs(params.cutoff1, params.cutoff2)
        anchors = self.compute_anchors(c1, c2)

        xs = np.linspace(self.domain.x_min, self.domain.x_max, self.domain.nx, dtype=np.float64)
        ys = np.linspace(self.domain.y_min, self.domain.y_max, self.domain.ny, dtype=np.float64)

        gammas = params.get_gammas()
        gamma_y = np.interp(ys, anchors, gammas, left=gammas[0], right=gammas[-1])
        gamma_field = gamma_y[:, None]  # broadcast along x

        divisors = 1.0 + params.aggressiveness * (-xs)  # shape (nx,)
        gamma_adjusted = gamma_field / divisors[None, :]  # (ny, nx)

        return self.mapping.map(gamma_adjusted)


# --------------------------- Streamlit UI ---------------------------

st.set_page_config(page_title="Risk Assessment Visualization", page_icon="📊", layout="wide")

st.markdown("## Risk Assessment Field — Interactive")

# Controls in main area (not sidebar), so they're always visible.
domain = DomainConfig()

colA, colB, colC = st.columns(3)
with colA:
    gamma_deficit = st.slider("Gamma Deficit (γ₁)", 1.0, 25.0, 2.0, 0.01)
    cutoff1 = st.slider("Cutoff 1 (c₁)", domain.y_min, domain.y_max, 0.90, 0.001)
with colB:
    gamma_normal = st.slider("Gamma Normal (γ₂)", 1.0, 25.0, 5.0, 0.01)
    cutoff2 = st.slider("Cutoff 2 (c₂)", domain.y_min, domain.y_max, 1.30, 0.001)
with colC:
    gamma_surplus = st.slider("Gamma Surplus (γ₃)", 1.0, 25.0, 12.0, 0.01)
    aggressiveness = st.slider("Aggressiveness (α)", 0.0, 2.0, 1.0, 0.01)

params = RiskParameters(
    gamma_deficit=gamma_deficit,
    gamma_normal=gamma_normal,
    gamma_surplus=gamma_surplus,
    cutoff1=cutoff1,
    cutoff2=cutoff2,
    aggressiveness=aggressiveness
)

# Compute field
mapping = RiskMappingTable()
computer = RiskFieldComputer(domain, mapping)
masker = ParabolicMask(domain)

xs = np.linspace(domain.x_min, domain.x_max, domain.nx, dtype=np.float64)
ys = np.linspace(domain.y_min, domain.y_max, domain.ny, dtype=np.float64)
X, Y = np.meshgrid(xs, ys)

Z = computer.compute_field(params)
mask = ~masker.compute_mask(X, Y)
Z_masked = np.ma.array(Z, mask=mask)

# Colormap (cached implicitly by reconstruction cost being tiny)
colors = [
    (0.00, "#1a7d3a"),
    (0.25, "#52c41a"),
    (0.45, "#fadb14"),
    (0.65, "#fa8c16"),
    (0.85, "#f5222d"),
    (1.00, "#820014"),
]
cmap = LinearSegmentedColormap.from_list("risk_gradient", colors)
cmap.set_bad(color="#0a0a0a")

# Figure
fig, ax = plt.subplots(figsize=(11, 7))
im = ax.imshow(
    Z_masked,
    origin="lower",
    extent=[domain.x_min, domain.x_max, domain.y_min, domain.y_max],
    aspect="auto",
    cmap=cmap,
    vmin=0.0,
    vmax=1.0,
    interpolation="bilinear",
)

# Boundaries & cutoffs
y_upper, y_lower = masker.get_boundaries(xs)
ax.plot(xs, y_upper, "w--", lw=1.3, alpha=0.8)
ax.plot(xs, y_lower, "w--", lw=1.3, alpha=0.8)

c1_s, c2_s = computer.sanitize_cutoffs(params.cutoff1, params.cutoff2)
ax.axhline(c1_s, color="#e0e0e0", ls=":", lw=1.5, alpha=0.9)
ax.axhline(c2_s, color="#e0e0e0", ls=":", lw=1.5, alpha=0.9)

# Labels
ax.set_xlabel("Time")
ax.set_ylabel("Wealth Level")
ax.set_title("Risk Assessment Field")

cbar = fig.colorbar(im, ax=ax, pad=0.02, aspect=30)
cbar.set_label("Risk Level")

ax.grid(True, alpha=0.2, ls=":", lw=0.5)
ax.set_xlim(domain.x_min, domain.x_max)
ax.set_ylim(domain.y_min, domain.y_max)

st.pyplot(fig, use_container_width=True)
plt.close(fig)  # ensure clean redraws on interaction