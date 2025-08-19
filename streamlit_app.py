import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from dataclasses import dataclass
from typing import Tuple

# -----------------------------------------------------------------------------
# Page setup
st.set_page_config(
    page_title="Risk Assessment Visualization",
    page_icon="📊",
)

# -----------------------------------------------------------------------------
# Core model (no matplotlib)

@dataclass
class DomainConfig:
    x_min: float = -10.0
    x_max: float = 0.0
    y_min: float = 0.5
    y_max: float = 2.0
    nx: int = 420   # tune for responsiveness
    ny: int = 320

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
    """(log y - l0)^2 <= k_log (x - x_min) in (x,y)-space."""
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
        y_up = np.clip(np.exp(self.l0 + span_log), self.domain.y_min, self.domain.y_max)
        y_lo = np.clip(np.exp(self.l0 - span_log), self.domain.y_min, self.domain.y_max)
        return y_up, y_lo

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

    def compute_field(self, params: RiskParameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = np.linspace(self.domain.x_min, self.domain.x_max, self.domain.nx, dtype=np.float64)
        ys = np.linspace(self.domain.y_min, self.domain.y_max, self.domain.ny, dtype=np.float64)

        c1, c2 = self.sanitize_cutoffs(params.cutoff1, params.cutoff2)
        anchors = self.compute_anchors(c1, c2)
        gammas = params.get_gammas()
        gamma_y = np.interp(ys, anchors, gammas, left=gammas[0], right=gammas[-1])

        divisors = 1.0 + params.aggressiveness * (-xs)          # (nx,)
        gamma_adj = (gamma_y[:, None]) / divisors[None, :]      # (ny, nx)

        Z = self.mapping.map(gamma_adj)
        return xs, ys, Z

# -----------------------------------------------------------------------------
# UI

'''
# 📊 Risk Assessment Field

Interactive heatmap of risk level over Time × Wealth.
'''

# spacing
''
''

domain = DomainConfig()

colA, colB, colC = st.columns(3)
with colA:
    gamma_deficit = st.slider('Gamma Deficit (γ₁)', 1.0, 25.0, 2.0, 0.01)
    cutoff1 = st.slider('Cutoff 1 (c₁)', domain.y_min, domain.y_max, 0.90, 0.001)
with colB:
    gamma_normal = st.slider('Gamma Normal (γ₂)', 1.0, 25.0, 5.0, 0.01)
    cutoff2 = st.slider('Cutoff 2 (c₂)', domain.y_min, domain.y_max, 1.30, 0.001)
with colC:
    gamma_surplus = st.slider('Gamma Surplus (γ₃)', 1.0, 25.0, 12.0, 0.01)
    aggressiveness = st.slider('Aggressiveness (α)', 0.0, 2.0, 1.0, 0.01)

params = RiskParameters(
    gamma_deficit=gamma_deficit,
    gamma_normal=gamma_normal,
    gamma_surplus=gamma_surplus,
    cutoff1=cutoff1,
    cutoff2=cutoff2,
    aggressiveness=aggressiveness
)

# Compute field + mask
mapping = RiskMappingTable()
computer = RiskFieldComputer(domain, mapping)
masker = ParabolicMask(domain)

xs, ys, Z = computer.compute_field(params)
X, Y = np.meshgrid(xs, ys)
mask = masker.compute_mask(X, Y)

# Data for charts
heat_df = pd.DataFrame({
    "x": X[mask].ravel(),
    "y": Y[mask].ravel(),
    "z": Z[mask].ravel()
})

y_upper, y_lower = masker.get_boundaries(xs)
bound_df = pd.DataFrame({
    "x": np.concatenate([xs, xs]),
    "y": np.concatenate([y_upper, y_lower]),
    "which": (["upper"] * len(xs)) + (["lower"] * len(xs)),
})

c1_s, c2_s = computer.sanitize_cutoffs(params.cutoff1, params.cutoff2)
cuts_df = pd.DataFrame({
    "y": [c1_s, c2_s],
    "which": ["c1", "c2"]
})

st.header('Risk heatmap', divider='gray')
''

# Altair: build charts with data passed to the constructor (Altair v5)
color_domain = [0.00, 0.25, 0.45, 0.65, 0.85, 1.00]
color_range  = ["#1a7d3a", "#52c41a", "#fadb14", "#fa8c16", "#f5222d", "#820014"]

heat = alt.Chart(heat_df, width='container', height=520).mark_rect().encode(
    x=alt.X('x:Q', title='Time', scale=alt.Scale(domain=(domain.x_min, domain.x_max))),
    y=alt.Y('y:Q', title='Wealth Level', scale=alt.Scale(domain=(domain.y_min, domain.y_max))),
    color=alt.Color('z:Q', title='Risk Level',
                    scale=alt.Scale(domain=color_domain, range=color_range, clamp=True)),
    tooltip=[
        alt.Tooltip('x:Q', format='.2f', title='Time'),
        alt.Tooltip('y:Q', format='.3f', title='Wealth'),
        alt.Tooltip('z:Q', format='.3f', title='Risk'),
    ],
)

line_bound = alt.Chart(bound_df, width='container', height=520).mark_line(strokeDash=[4,4], opacity=0.8).encode(
    x='x:Q',
    y='y:Q',
    detail='which:N'
)

# Horizontal rules at c1 and c2
line_cuts = alt.Chart(cuts_df, width='container', height=520).mark_rule(strokeDash=[2,4], opacity=0.8, color='#e0e0e0').encode(
    y='y:Q'
)

chart = alt.layer(heat, line_bound, line_cuts).configure_view(strokeOpacity=0)

st.altair_chart(chart, use_container_width=True)