# Physics in Machine Learning — Open Artifact

<div align="center">

<img src="figures/emblem.svg" alt="Physics in ML Emblem" width="120" height="120"/>

**A research-grade, reproducible open artifact demonstrating physics-informed neural surrogates, uncertainty calibration, and multi-fidelity learning for scientific simulation.**

[![License: CC-BY 4.0](https://img.shields.io/badge/License-CC--BY%204.0-6DBE4A?style=flat-square&logo=creativecommons)](LICENSE)
[![Artifact Status](https://img.shields.io/badge/Artifact-Open-0F62FE?style=flat-square)](https://github.com/open-artifact/physics-in-ml)
[![PDF](https://img.shields.io/badge/PDF-10%20pages-D4AF37?style=flat-square&logo=adobe-acrobat-reader)](artifact.pdf)
[![Docker](https://img.shields.io/badge/Docker-Ready-00A6A6?style=flat-square&logo=docker)](env/Dockerfile)
[![Python](https://img.shields.io/badge/Python-3.10%2B-2B2B2B?style=flat-square&logo=python)](env/requirements.txt)
[![CI](https://img.shields.io/badge/CI-passing-6DBE4A?style=flat-square&logo=github-actions)](https://github.com/open-artifact/physics-in-ml/actions)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.placeholder-D4AF37?style=flat-square)](https://zenodo.org)

> *"Physics tells us what is possible; data tells us what is real; together they tell us what is safe."*

</div>

---

## Quick Links

| Resource | Description |
|----------|-------------|
| 📄 [`artifact.pdf`](artifact.pdf) | 10-page publication-quality PDF artifact |
| 📓 [`notebooks/`](notebooks/) | Numbered Jupyter notebooks 00–06 |
| 📦 [`assets.zip`](assets.zip) | Paper texture, fonts, emblem, all figures |
| 🐍 [`src/`](src/) | Model implementations: PINN, FNO, GP correction |
| 📊 [`data/`](data/) | Sample HDF5, manifests, metadata JSON |
| 🖼 [`figures/`](figures/) | SVG and PNG figure assets |
| ⚙️ [`env/`](env/) | `requirements.txt`, Dockerfile, seed log |
| 🔬 [`scripts/`](scripts/) | `run_experiment.sh`, `eval_metrics.py`, `render_readme.py` |
| 📜 [`LICENSE`](LICENSE) | CC-BY 4.0 |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Quick Start and Repo Structure](#2-quick-start-and-repo-structure)
3. [Data and Benchmarks](#3-data-and-benchmarks)
4. [Methods](#4-methods)
5. [Experiments and Evaluation](#5-experiments-and-evaluation)
6. [Figures and Visuals](#6-figures-and-visuals)
7. [Advanced Insights and Research-Grade Notes](#7-advanced-insights-and-research-grade-notes)
8. [Reproducibility Manifest](#8-reproducibility-manifest)
9. [CI / CD and Automation](#9-ci--cd-and-automation)
10. [Contribution Guidelines](#10-contribution-guidelines)
11. [Citation](#11-citation)
12. [License](#12-license)

---

## 1. Project Overview

### Physics in Machine Learning — Open Artifact

> **Open Artifact** · Research-grade · Ivy-League × IIT style · CC-BY 4.0

This README is the **canonical entry point** for reproducing the 10-page artifact, datasets, code, figures, and Claude/Kimmi prompts. It is written to be **publication quality**, **reproducible**, and **immediately usable** by academic and industry researchers.

### Purpose

- Provide a compact, reproducible research artifact demonstrating how **physics priors**, **neural surrogates**, and **uncertainty calibration** combine to accelerate simulation and enable safe deployment.
- Serve as a single README that Claude/Kimmi can use to render the full 10-page PDF, generate figures, and produce assets for a polished deliverable.
- Bridge the gap between simulation-heavy physics and data-driven ML — capturing the *speed* of surrogates and the *reliability* of physics constraints.

### What This Delivers

| Deliverable | Description |
|-------------|-------------|
| **`README.md`** | Complete artifact manifest (this file) |
| **`artifact.pdf`** | 10-page publication PDF, 300 DPI, embedded fonts |
| **`assets.zip`** | Paper texture, handwritten font, emblem, icons, all figures |
| **`figures/`** | SVG vector plots + high-res PNG heatmaps |
| **`data/sample.h5`** | Reproducible sample dataset (≤200 MB) |
| **`src/`** | PINN, Fourier Neural Operator, GP correction implementations |
| **`notebooks/`** | End-to-end numbered notebooks 00–06 |

### Motivating Framing

Standard data-driven surrogates ignore physical conservation laws, producing fast but untrustworthy predictions outside the training distribution. Physics-informed neural networks (PINNs) and neural operators close this gap by encoding PDEs as soft or hard constraints. The residual gap — what the physics model still misses — is corrected by a Gaussian Process that also provides calibrated uncertainty estimates. This three-component pipeline (*encode → surrogate → correct*) is the central contribution of this artifact.

```
Data Sources ──► Physics-aware Models ──► Decision Layer
     │                   │                      │
   Mesh fields      PINN + FNO             UQ + GP
   Time series      Residual               Calibrated
   Sim/exp pairs    correction             deployment
```

> **Handwritten marginalia:** *"enforce conservation"* · *"calibrate on held-out physics"* · *"monitor drift"*

---

## 2. Quick Start and Repo Structure

### One-line Install

```bash
git clone <repo-url> && cd physics-in-ml && ./run_sample.sh
```

### Minimal Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r env/requirements.txt
```

### Run the Canonical Demo

```bash
./scripts/run_experiment.sh --task demo --seed 42
# produces artifact.pdf and assets.zip in output/
```

### Repository Layout

```
physics-in-ml/
│
├── README.md                    ← This file; canonical artifact manifest
│
├── artifact.pdf                 ← 10-page publication-quality PDF
├── assets.zip                   ← All visual and font assets
│
├── data/
│   ├── sample.h5                ← Sample dataset (HDF5, ≤200 MB)
│   ├── synthetic_ensemble_manifest.json
│   ├── sensor_streams_manifest.json
│   └── hybrid_pairs_manifest.json
│
├── src/
│   ├── pinn.py                  ← Physics-informed neural network
│   ├── fno.py                   ← Fourier neural operator
│   ├── gp_correction.py         ← GP residual correction + UQ
│   ├── neural_operator.py       ← Graph operator variant
│   └── calibration.py           ← Temperature scaling, ECE/NLL
│
├── notebooks/
│   ├── 00_data_exploration.ipynb
│   ├── 01_pinn_training.ipynb
│   ├── 02_fno_training.ipynb
│   ├── 03_gp_correction.ipynb
│   ├── 04_calibration.ipynb
│   ├── 05_benchmark_suite.ipynb
│   └── 06_interpretability_uq.ipynb
│
├── figures/
│   ├── figure1-pipeline.svg
│   ├── figure2-architecture.svg
│   ├── figure3-results-panel.svg
│   ├── figure3-residuals.png
│   ├── figure5-benchmark-panel.svg
│   ├── figure5-heatmaps.png
│   ├── figure6-transfer-ablation.svg
│   ├── figure7-results-panel.svg
│   ├── figure7-residuals.png
│   ├── figure8-interpretability-uq.svg
│   └── figure8-attribution.png
│
├── assets/
│   ├── paper_texture.png        ← Off-white handmade texture
│   ├── handwritten_font.ttf     ← Natural pen script font
│   ├── emblem.svg               ← Project emblem
│   └── icons.svg                ← Icon sprite sheet
│
├── env/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── seed_log.txt             ← All random seeds used
│
├── scripts/
│   ├── run_experiment.sh        ← Canonical experiment runner
│   ├── eval_metrics.py          ← Metric evaluation script
│   └── render_readme.py         ← README → PDF renderer
│
└── LICENSE                      ← CC-BY 4.0
```

---

## 3. Data and Benchmarks

### Datasets Included

| Dataset | Description | Size | Format |
|---------|-------------|------|--------|
| **Synthetic Ensemble** | Multi-fidelity meshes with BC vectors; 10k samples (coarse/mid/high) | ~150 MB | HDF5 |
| **Sensor Streams** | 500 hours of multi-channel time series with labeled anomaly events | ~30 MB | HDF5 |
| **Hybrid Pairs** | 2k paired sim/exp samples for domain transfer experiments | ~20 MB | HDF5 |

### Data Manifest Example

```json
{
  "dataset": "synthetic_ensemble",
  "n_samples": 10000,
  "fidelities": ["low", "mid", "high"],
  "fields": ["E_x", "E_y", "H_z"],
  "format": "HDF5",
  "checksum": "sha256:a3f8b2c1d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0"
}
```

### Synthetic Data Generator Pseudocode

```python
def generate_synthetic_ensemble(
    n_samples: int = 10000,
    fidelities: list = ["low", "mid", "high"],
    noise_level: float = 0.02,
    bias: float = 0.0,
    seed: int = 42
) -> dict:
    """
    Generates multi-fidelity PDE solution ensemble with experimental drift.

    Args:
        n_samples:   Number of samples per fidelity level
        fidelities:  Resolution levels to simulate
        noise_level: Gaussian noise sigma (varies to simulate drift)
        bias:        Systematic shift to simulate experimental offset
        seed:        Random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    meshes = {}
    for fidelity in fidelities:
        resolution = {"low": 32, "mid": 64, "high": 128}[fidelity]
        bc_vectors = rng.standard_normal((n_samples, 4))  # boundary conditions
        solutions = solve_pde_batch(bc_vectors, resolution=resolution)
        noise = rng.normal(0, noise_level, solutions.shape)
        meshes[fidelity] = solutions + noise + bias

    return {
        "fields": meshes,
        "bc_vectors": bc_vectors,
        "metadata": {"seed": seed, "noise_level": noise_level, "bias": bias}
    }
```

### Benchmark Tasks Summary

| Task | Data Type | Primary Metric | Secondary Metrics |
|------|-----------|---------------|-------------------|
| Multi-fidelity PDE mapping | Field on mesh | RMSE | Physics residual L2 |
| Sensor array anomaly detection | Multichannel time series | AUC | Detection latency |
| Domain transfer sim→exp | Paired sim/exp fields | RMSE on exp | ECE |
| Inverse constrained design | Parameter → performance | #evals to target | Constraint satisfaction rate |

### Evaluation Protocol

- **Seeds:** Report mean ± std over 5 seeds.
- **Metrics:** RMSE, physics residual L2, ECE, NLL, runtime speedup, inference latency.
- **Splits:** Train 70% BCs, val 15%, test 15% — **hold out unseen BCs** (not random shuffle).
- **Time series:** Contiguous 70/15/15 time blocks to prevent data leakage.

---

## 4. Methods

### Pipeline Architecture

The end-to-end pipeline consists of three learned components:

```
┌─────────────────────────────────────────────────────────────────┐
│                   PHYSICS-IN-ML PIPELINE                       │
│                                                                 │
│   INPUT                   SURROGATE              OUTPUT        │
│                                                                 │
│  ┌─────────┐   ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐  │
│  │  Mesh   │──►│  PINN  │►│  FNO   │►│  GP    │►│ Calibrated│  │
│  │  BCs    │   │encoder │ │decode  │ │correct │ │prediction │  │
│  │  ICs    │   └────────┘ └────────┘ └────────┘ └──────────┘  │
│  └─────────┘                                                    │
│                                                                 │
│  ← Physics priors ──────────────────── Uncertainty bounds →    │
└─────────────────────────────────────────────────────────────────┘
```

### PINN Loss Function

The physics-informed loss combines data fidelity, PDE residual, and boundary condition terms:

$$L = \lambda_{\text{data}} \sum_i \|u_\theta(x_i) - y_i\|^2 + \lambda_{\text{phys}} \sum_j \|\mathcal{N}[u_\theta](x_j)\|^2 + \lambda_{\text{bc}} \sum_k \|\mathcal{B}[u_\theta](x_k) - g_k\|^2$$

where:
- $u_\theta$ is the neural network surrogate
- $\mathcal{N}[\cdot]$ is the differential operator (PDE residual)
- $\mathcal{B}[\cdot]$ is the boundary operator
- $\lambda_{\text{data}}, \lambda_{\text{phys}}, \lambda_{\text{bc}}$ are adaptive loss weights

### Fourier Neural Operator (FNO)

The FNO learns a mapping between function spaces by parameterizing the integral kernel in Fourier space:

$$(\mathcal{K}_\phi v_t)(x) = \mathcal{F}^{-1}\left(R_\phi \cdot \mathcal{F}(v_t)\right)(x)$$

Combined with a local linear transform $W$, each FNO layer reads:

$$v_{t+1}(x) = \sigma\left(W v_t(x) + (\mathcal{K}_\phi v_t)(x)\right)$$

### GP Residual Correction

After PINN/FNO inference, a Gaussian Process corrects systematic residuals:

$$\epsilon(x) = f_{\text{surrogate}}(x) - f_{\text{true}}(x) \sim \mathcal{GP}(0, k(x, x'))$$

Posterior predictive distribution provides calibrated uncertainty:

$$p(\epsilon^* | x^*, X, \epsilon) = \mathcal{N}(\mu^*, \sigma^{*2})$$

### Temperature Scaling (Post-hoc Calibration)

$$\hat{q}_i = \text{softmax}(z_i / T)$$

where $T > 1$ softens overconfident predictions. Optimal $T$ is found by minimizing NLL on the validation set.

Expected Calibration Error (ECE):

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

### Code Skeleton — PINN Training Loop

```python
import torch
import torch.nn as nn
from src.pinn import PINN
from src.gp_correction import GPCorrection

# ── Model Instantiation ──────────────────────────────────────────
model = PINN(
    layers=[2, 64, 64, 64, 1],
    activation=nn.Tanh(),
    lambda_phys=0.1,        # start low, anneal to 1.0
    lambda_bc=1.0,
)
gp = GPCorrection(kernel="matern52", noise=1e-3)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

# ── Training Loop ─────────────────────────────────────────────────
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    # Data fidelity term
    u_pred = model(x_data)
    loss_data = nn.MSELoss()(u_pred, y_data)

    # Physics residual term (automatic differentiation)
    u_phys = model(x_phys)
    residual = pde_operator(u_phys, x_phys)          # ∂u/∂t + N[u] = 0
    loss_phys = (residual ** 2).mean()

    # Boundary conditions
    u_bc = model(x_bc)
    loss_bc = nn.MSELoss()(u_bc, g_bc)

    # Adaptive weighting (anneal λ_phys from 0.1 → 1.0)
    lam = min(1.0, 0.1 + epoch / epochs)
    loss = loss_data + lam * loss_phys + loss_bc

    loss.backward()
    optimizer.step()
    scheduler.step()

# ── GP Residual Correction ────────────────────────────────────────
residuals = (model(x_val) - y_val).detach().numpy()
gp.fit(x_val.numpy(), residuals)
correction, std = gp.predict(x_test.numpy())
```

---

## 5. Experiments and Evaluation

### Canonical Experiments

| # | Experiment | Description |
|---|------------|-------------|
| 1 | **Baseline suite** | Linear surrogate, random forest, small CNN |
| 2 | **Physics-aware** | PINN with adaptive physics weighting |
| 3 | **Operator family** | Fourier neural operator and graph operator variants |
| 4 | **Hybrid** | Neural surrogate + residual correction network |
| 5 | **UQ** | Deep ensembles, MC dropout, GP residual correction |

### Hyperparameter Defaults

| Parameter | Default | Range |
|-----------|---------|-------|
| Learning rate | `1e-4` | `[1e-5, 1e-3]` |
| Batch size | `32` | `[16, 128]` |
| Epochs | `200–1000` | — |
| $\lambda_{\text{phys}}$ start | `0.1` | `[0.01, 0.5]` |
| $\lambda_{\text{phys}}$ final | `1.0` | `[0.5, 5.0]` |
| FNO modes | `16` | `[8, 32]` |
| GP kernel | `matern52` | `[rbf, matern32]` |

### Ablations to Run

- Remove physics loss ($\lambda_{\text{phys}} = 0$).
- Remove correction network (surrogate only).
- Replace spectral layers with standard conv layers.
- Vary training set size: 10%, 25%, 50%, 100%.
- Remove GP correction (surrogate + calibration only).

### Statistical Rigor

- Use **paired Wilcoxon signed-rank tests** across seeds for ablation comparisons.
- Report effect sizes and 95% confidence intervals for power-law exponents in RMSE vs $n$ fits.
- Fit the scaling model:

$$\text{RMSE}(n) \propto n^{-\alpha}$$

and report $\hat{\alpha}$ with 95% CI. Physics-informed models are expected to yield $\alpha > 0.5$ (faster convergence than standard surrogates).

### Primary Tasks — Metrics

| Task | Primary | Secondary |
|------|---------|-----------|
| Multi-fidelity PDE mapping | RMSE, physics residual L2 | Runtime speedup |
| Sensor anomaly detection | AUC | Detection latency |
| Domain transfer (sim→exp) | RMSE (exp), ECE | NLL |
| Inverse constrained design | #evals to target | Constraint satisfaction rate |

### Compute Logging

Log wall-clock time, GPU type, peak memory, and throughput (samples/sec). Save:
- `env/pip_freeze.txt`
- `env/docker_manifest.json`

---

## 6. Figures and Visuals

### Visual Style Guide

| Element | Specification |
|---------|---------------|
| **Paper texture** | Off-white handmade, deckled edges, subtle fiber specks |
| **Heading font** | Garamond (body + headings) |
| **Caption font** | Inter / FF Meta |
| **Code font** | Menlo |
| **Handwritten notes** | Natural pen script, indigo `#2E2B7A`, tilt −6° to −10° |

### Color Palette

| Swatch | Name | Hex | Usage |
|--------|------|-----|-------|
| 🟩 | Apple Green | `#6DBE4A` | Fast baselines, positive delta |
| 🟦 | IBM Blue | `#0F62FE` | Accurate / physics-aware models |
| 🟨 | Bloomberg Gold | `#D4AF37` | Hybrid models |
| 🟦 | Anthropic Teal | `#00A6A6` | Safety / uncertainty bounds |
| ⬛ | OpenAI Charcoal | `#2B2B2B` | Text, axes, annotations |

### Figure Inventory

| Figure | Filename | Type | Content |
|--------|----------|------|---------|
| Fig 1 | `figure1-pipeline.svg` | SVG vector | Data Sources → Physics Models → Decision Layer |
| Fig 2 | `figure2-architecture.svg` | SVG vector | PINN + FNO + GP architecture diagram |
| Fig 3A | `figure3-results-panel.svg` | SVG vector | Runtime vs RMSE scatter (log-x), Pareto frontier |
| Fig 3B | `figure3-results-panel.svg` | SVG vector | Calibration reliability diagram, ECE annotations |
| Fig 3C | `figure3-residuals.png` | PNG heatmap | Stacked residual heatmaps (high-fi / surrogate / error) |
| Fig 5A | `figure5-benchmark-panel.svg` | SVG vector | Runtime vs accuracy, calibration, speedup histogram |
| Fig 5B | `figure5-heatmaps.png` | PNG heatmap | Benchmark heatmaps with grain overlay |
| Fig 6 | `figure6-transfer-ablation.svg` | SVG vector | Log-log RMSE vs #fine-tune samples + ablation bar grid |
| Fig 7 | `figure7-results-panel.svg` | SVG vector | Extended results narrative panels |
| Fig 7R | `figure7-residuals.png` | PNG heatmap | Residual analysis |
| Fig 8 | `figure8-interpretability-uq.svg` | SVG vector | Latent mode alignment + explained variance bars |
| Fig 8A | `figure8-attribution.png` | PNG heatmap | Integrated gradients attribution map |

### Figure 1 — Pipeline Schematic

```
figure1-pipeline.svg
─────────────────────────────────────────────────────────────
[Data Sources]        [Physics-aware Models]    [Decision Layer]
  Mesh fields    →       PINN encoder      →    Calibrated
  Time series    →       FNO decoder       →    predictions
  Sim/exp pairs  →    GP residual correct  →    + UQ bounds

↑ "enforce conservation"    ↑ "calibrate on held-out physics"    ↑ "monitor drift"
  (handwritten marginalia, indigo #2E2B7A, −8° tilt)
```

### Figure 3 — Results Panel

```
figure3-results-panel.svg  |  figure3-residuals.png
─────────────────────────────────────────────────────────────
Panel A: Runtime vs RMSE           Panel C: Residual Heatmaps
  log(runtime) on x-axis             Row 1: High-fidelity truth
  RMSE on y-axis                     Row 2: Surrogate prediction
  Pareto frontier highlighted        Row 3: |Error| (abs diff)
  5 model families, 5 colors         Perceptually uniform colormap
                                     Paper grain overlay applied
Panel B: Calibration Diagram
  Reliability diagram
  Confidence on x-axis
  Accuracy on y-axis
  ECE annotated per model
```

### Figure 6 — Transfer Efficiency & Ablation

```
figure6-transfer-ablation.svg
─────────────────────────────────────────────────────────────
Left panel: Log-log RMSE vs #fine-tune samples
  Models: Baseline (green), Physics-aware (blue), Hybrid (gold)
  Power-law fits with exponent annotations: α = 0.82, 1.24, 1.01
  "Fast convergence with physics prior" (handwritten callout)

Right panel: Ablation bar grid (4 × 2)
  Rows: no physics loss | no GP | no FNO | full model
  Cols: RMSE | ECE
  Δ from full model annotated on each bar
```

### Live Visuals Styling Notes

- **Vector SVG** for scatter and calibration plots; embed fonts.
- **Heatmaps:** high-res PNG at 300 DPI with paper grain overlay (`assets/paper_texture.png`).
- **Color mapping:** Apple Green = fast, IBM Blue = accurate, Bloomberg Gold = hybrid, Anthropic Teal = safety, OpenAI Charcoal = text.
- **Handwritten marginalia:** indigo `#2E2B7A`, stroke width 0.5–1.2 px, tilt −6° to −10° for authenticity.

---

## 7. Advanced Insights and Research-Grade Notes

### Design Principles for Research-Grade Artifacts

| Principle | Guidance |
|-----------|----------|
| **Minimal but complete** | Include the smallest dataset and code that reproduces the main claims. |
| **Transparent defaults** | Publish all hyperparameters, seeds, and environment specs. |
| **Statistical rigor** | Always report variance across seeds; use nonparametric paired tests for ablations. |
| **Interpretability first** | Include latent probes and mode matching to connect learned representations to physics. |
| **Calibration as first class** | Report ECE, NLL, and sharpness; provide per-region calibration diagnostics. |

### Pitfalls and Mitigations

| Pitfall | Mitigation |
|---------|------------|
| **Overfitting to synthetic regimes** | Mitigate with domain randomization and hold-out BCs. |
| **Misleading speedups** | Always report end-to-end wall-clock time including I/O and pre/post processing. |
| **Hidden data leakage** | Enforce BC splits and contiguous time blocks for time series. |
| **Overconfident UQ** | Validate calibration on out-of-distribution physics regimes. |
| **Physics loss instability** | Anneal $\lambda_{\text{phys}}$ from 0.1 → 1.0; use gradient clipping. |

### Immediate Experiments to Prioritize

1. Measure exponent $\alpha$ in $\text{RMSE}(n) \propto n^{-\alpha}$ for models with and without physics loss.
2. Compare ECE before and after GP residual correction across spatial regions.
3. Run constrained Bayesian optimization with surrogate ensembles and report median #evaluations to target.
4. Probe latent space modes against known physical eigenmodes (e.g., curl-free, divergence-free components).
5. Measure calibration sharpness: does physics loss improve or hurt sharpness relative to ECE?

### Interpretability Case Studies

**Latent Mode Alignment:**

```python
# Extract latent activations at bottleneck layer
z = model.encoder(x_test)  # shape: [N, d_latent]

# Align latent modes with known physics modes
phi_curl, phi_div = compute_physics_modes(x_test)  # shape: [N, 2]
alignment = np.corrcoef(z.T, phi_curl)[:d_latent, d_latent:]

# Report: which latent dims align with physics modes?
print("Max alignment (curl-free):", alignment.max())
```

**Integrated Gradients Attribution:**

```python
from captum.attr import IntegratedGradients

ig = IntegratedGradients(model)
attributions = ig.attribute(
    x_test,
    baselines=torch.zeros_like(x_test),
    target=0,
    n_steps=50,
)
# Visualize: attribution heatmap → figure8-attribution.png
```

---
- [ ] Verify dataset checksums match `data/*_manifest.json`.
- [ ] Run `scripts/run_experiment.sh --task demo --seed 42` and confirm `output/artifact.pdf` is generated.
- [ ] Run `scripts/eval_metrics.py --checkpoint output/demo_ckpt.pt` and compare to reported Table 1 values (RMSE < 0.015, ECE < 0.03).
- [ ] Run all 7 notebooks `00–06` end-to-end; no cell should error.
- [ ] Confirm figures match the reference PNGs in `figures/reference/`.
- [ ] Confirm `env/pip_freeze.txt` matches the installed environment within patch version.

### Compute Requirements

| Task | GPU Memory | Wall Time | CPU Fallback |
|------|-----------|-----------|--------------|
| Demo (seed 42) | 4 GB | ~15 min | ~2 hours |
| Full benchmark | 16 GB | ~4 hours | ~24 hours |
| Figure generation | 2 GB | ~20 min | ~1 hour |

---

## 9. CI / CD and Automation

### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: Physics-in-ML CI

on: [push, pull_request]

jobs:
  test:

### Render README to PDF

```bash
python scripts/render_readme.py \
  --input README.md \
  --output output/artifact.pdf \
  --template assets/paper_texture.png \
  --font assets/handwritten_font.ttf \
  --dpi 300
```

---

## 10. Contribution Guidelines

### Before Opening a PR

1. Run the full test suite: `pytest tests/ -v`
2. Run pre-commit: `pre-commit run --all-files`
3. Confirm reproducibility checklist (§8) passes.
4. Add a test for any new model component under `tests/`.

### Contribution Scope

| Category | Welcomed |
|----------|---------|
| New benchmark tasks | ✅ with dataset manifest and eval script |
| New model architectures | ✅ with ablation results |
| Additional figures | ✅ following visual style guide (§6) |
| Bug fixes | ✅ always |
| New dependencies | ⚠️ discuss in an issue first |
| Changing default hyperparameters | ⚠️ requires re-running full benchmark |

### Issue Labels

| Label | Meaning |
|-------|---------|
| `reproducibility` | Numerical discrepancy or environment issue |
| `new-dataset` | Proposal to add a dataset |
| `figure-quality` | Visual output improvement |
| `physics-prior` | New PDE constraint or operator |
| `good-first-issue` | Suitable for new contributors |

### Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct. By participating, you agree to uphold a welcoming, inclusive, and respectful community.

---

## 11. Citation

If you use this artifact in your research, please cite:

```bibtex
@misc{physics_in_ml_2025,
  title        = {Physics in Machine Learning — Open Artifact},
  year
**Open Artifact** · Physics in Machine Learning · CC-BY 4.0

*Publication quality · Reproducible · Immediately usable*

Made with rigour, physics, and caffeine.

</div>
