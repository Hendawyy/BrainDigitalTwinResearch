# 🧠 Neuro-DT: Brain Digital Twin for Alzheimer's Disease Progression

A multimodal deep learning framework that classifies Alzheimer's disease stage from
T1-weighted MRI + clinical features, explains its predictions with Grad-CAM, and
simulates individual patient disease trajectories — with and without genetic risk
factors and pharmacological intervention — using a Markov chain Digital Twin engine.

Developed as a master's thesis at the **Arab Academy for Science, Technology and
Maritime Transport**.

**Author:** Seif Hendawy
**Supervisors:** Prof. Fahima Maghraby · Assoc. Prof. Ahmed Salem

**Live dashboard:** `https://neuro-dt-dashboard.azurewebsites.net/`

---

## 💡 What this is

Neuro-DT combines five components into one pipeline:

1. 🧬 **3D CNN + Transformer multimodal classifier** — MRI imaging fused with tabular
   clinical features, classifying CN / MCI / Dementia
2. 🔥 **Grad-CAM explainability** — visualizes which brain regions drove each prediction
3. ⛓️ **Markov chain prognostic engine** — empirical disease-state transition
   probabilities estimated from longitudinal ADNI visit sequences
4. 🎲 **Monte Carlo Digital Twin simulation** — projects an individual patient's 5-year
   disease trajectory, with APOE4-genotype-stratified and drug-intervention scenarios
5. 📊 **Streamlit clinical dashboard** — an interactive tool for running all of the above
   against a real patient's scan and clinical data, deployed on Azure App Service

Two model generations exist in this repo: a **CPU-only baseline** (proof of concept,
Fold 4 only, no ablation study) and a **GPU-trained model** (full 5-fold
cross-validation, complete ablation study, complete Digital Twin pipeline). The
dashboard serves the GPU model. See the Results section below for why this
distinction matters and isn't just "GPU is faster."

---

## 📁 Repository structure

```
BrainDigitalTwinResearch/
├── Dashboard/                              # Production Streamlit clinical dashboard
│   ├── app.py                              # Main application
│   ├── model.py                            # MultimodalTransformer architecture
│   ├── Dockerfile                          # Container build definition
│   ├── requirements.txt                    # Python dependencies
│   └── strip_checkpoint.py                 # Utility: shrinks a training checkpoint
│                                            #   for inference-only deployment (289MB→97MB)
├── Framework/                              # Architecture diagrams
│   ├── Final BDT Diagram.drawio.png
│   └── Final BDT Diagram NO SUBTEXT.drawio.png
├── NeuroDT_Notebooks/                      # Training & analysis notebooks
│   ├── NeuroDT_GPU.ipynb                   # Primary notebook — full pipeline
│   └── NeuroDT_CPU.ipynb                   # CPU-compatible baseline notebook
├── LICENSE
└── README.md
```

> **Not included in this repo:** the ADNI dataset itself (license-restricted — see
> the Dataset section below), the 33GB tensor cache, and trained model checkpoints
> (hundreds of MB each). These live in Azure Blob Storage; see the Reproducing
> this project section below.

---

## 🏗️ Architecture

![BDT Framework Diagram](Framework/Final%20BDT%20Diagram.drawio.png)

**Model:** 3D DenseNet121 (imaging backbone) → global average pooling → concatenate
with 4 tabular features (`AGE`, `PTEDUCAT`, `MMSE`, `APOE4`) → linear projection +
LayerNorm + GELU → Transformer encoder (2 layers, 8 heads, `norm_first=True`) →
3-class classifier head. **~24M parameters.**

---

## 🗂️ Dataset

- **Source:** [ADNI](http://adni.loni.usc.edu/) (Alzheimer's Disease Neuroimaging
  Initiative) — access requires a separate application and approval; the raw data
  is not redistributable and is therefore not included in this repo
- **1,549 T1-weighted MPRAGE MRI scans** across ADNI-1/GO/2/3, ~500 unique patients
- Class distribution: **469 CN** (30.3%) · **603 MCI** (38.9%) · **477 Dementia** (30.8%)
- 4 tabular features per scan: `AGE`, `PTEDUCAT` (years of education), `MMSE`
  (Mini-Mental State Exam, 0–30), `APOE4` (allele count, 0/1/2)

---

## 📊 Results

### 🤔 Why there are two models

The original training pipeline ran on a CPU-only Azure compute instance
(`Standard_E4ds_v4`). Under `CosineAnnealingLR` with no warmup and `batch_size=4`,
4 of 5 cross-validation folds spiked in validation loss within the first 1–3 epochs
and were killed by early stopping before any real learning happened — only **Fold 4**
trained to completion. The full deep-learning ablation study was projected at
69–129 hours on that hardware and never ran at all.

The pipeline was migrated to a GPU (RTX 5070 Ti), the training recipe fixed
(`OneCycleLR`, larger batch size, mixed precision), and re-run: all 5 folds now
train to completion, giving a genuine cross-validated result, plus the complete
ablation study and full Digital Twin pipeline that CPU never reached.

### 📈 Headline comparison

| | CPU model | GPU model |
|---|---|---|
| Folds completed | 1 of 5 (Fold 4 only) | 5 of 5 |
| Reported AUC | 0.9120 (single fold — not a valid generalization estimate) | **0.8706 ± 0.0461** (5-fold CV mean) |
| Fold 4 head-to-head (identical patient split) | 0.9120 | **0.9511** |
| Ablation study | Not run (infeasible on CPU) | 11 models (7 DL variants + 4 classical baselines) |
| Full Digital Twin pipeline | Never reached (training never finished) | Ran end to end |

**Read the two AUC numbers carefully:** 0.8706 (GPU mean) looks lower than 0.9120
(CPU) only because CPU's single data point happened to land on GPU's *easiest*
fold. The other four GPU folds (0.8821, 0.8591, 0.8125, 0.8481) were never sampled
by CPU at all. On the one truly apples-to-apples comparison — Fold 4, identical
patients, both machines — **GPU wins by +0.039 AUC**, and it's also the only one of
the two with an actual variance estimate behind it.

### 🎯 GPU model — full results

**5-fold cross-validation:**

| Fold | Epochs completed | Val loss | Macro AUC |
|---|---|---|---|
| 1 | 8 (early-stopped) | 0.603 | 0.882 |
| 2 | 8 (early-stopped) | 0.643 | 0.859 |
| 3 | 2 (early-stopped) | 0.757 | 0.813 |
| **4 (deployed)** | **20 (full run)** | **0.461** | **0.951** |
| 5 | 5 (early-stopped) | 0.670 | 0.848 |
| **Mean** | | | **0.871 ± 0.046** |

**Per-class AUC (Fold 4, one-vs-rest):** CN 0.982 · Dementia 0.958 · MCI 0.913
**Fold 4 accuracy:** 85% · **Macro F1:** 0.85

**Ablation study** (single-fold, Fold 4 — read the top results as within noise of
each other, not a meaningful ranking):

| Model | Macro AUC | Accuracy | Macro F1 |
|---|---|---|---|
| A5 · Transformer, 1 layer | 0.956 | 85.2% | 0.855 |
| A4 · CNN + linear fusion | 0.949 | 84.2% | 0.845 |
| A0 · Neuro-DT full model | 0.949 | 85.2% | 0.855 |
| A2 · CNN only (no tabular) | 0.947 | 84.5% | 0.850 |
| B3 · Random Forest | 0.943 | 82.6% | 0.830 |
| B4 · Gradient Boosting | 0.941 | 82.3% | 0.825 |
| A3 · CNN + MLP fusion | 0.939 | 83.9% | 0.844 |
| A6 · no class weights | 0.929 | 78.4% | 0.791 |
| A1 · tabular only (no MRI) | 0.872 | 68.4% | 0.675 |
| B2 · SVM (RBF) | 0.871 | 73.5% | 0.735 |
| B1 · Logistic Regression | 0.865 | 71.0% | 0.707 |

**What this shows:** imaging is doing the real work (dropping MRI costs ~0.077 AUC
and 17 points of accuracy, the largest single drop in the study); the extra fusion
machinery beyond a plain CNN barely earns its keep (CNN-only ≈ full model); and a
well-tuned Random Forest on the same 4 tabular features gets closer to the deep
model than expected — a real result to report, not a bug.

### 📚 Comparison with published literature

| Paper | Method | AUC / Accuracy |
|---|---|---|
| Basaia et al. 2019 | 3D CNN | ~0.85 AUC |
| Wen et al. 2020 | CNN benchmark study | ~0.83 AUC |
| Venugopalan 2021 | Multimodal ML | ~0.87 AUC |
| Kushol 2022 (ADDformer) | Transformer | ~0.89 AUC |
| Hu 2023 (Conv-Swinformer) | CNN + Swin Transformer | 92.9% acc (2-class) |
| DT-GPT 2025 | GPT on EHR | 1.8% MAE reduction |
| **Neuro-DT 2026 (ours)** | 3D DenseNet + Transformer + Markov | **0.951 AUC · 85% acc** |

Neuro-DT is the only one of these combining 3D MRI classification, individual
patient simulation, time-varying drug intervention modelling, Grad-CAM
explainability, and an LLM advisory layer in one framework.

### 🔬 External-validity findings from the Digital Twin pipeline

- ⛓️ **Markov chain, reproduced without being told to:** APOE4-positive patients show
  a higher MCI→Dementia transition rate (0.184) than APOE4-negative (0.133) — the
  correct clinical direction, learned purely from data.
- 🔥 **Grad-CAM** shows a fairly diffuse, whole-hemisphere activation pattern rather
  than a focal hippocampal/medial-temporal-lobe region in the cases examined so
  far — flagged as needing a multi-patient check before drawing conclusions about
  learned vs. shortcut features.
- 💊 **Drug-intervention simulation** (Lecanemab −30%, Donanemab −35% on the
  MCI→Dementia transition) uses assumed, not trial-calibrated, effect sizes —
  stated explicitly to avoid overclaiming.

---

## 📓 Notebooks

### `NeuroDT_Notebooks/NeuroDT_GPU.ipynb`

The primary, most complete notebook. Requires a CUDA-capable GPU for the training
cells; everything downstream of training (evaluation, Grad-CAM, Markov chain,
Digital Twin, what-if simulations) runs fine on CPU. Key cells:

- **Cell 7** — set `BEST_MODEL_DIR` / `CACHE_DIR` to wherever you've placed the
  checkpoints and tensor cache (see the Reproducing this project section below)
- **Cell 8 / 8B** — the actual training loop (loss-selected / AUC-selected
  checkpoint criteria respectively) — **skip these** if you already have trained
  checkpoints
- **Cell 9** — Kernel Recovery: reloads a completed checkpoint without retraining;
  use this instead of Cell 8 to go straight to evaluation
- **Cells 10–17** — ablation study (7 architecture variants + 4 classical baselines)
- **Cells 18–33** — evaluation, Grad-CAM, Markov chain, Digital Twin assembly, and
  every what-if simulation

### `NeuroDT_Notebooks/NeuroDT_CPU.ipynb`

The CPU-compatible baseline notebook — the original training run and its (partial,
one-fold) results, kept for direct comparison against the GPU run. Also usable as a
CPU-only path for the post-training analysis cells if no GPU is available.

---

## 🖥️ Dashboard

### 🐳 Running locally with Docker

```bash
cd Dashboard
docker build -t neuro-dt .
docker run -p 8000:8000 \
  -e AZURE_CLIENT_SECRET=<your-service-principal-secret> \
  -e GOOGLE_API_KEY=<your-google-ai-studio-key> \
  -v /path/to/checkpoints:/app/checkpoints \
  neuro-dt
```

Then open `http://localhost:8000`.

### 🔑 Environment variables

| Variable | Purpose | Required? |
|---|---|---|
| `AZURE_CLIENT_SECRET` | Service principal secret for on-demand MRI scan download from Azure Blob Storage | Only if using the blob-path scan input |
| `GOOGLE_API_KEY` | Google AI Studio key for Gemma-generated lifestyle recommendations | No — falls back to a static recommendation template if unset |

Never hardcode these — always read from the environment (the app already does this
via `os.environ.get(...)`).

### ☁️ Deployed environment

Live at `https://neuro-dt-dashboard.azurewebsites.net/` — Azure App Service
(`neuro-dt-dashboard`, Basic B2 plan, West Europe), built via Azure Container
Registry from this same `Dashboard/` folder. The B2 plan's disk limits are why
MRI scans are fetched on demand from Blob Storage rather than bundling the 33GB
tensor cache into the container image.

---

## 🔁 Reproducing this project

1. **Get ADNI access** at [adni.loni.usc.edu](http://adni.loni.usc.edu/) — approval
   takes a few days.
2. **Preprocess and cache** the MRI volumes as described in `NeuroDT_GPU.ipynb`'s
   early cells (MONAI pipeline: `LoadImaged(reader='PydicomReader', force=True)` →
   reorient → resample to 1.5mm → resize/crop to 128³ → intensity-scale → cache as
   `.pt` tensors). `force=True` is required — ADNI-1 era scans predate the DICOM
   preamble standard and fail to load without it.
3. **Train** using `NeuroDT_GPU.ipynb` (GPU strongly recommended — the CPU run is
   documented above specifically as a cautionary example of what goes wrong without
   one) or run `NeuroDT_CPU.ipynb` for the baseline.
4. **Deploy the dashboard** using the `Dashboard/` folder as described above, with
   your own trained checkpoints in place of the ones this repo doesn't include.

---

## 📝 Citation

```
Hendawy, S. (2026). A Multimodal Deep Learning Framework for a Digital Twin
Simulating Alzheimer's Disease Progression. Master's Thesis, Arab Academy for
Science, Technology and Maritime Transport. Supervisors: Prof. F. Maghraby,
Assoc. Prof. A. Salem.
```

## 📜 License

See `LICENSE`.

## 🙏 Acknowledgments

Data used in this project were obtained from the Alzheimer's Disease Neuroimaging
Initiative (ADNI) database ([adni.loni.usc.edu](http://adni.loni.usc.edu/)).
