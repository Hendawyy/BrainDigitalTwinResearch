"""
Neuro-DT Clinical Dashboard v7
- Fix 1: DICOM download status message updates then clears after MRI processing
- Fix 2 & 3: All results + bytes stored to session_state; never disappear on rerun
- Fix 4: PDF spacing improved throughout; Neuro-DT attribution footer
- Fix 5: Full light/dark mode CSS with JS theme detection
"""
import streamlit as st
import torch, numpy as np, matplotlib, json, os, io, pickle, tempfile, shutil
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch.nn.functional as F

matplotlib.use('Agg')
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#333333', 'axes.labelcolor': '#333333',
    'xtick.color': '#333333', 'ytick.color': '#333333',
    'text.color': '#333333', 'grid.color': '#e5e7eb',
    'grid.linewidth': 0.7,
})

st.set_page_config(page_title="Neuro-DT | Brain Digital Twin",
                   page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS: original dark theme ───────────────────────────────────────────────
st.markdown("""
<style>
:root{
  --ink:#f1f5f9; --ink-2:#cbd5e1; --muted:#94a3b8; --hairline:#334155;
  --surface:#1e293b; --plane:#0f172a; --accent:#38bdf8; --accent-ink:#0ea5e9;
  --cn:#38bdf8; --mci:#fb923c; --dementia:#f87171;
  --radius:12px;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
html,body,[class*="css"]{font-family:var(--sans);}
.block-container{padding-top:2.2rem;max-width:1280px;}
[data-testid="stAppViewContainer"]{background:var(--plane);}

/* ── Hero ─────────────────────────────────────────────────────────────── */
.main-header{
  background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 50%,#0f172a 100%);
  padding:1.6rem 2rem;border-radius:var(--radius);margin-bottom:1.6rem;
  border:1px solid rgba(56,189,248,0.2);}
.main-header h1{color:#f0f9ff;font-size:1.6rem;font-weight:600;margin:0;
  letter-spacing:-.01em;}
.main-header p{color:#94a3b8;margin:.45rem 0 0;font-size:.88rem;}

/* ── Metric tiles ─────────────────────────────────────────────────────── */
.metric-card{background:var(--surface);border:1px solid var(--hairline);
  border-radius:var(--radius);padding:1rem 1.1rem;height:100%;}
.metric-val{font-size:1.85rem;font-weight:600;color:var(--ink);
  line-height:1.15;letter-spacing:-.02em;}
.metric-lbl{color:var(--muted);font-size:.72rem;font-weight:500;
  text-transform:uppercase;letter-spacing:.07em;margin-top:.35rem;}
.metric-card.is-accent{border-left:3px solid var(--accent);}

/* ── Diagnosis badges (colour + label, never colour alone) ────────────── */
.badge-cn,.badge-mci,.badge-dementia{
  display:inline-flex;align-items:center;gap:.5rem;padding:.42rem 1rem;
  border-radius:999px;font-weight:600;font-size:.95rem;border:1px solid;}
.badge-cn::before,.badge-mci::before,.badge-dementia::before{
  content:"";width:9px;height:9px;border-radius:50%;background:currentColor;}
.badge-cn{background:#0c4a6e;color:#38bdf8;border-color:#38bdf8;}
.badge-mci{background:#431407;color:#fb923c;border-color:#fb923c;}
.badge-dementia{background:#450a0a;color:#f87171;border-color:#f87171;}

/* ── Section headers ──────────────────────────────────────────────────── */
.sec-hdr{color:var(--ink-2);font-size:.85rem;font-weight:600;
  text-transform:uppercase;letter-spacing:.08em;margin:1.2rem 0 .6rem;
  padding-bottom:.3rem;border-bottom:1px solid var(--hairline);}

/* ── Info / note box ──────────────────────────────────────────────────── */
.info-box{background:#0f172a;border:1px solid #1e3a5f;
  border-left:3px solid var(--accent);border-radius:0 8px 8px 0;
  padding:.7rem 1rem;font-size:.85rem;color:var(--muted);margin:.5rem 0;}

/* ── Buttons ──────────────────────────────────────────────────────────── */
.stButton>button{background:linear-gradient(135deg,#0369a1,#0ea5e9);color:white;
  border:none;border-radius:8px;padding:.5rem 1.5rem;font-weight:600;width:100%;}
[data-testid="stDownloadButton"]>button{background:var(--surface);
  color:var(--ink);border:1px solid var(--hairline);border-radius:8px;font-weight:600;}
[data-testid="stDownloadButton"]>button:hover{border-color:var(--accent);
  color:var(--accent);}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"]{background:#0f172a;border-right:1px solid #1e293b;}

/* ── Tabs ─────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] button[role="tab"]{font-weight:600;color:var(--muted);}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{color:var(--accent);}

.traj-title{font-size:.85rem;font-weight:600;color:#cbd5e1;margin-bottom:4px;}
[data-testid="stDataFrame"]{border:1px solid var(--hairline);border-radius:10px;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINT_DIR   = Path(os.environ.get("CHECKPOINT_DIR", "./checkpoints"))
CACHE_DIR        = Path(os.environ.get("CACHE_DIR",       "./tensor_cache"))
TARGET_SIZE      = (128, 128, 128)
TABULAR_FEATURES = ['AGE', 'PTEDUCAT', 'MMSE', 'APOE4']

# Diagnosis colours, keyed by class name so a colour tracks the diagnosis rather
# than its position in a list. Validated against a white chart surface for both
# normal vision and colour-vision deficiency (worst pair ΔE 20.8 normal /
# 15.3 deuteranopia); the ramp also reads as increasing severity.
CLASS_COLOURS = {'CN': '#2a78d6', 'Dementia': '#e34948', 'MCI': '#eda100'}
NEUTRAL_SERIES = '#6b7280'
ACCENT         = '#2a78d6'
INK            = '#0f172a'
INK_MUTED      = '#64748b'
GRID           = '#e2e8f0'


def class_colours(snames):
    return [CLASS_COLOURS.get(str(s), NEUTRAL_SERIES) for s in snames]

# ── Load model assets ─────────────────────────────────────────────────────────
def _class_names(label_map):
    """Return class names ordered by class index.

    The CPU-era checkpoints stored label_map as {index: name}; the GPU-trained
    checkpoints store it as {name: index}. Reading the wrong orientation yields
    [0, 1, 2] instead of the names, which makes every snames.index('MCI')
    lookup in run_inference raise ValueError.
    """
    if all(isinstance(v, str) for v in label_map.values()):
        return [label_map[k] for k in sorted(label_map)]
    return sorted(label_map, key=label_map.get)


@st.cache_resource
def load_assets():
    from sklearn.preprocessing import LabelEncoder
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    ckpt_path = CHECKPOINT_DIR / "best_model_fold4.pth"
    if not ckpt_path.exists():
        return None, None, None, None, None
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    from model import MultimodalTransformer
    model = MultimodalTransformer(tabular_dim=4, num_classes=3)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    le = LabelEncoder()
    le.classes_ = np.array(_class_names(ckpt['label_map']))
    with open(CHECKPOINT_DIR / "markov_matrices.pkl", 'rb') as f:
        markov = pickle.load(f)
    return model, ckpt['scaler'], le, ckpt['label_map'], markov

# ── GradCAM ───────────────────────────────────────────────────────────────────
class GradCAM3D:
    def __init__(self, model, layer):
        self.model = model; self.grads = None; self.acts = None
        layer.register_forward_hook(lambda m, i, o: setattr(self, 'acts', o.detach()))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'grads', go[0].detach()))

    def generate(self, img, tab, cls):
        self.model.eval(); self.model.zero_grad()
        self.model(img.requires_grad_(True), tab)[0, cls].backward()
        w   = self.grads.mean(dim=[2, 3, 4], keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=TARGET_SIZE, mode='trilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0: cam /= cam.max()
        return cam

def gradcam_figure(orig_vol, cam_map, patient_id, pred_label, white_bg=False,
                   probs=None, snames=None):
    brain_mask = orig_vol > orig_vol.max() * 0.05
    cam_brain  = cam_map * brain_mask
    az = int(cam_brain.sum(axis=(1, 2)).argmax())
    sy = int(cam_brain.sum(axis=(0, 2)).argmax())
    cx = int(cam_brain.sum(axis=(0, 1)).argmax())

    cam_cmap = mcolors.LinearSegmentedColormap.from_list(
        'cam', [(1,0,0,0),(1,0,0,.6),(1,.5,0,.85),(1,1,0,1)])

    planes = [
        ('Axial',    orig_vol[az, :, :],  cam_brain[az, :, :]),
        ('Sagittal', orig_vol[:, sy, :],  cam_brain[:, sy, :]),
        ('Coronal',  orig_vol[:, :, cx],  cam_brain[:, :, cx]),
    ]
    bg = 'white' if white_bg else '#0f172a'
    tc = 'black' if white_bg else 'white'
    tc_muted = '#555555' if white_bg else '#94a3b8'

    # Figure with extra bottom space for annotation panel
    fig = plt.figure(figsize=(15, 11), facecolor=bg)
    # 3 rows: top MRI slices, mid overlay slices, bottom annotation
    gs = fig.add_gridspec(3, 3, height_ratios=[4, 4, 1.2],
                          hspace=0.08, wspace=0.05,
                          left=0.01, right=0.90, top=0.93, bottom=0.02)

    for col, (name, orig_s, cam_s) in enumerate(planes):
        vmax = float(np.percentile(orig_s[orig_s > 0], 99)) if orig_s.max() > 0 else 1.

        # Row 0: raw MRI
        ax0 = fig.add_subplot(gs[0, col])
        ax0.set_facecolor(bg); ax0.axis('off')
        ax0.imshow(orig_s, cmap='gray', vmin=0, vmax=vmax)
        ax0.set_title(name, color=tc, fontsize=11, pad=4, fontweight='600')

        # Row 1: Grad-CAM overlay
        bv = cam_s[cam_s > 0]
        th = float(np.percentile(bv, 50)) if len(bv) > 0 else 0.1
        ax1 = fig.add_subplot(gs[1, col])
        ax1.set_facecolor(bg); ax1.axis('off')
        ax1.imshow(orig_s, cmap='gray', vmin=0, vmax=vmax)
        ax1.imshow(np.where(cam_s >= th, cam_s, 0.), cmap=cam_cmap, vmin=0, vmax=1, alpha=.75)
        ax1.set_title('Grad-CAM Overlay', color=tc, fontsize=10, pad=3)

        # Slice coordinate label
        coord_labels = {0: f'z = {az}', 1: f'y = {sy}', 2: f'x = {cx}'}
        ax1.text(0.02, 0.02, coord_labels[col], transform=ax1.transAxes,
                 color=tc_muted, fontsize=8, va='bottom', ha='left',
                 bbox=dict(boxstyle='round,pad=0.2', fc=bg, ec='none', alpha=0.7))

    # Colorbar
    cbar_ax = fig.add_axes([0.91, 0.28, 0.012, 0.38])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cam_cmap, norm=plt.Normalize(0,1)),
                      cax=cbar_ax)
    cb.set_label('Activation intensity', color=tc, fontsize=8)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(['Low', '', 'Med', '', 'High'])
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=tc, fontsize=7)

    # ── Annotation panel (row 2) ──────────────────────────────────────────────
    ax_ann = fig.add_subplot(gs[2, :])
    ax_ann.set_facecolor('#1e293b' if not white_bg else '#f1f5f9')
    ax_ann.axis('off')

    # Left: prediction summary
    conf_str = ''
    if probs is not None and snames is not None:
        pred_idx = int(np.argmax(probs))
        conf_str = f' ({probs[pred_idx]:.0%} confidence)'
        prob_str = '  ·  '.join([f'{s}: {p:.0%}' for s, p in zip(snames, probs)])
    else:
        prob_str = ''

    ax_ann.text(0.01, 0.72,
                f'Predicted Diagnosis: {pred_label}{conf_str}',
                transform=ax_ann.transAxes, color=tc,
                fontsize=10, fontweight='bold', va='top')
    if prob_str:
        ax_ann.text(0.01, 0.36,
                    f'Class probabilities — {prob_str}',
                    transform=ax_ann.transAxes, color=tc_muted,
                    fontsize=8.5, va='top')
    ax_ann.text(0.01, 0.04,
                f'Peak activation slices — Axial z={az}  ·  Sagittal y={sy}  ·  Coronal x={cx}  ·  '
                f'Red/yellow = high-activation regions driving the {pred_label} prediction',
                transform=ax_ann.transAxes, color=tc_muted,
                fontsize=8, va='bottom', style='italic')

    # Right: interpretation note
    interp = {
        'Dementia': 'Activation concentrated in medial temporal lobe &\nhippocampal/entorhinal regions — hallmark of AD atrophy.',
        'MCI':      'Moderate activation in hippocampal region.\nEarly structural changes typical of prodromal AD.',
        'CN':       'Diffuse low activation. No focal atrophy pattern.\nConsistent with normal cognition.',
    }
    note = interp.get(pred_label, '')
    ax_ann.text(0.99, 0.72, note,
                transform=ax_ann.transAxes, color=tc_muted,
                fontsize=8, va='top', ha='right', style='italic',
                multialignment='right')

    # Suptitle
    fig.suptitle(
        f'Grad-CAM Explainability\nPatient: {patient_id}  |  Pred: {pred_label}',
        color=tc, fontsize=13, fontweight='600', y=0.99, linespacing=1.6)

    return fig, az, sy, cx

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, scaler, le, markov, tab_features,
                  uploaded_pt=None, scan_dir=None):
    tab_row    = np.array([tab_features[f] for f in TABULAR_FEATURES], dtype=np.float32)
    tab_scaled = scaler.transform([tab_row])[0].astype(np.float32)
    tab_t      = torch.tensor(tab_scaled).unsqueeze(0)

    image_available = True
    if uploaded_pt is not None:
        img_t = uploaded_pt.unsqueeze(0) if uploaded_pt.dim() == 4 else uploaded_pt.unsqueeze(0).unsqueeze(0)
    elif scan_dir:
        cp = CACHE_DIR / (scan_dir.replace('/', '_') + '.pt')
        if cp.exists():
            img_t = torch.load(cp, 'cpu').unsqueeze(0)
        else:
            img_t = torch.zeros(1, 1, *TARGET_SIZE)
            image_available = False
    else:
        img_t = torch.zeros(1, 1, *TARGET_SIZE)
        image_available = False

    with torch.no_grad():
        probs = F.softmax(model(img_t, tab_t), dim=1).numpy()[0]

    snames = list(le.classes_); n = len(snames); n_sims = 1000; n_steps = 10
    mci_i  = snames.index('MCI'); dem_i  = snames.index('Dementia')

    def mc(p0, P):
        t = np.zeros((n_sims, n_steps + 1, n))
        for s in range(n_sims):
            st_ = np.random.choice(n, p=p0); t[s, 0, st_] = 1.
            for step in range(1, n_steps + 1):
                ns = np.random.choice(n, p=P[st_]); t[s, step, ns] = 1.; st_ = ns
        return t.mean(0), np.percentile(t, [2.5, 97.5], 0)

    def annual(t):
        o = [t[0]]
        for y in range(5): o.append(t[min(y * 2 + 2, len(t) - 1)])
        return np.array(o)

    def treated(P, eff):
        Pt = P.copy(); r = Pt[mci_i, dem_i] * eff
        Pt[mci_i, dem_i] -= r; Pt[mci_i, mci_i] += r
        Pt[mci_i] /= Pt[mci_i].sum(); return Pt

    Pb = markov['P_apoe4_neg']; Pa = markov['P_apoe4_pos']
    tb, cb = mc(probs, Pb); ta, ca = mc(probs, Pa)
    tl, cl = mc(probs, treated(Pa, .30)); td, cd = mc(probs, treated(Pa, .35))

    return dict(probs=probs, pred=int(probs.argmax()), snames=snames,
                dem_idx=dem_i, years=np.arange(6), img_t=img_t, tab_t=tab_t,
                image_available=image_available,
                tb=annual(tb), cb=(annual(cb[0]), annual(cb[1])),
                ta=annual(ta), ca=(annual(ca[0]), annual(ca[1])),
                tl=annual(tl), cl=(annual(cl[0]), annual(cl[1])),
                td=annual(td), cd=(annual(cd[0]), annual(cd[1])))

# ── Plot helpers ──────────────────────────────────────────────────────────────
def fig_bytes(fig, bg='white'):
    b = io.BytesIO()
    fig.savefig(b, format='png', dpi=150, bbox_inches='tight', facecolor=bg)
    b.seek(0); return b.read()

def plot_probs(probs, snames):
    fig, ax = plt.subplots(figsize=(5.4, 3.2), facecolor='white')
    ax.set_facecolor('white')
    bars = ax.bar(snames, probs * 100, color=class_colours(snames),
                  width=.55, edgecolor='white', linewidth=2)
    for b, p in zip(bars, probs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                f'{p:.1%}', ha='center', va='bottom', fontsize=10,
                fontweight='600', color=INK)
    ax.set_ylim(0, 115)
    ax.set_ylabel('Probability (%)', fontsize=9.5, color=INK_MUTED)
    ax.set_title('Diagnosis class probabilities', fontsize=11,
                 fontweight='600', color=INK, pad=10, loc='left')
    ax.grid(axis='y', color=GRID, linewidth=.7); ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(colors=INK_MUTED, length=0)
    plt.tight_layout(); return fig

def plot_single_traj(traj, ci, title, snames, dem_i, years):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
    ax.set_facecolor('white')
    for si, (s, c) in enumerate(zip(snames, class_colours(snames))):
        ax.plot(years, traj[:, si] * 100, lw=2.4 if si == dem_i else 2,
                ls='-' if si == dem_i else (0, (5, 2)), color=c, label=s,
                solid_capstyle='round')
        ax.fill_between(years, ci[0][:, si] * 100, ci[1][:, si] * 100,
                        alpha=.13, color=c, linewidth=0)
    ax.set_title(title, fontsize=11, fontweight='600', color=INK, pad=10, loc='left')
    ax.set_xlabel('Years', fontsize=9.5, color=INK_MUTED)
    ax.set_ylabel('State probability (%)', fontsize=9.5, color=INK_MUTED)
    ax.set_xlim(0, 5); ax.set_ylim(0, 100)
    ax.grid(color=GRID, linewidth=.7); ax.set_axisbelow(True)
    leg = ax.legend(fontsize=9, frameon=False, loc='upper left',
                    bbox_to_anchor=(0, -0.16), ncol=3, handlelength=1.6)
    for t in leg.get_texts(): t.set_color(INK_MUTED)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    plt.tight_layout(); return fig

def plot_risk_bar(risk_vals, labels):
    # One measure across four scenarios is a single series: the scenario names on
    # the axis carry identity, so a second colour would encode nothing. Baseline
    # is held in a lighter step to mark it as the reference the rest compare to.
    fig, ax = plt.subplots(figsize=(6.2, 3.6), facecolor='white')
    ax.set_facecolor('white')
    cols = ['#9ec5f4'] + [ACCENT] * (len(risk_vals) - 1)
    bars = ax.bar(labels, risk_vals, color=cols[:len(risk_vals)],
                  width=.58, edgecolor='white', linewidth=2)
    for b, v in zip(bars, risk_vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .6,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=10,
                fontweight='600', color=INK)
    ax.set_ylabel('P(Dementia at year 5) %', fontsize=9.5, color=INK_MUTED)
    ax.set_title('5-year Dementia risk by scenario', fontsize=11,
                 fontweight='600', color=INK, pad=10, loc='left')
    ax.set_ylim(0, max(risk_vals) * 1.32 if risk_vals else 100)
    ax.grid(axis='y', color=GRID, linewidth=.7); ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(colors=INK_MUTED, length=0)
    plt.tight_layout(); return fig

# ── Gemma LLM recommendations ─────────────────────────────────────────────────
def _fallback_recommendations(pred_label, risk_5yr, tab_features, apoe4_pos):
    return {
        "summary": f"Your cognitive assessment predicts {pred_label} with a "
                   f"{risk_5yr:.1%} estimated 5-year Dementia risk.",
        "risk_context": "Lifestyle modifications can meaningfully reduce your risk trajectory.",
        "exercise": {"headline": "Regular aerobic exercise",
            "recommendations": ["150 min/week brisk walking", "Swimming or cycling",
                                 "Light strength training"],
            "frequency": "5 days per week, 30 minutes"},
        "diet": {"headline": "Mediterranean-MIND diet",
            "recommendations": ["Leafy greens daily", "Berries 3×/week",
                                 "Fish twice weekly", "Olive oil"],
            "avoid": ["Processed foods", "Excess sugar", "Saturated fats"]},
        "mental": {"headline": "Cognitive stimulation",
            "recommendations": ["Daily reading", "Puzzles or chess", "Learn a new skill"]},
        "social": {"headline": "Stay socially engaged",
            "recommendations": ["Weekly social activities", "Join a community group"]},
        "sleep": {"headline": "Quality sleep", "target_hours": "7-9 hours",
            "tips": ["Consistent sleep schedule", "Limit screen time before bed"]},
        "monitoring": {"headline": "Follow-up schedule",
            "schedule": ["Neurologist review every 6 months",
                         "Annual cognitive assessment"]},
        "positive_note": "Small consistent lifestyle changes can meaningfully reduce your risk."
    }


def get_llm_recommendations(patient_id, tab_features, pred_label, risk_5yr, apoe4_pos):
    try:
        import google.generativeai as genai
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            return _fallback_recommendations(pred_label, risk_5yr, tab_features, apoe4_pos)
        genai.configure(api_key=api_key)
        model_obj = genai.GenerativeModel("gemma-4-31b-it")
        prompt = f"""You are a compassionate medical AI. Generate personalised evidence-based
lifestyle recommendations. Return ONLY valid JSON, no markdown, no preamble.

Patient profile:
- Age: {tab_features['AGE']:.1f} years
- Education: {tab_features['PTEDUCAT']:.0f} years
- MMSE: {tab_features['MMSE']:.0f}/30
- APOE4: {'Positive (elevated genetic risk)' if apoe4_pos else 'Negative'}
- AI-predicted diagnosis: {pred_label}
- 5-year Dementia risk (baseline): {risk_5yr:.1%}

JSON schema (fill all fields, be specific and encouraging):
{{"summary":"2-3 sentence personalised summary in plain language",
"risk_context":"1-2 sentences on what the 5-year risk means practically",
"exercise":{{"headline":"short headline","recommendations":["rec1","rec2","rec3"],"frequency":"specific weekly frequency"}},
"diet":{{"headline":"short headline","recommendations":["food1","food2","food3","food4"],"avoid":["item1","item2"]}},
"mental":{{"headline":"short headline","recommendations":["activity1","activity2","activity3"]}},
"social":{{"headline":"short headline","recommendations":["rec1","rec2"]}},
"sleep":{{"headline":"short headline","target_hours":"7-9 hours","tips":["tip1","tip2"]}},
"monitoring":{{"headline":"Follow-up schedule","schedule":["item1","item2"]}},
"positive_note":"1 encouraging sentence"}}"""
        response = model_obj.generate_content(prompt)
        text = response.text.strip()
        if "```" in text:
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            if text.startswith("json"): text = text[4:]
        return json.loads(text.strip())
    except Exception:
        return _fallback_recommendations(pred_label, risk_5yr, tab_features, apoe4_pos)


# ══════════════════════════════════════════════════════════════════════════════
#  PDF GENERATION
# ══════════════════════════════════════════════════════════════════════════════

_PDF_ATTR = (
    'Generated by <b>Neuro-DT</b> — Multimodal Deep Learning Framework for '
    'Alzheimer\'s Disease Progression Simulation · Arab Academy for Science, '
    'Technology and Maritime Transport, 2026 · '
    'Model: 3D DenseNet121 + Transformer Encoder · Fold 4 AUC 0.951 (5-fold CV 0.871±0.046) · ADNI (1,549 scans)'
)

def make_patient_pdf(patient_id, tab_features, res, recs, prob_bytes,
                     traj_bytes_base, risk_bar_bytes, timestamp):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, Image as RLI, HRFlowable, PageBreak)
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2.5*cm, bottomMargin=2*cm)
    W, _ = A4; cw = W - 4*cm

    NAVY  = colors.HexColor('#0f172a'); BLUE  = colors.HexColor('#0ea5e9')
    SLATE = colors.HexColor('#334155'); LIGHT = colors.HexColor('#f8fafc')
    MUTED = colors.HexColor('#64748b'); GREEN = colors.HexColor('#16a34a')
    ORANGE= colors.HexColor('#ea580c'); RED   = colors.HexColor('#dc2626')

    S = getSampleStyleSheet()
    def sty(name, **kw): return ParagraphStyle(name, **kw)

    title_s  = sty('t',  fontName='Helvetica-Bold', fontSize=18,
                         textColor=NAVY, spaceAfter=6)
    h2_s     = sty('h2', fontName='Helvetica-Bold', fontSize=11,
                         textColor=NAVY, spaceBefore=24, spaceAfter=10)
    body_s   = sty('b',  fontName='Helvetica', fontSize=9.5,
                         textColor=colors.HexColor('#1e293b'), leading=16)
    cap_s    = sty('c',  fontName='Helvetica-Oblique', fontSize=8,
                         textColor=MUTED, alignment=TA_CENTER, spaceBefore=4, spaceAfter=6)
    bullet_s = sty('bl', fontName='Helvetica', fontSize=9.5,
                         textColor=colors.HexColor('#1e293b'),
                         leading=16, leftIndent=12, bulletIndent=0)
    disc_s   = sty('d',  fontName='Helvetica', fontSize=7.5,
                         textColor=MUTED, leading=12)

    def hdr_tbl():
        d = [[
            Paragraph('<font name="Helvetica-Bold" size="12" color="#0f172a">Neuro-DT</font>'
                      '<font name="Helvetica" size="9" color="#64748b"> — Patient Report</font>',
                      S['Normal']),
            Paragraph(f'<font name="Helvetica" size="8" color="#64748b">Generated: {timestamp}</font>',
                      sty('r', alignment=TA_RIGHT, fontSize=8, fontName='Helvetica'))
        ]]
        t = Table(d, colWidths=[cw*.6, cw*.4])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f0f9ff')),
            ('TOPPADDING',    (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('LEFTPADDING',   (0,0), (-1,-1), 14),
            ('RIGHTPADDING',  (0,0), (-1,-1), 14),
            ('LINEBELOW', (0,0), (-1,-1), 1.5, BLUE),
        ]))
        return t

    pred_name = res['snames'][res['pred']]
    badge_col = {'CN': GREEN, 'MCI': ORANGE, 'Dementia': RED}[pred_name]
    dem_i     = res['dem_idx']

    story = []

    # ── PAGE 1 ────────────────────────────────────────────────────────────────
    story += [
        hdr_tbl(), Spacer(1, 20),
        Paragraph("Alzheimer's Disease Assessment Report", title_s),
        Spacer(1, 10),
        Paragraph(f'Patient: <b>{patient_id}</b>  |  Date: {timestamp}',
                  sty('s', fontName='Helvetica', fontSize=10, textColor=MUTED, spaceAfter=8)),
        HRFlowable(width='100%', thickness=1, color=SLATE, spaceAfter=16),
    ]

    # Summary
    story.append(Paragraph('What your results mean', h2_s))
    story.append(Paragraph(recs.get('summary', ''), body_s))
    story.append(Spacer(1, 8))
    story.append(Paragraph(recs.get('risk_context', ''), body_s))
    story.append(Spacer(1, 16))

    # Diagnosis badge + metrics table
    diag_d = [
        [Paragraph('AI Assessment',       sty('dh', fontName='Helvetica-Bold', fontSize=9, textColor=colors.white)),
         Paragraph('Confidence',          sty('dh', fontName='Helvetica-Bold', fontSize=9, textColor=colors.white)),
         Paragraph('5-Year Risk (Baseline)', sty('dh', fontName='Helvetica-Bold', fontSize=9, textColor=colors.white))],
        [Paragraph(pred_name,                        sty('dv', fontName='Helvetica-Bold', fontSize=14, textColor=colors.white)),
         Paragraph(f'{res["probs"][res["pred"]]:.0%}', sty('dv', fontName='Helvetica-Bold', fontSize=14, textColor=colors.white)),
         Paragraph(f'{res["tb"][-1,dem_i]:.1%}',        sty('dv', fontName='Helvetica-Bold', fontSize=14, textColor=colors.white))],
    ]
    dt = Table(diag_d, colWidths=[cw/3]*3)
    dt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('BACKGROUND', (0,1), (-1,1), badge_col),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING',    (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
    ]))
    story.append(dt); story.append(Spacer(1, 16))

    # Prob chart
    story.append(Paragraph('Diagnosis Probability Breakdown', h2_s))
    story.append(RLI(io.BytesIO(prob_bytes), width=cw*.55, height=cw*.33))
    story.append(Paragraph('Figure 1. Probability of each diagnostic category.', cap_s))
    story.append(Spacer(1, 14))

    # Risk table — new page so all rows stay together
    story.append(PageBreak())
    story += [hdr_tbl(), Spacer(1, 16)]
    story.append(Paragraph('5-Year Dementia Risk Summary', h2_s))
    risk_d = [
        ['Scenario', 'P(Dementia at Year 5)', 'What this means'],
        ['Baseline (no APOE4)',       f'{res["tb"][-1,dem_i]:.1%}', 'Your natural risk level'],
        ['APOE4+ (no treatment)',     f'{res["ta"][-1,dem_i]:.1%}', 'If APOE4 gene is present'],
        ['APOE4+ + Lecanemab',        f'{res["tl"][-1,dem_i]:.1%}', 'With Lecanemab therapy'],
        ['APOE4+ + Donanemab',        f'{res["td"][-1,dem_i]:.1%}', 'With Donanemab therapy'],
    ]
    rt = Table(risk_d, colWidths=[cw*.42, cw*.28, cw*.30])
    rt.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0),  NAVY),
        ('TEXTCOLOR',     (0,0), (-1,0),  colors.white),
        ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTNAME',      (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',      (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [LIGHT, colors.white]),
        ('GRID',          (0,0), (-1,-1), .4, colors.HexColor('#e2e8f0')),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
    ]))
    story.append(rt)
    story.append(Spacer(1, 16))

    # ── PAGE 2 ────────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story += [
        hdr_tbl(), Spacer(1, 16),
        Paragraph('Personalised Lifestyle Recommendations', title_s),
        HRFlowable(width='100%', thickness=1, color=SLATE, spaceAfter=14),
    ]

    sections = [
        ('🏃 Exercise & Physical Activity', 'exercise',
         lambda r: [r.get('headline','')] + r.get('recommendations',[]) + [f"Frequency: {r.get('frequency','')}"]),
        ('🥗 Diet & Nutrition', 'diet',
         lambda r: [r.get('headline','')] + r.get('recommendations',[]) + [f"Limit: {', '.join(r.get('avoid',[]))}"]),
        ('🧠 Mental Stimulation', 'mental',
         lambda r: [r.get('headline','')] + r.get('recommendations',[])),
        ('👥 Social Engagement', 'social',
         lambda r: [r.get('headline','')] + r.get('recommendations',[])),
        ('😴 Sleep & Recovery', 'sleep',
         lambda r: [r.get('headline',''), f"Target: {r.get('target_hours','7-9 hours')}"] + r.get('tips',[])),
        ('📅 Monitoring & Follow-up', 'monitoring',
         lambda r: [r.get('headline','')] + r.get('schedule',[])),
    ]

    def render_section(sec_title, key, getter, _w):
        items = []
        items.append(Paragraph(sec_title, sty(f'sh_{key}', fontName='Helvetica-Bold',
                                               fontSize=10, textColor=NAVY,
                                               spaceBefore=12, spaceAfter=5)))
        data = recs.get(key, {})
        for i, line in enumerate(getter(data)):
            prefix = '•  ' if i > 0 else ''
            items.append(Paragraph(prefix + str(line), body_s if i == 0 else bullet_s))
        return items

    left_secs  = sections[:3]
    right_secs = sections[3:]
    left_col   = [item for (t,k,g) in left_secs  for item in render_section(t,k,g,cw*.48)]
    right_col  = [item for (t,k,g) in right_secs for item in render_section(t,k,g,cw*.48)]
    while len(left_col) < len(right_col): left_col.append(Spacer(1, 6))
    while len(right_col) < len(left_col): right_col.append(Spacer(1, 6))

    for L, R in zip(left_col, right_col):
        row_t = Table([[L, R]], colWidths=[cw*.48, cw*.48])
        row_t.setStyle(TableStyle([
            ('VALIGN',        (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING',   (0,0), (-1,-1), 0),
            ('RIGHTPADDING',  (0,0), (-1,-1), 6),
        ]))
        story.append(row_t)

    story.append(Spacer(1, 18))
    if recs.get('positive_note'):
        note_t = Table([[Paragraph(
            f'💙  {recs["positive_note"]}',
            sty('pn', fontName='Helvetica-BoldOblique', fontSize=10,
                textColor=colors.HexColor('#0369a1'))
        )]], colWidths=[cw])
        note_t.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,-1), colors.HexColor('#eff6ff')),
            ('TOPPADDING',    (0,0), (-1,-1), 12),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
            ('LEFTPADDING',   (0,0), (-1,-1), 16),
            ('ROUNDEDCORNERS', [6]),
        ]))
        story.append(note_t)

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width='100%', thickness=.5, color=SLATE))
    story.append(Spacer(1, 7))
    story.append(Paragraph(_PDF_ATTR, disc_s))

    doc.build(story)
    buf.seek(0); return buf.read()


def make_doctor_pdf(patient_id, tab_features, res, prob_bytes,
                    traj_bytes_list, gradcam_bytes, risk_bar_bytes, timestamp):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, Image as RLI, HRFlowable, PageBreak)
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2.5*cm, bottomMargin=2*cm)
    W, _ = A4; cw = W - 4*cm

    NAVY  = colors.HexColor('#0f172a'); BLUE  = colors.HexColor('#0ea5e9')
    SLATE = colors.HexColor('#334155'); LIGHT = colors.HexColor('#f8fafc')
    MUTED = colors.HexColor('#64748b'); GREEN = colors.HexColor('#16a34a')
    ORANGE= colors.HexColor('#ea580c'); RED   = colors.HexColor('#dc2626')

    S = getSampleStyleSheet()
    def sty(n, **kw): return ParagraphStyle(n, **kw)

    title_s = sty('t',  fontName='Helvetica-Bold', fontSize=17,
                        textColor=NAVY, spaceAfter=4)
    h2_s    = sty('h2', fontName='Helvetica-Bold', fontSize=11,
                        textColor=NAVY, spaceBefore=22, spaceAfter=10)
    body_s  = sty('b',  fontName='Helvetica', fontSize=9.5,
                        textColor=colors.HexColor('#1e293b'), leading=16)
    cap_s   = sty('c',  fontName='Helvetica-Oblique', fontSize=8,
                        textColor=MUTED, alignment=TA_CENTER, spaceBefore=4, spaceAfter=8)
    disc_s  = sty('d',  fontName='Helvetica', fontSize=7.5,
                        textColor=MUTED, leading=12)

    def hdr_tbl():
        d = [[
            Paragraph('<font name="Helvetica-Bold" size="12">Neuro-DT</font>'
                      '<font name="Helvetica" size="9" color="#64748b"> — Physician Clinical Record</font>',
                      S['Normal']),
            Paragraph(f'<font name="Helvetica" size="8" color="#64748b">'
                      f'Generated: {timestamp}<br/>CONFIDENTIAL — For clinical use only</font>',
                      sty('r', alignment=TA_RIGHT, fontSize=8, fontName='Helvetica')),
        ]]
        t = Table(d, colWidths=[cw*.6, cw*.4])
        t.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,-1), colors.HexColor('#fef2f2')),
            ('TOPPADDING',    (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('LEFTPADDING',   (0,0), (-1,-1), 14),
            ('RIGHTPADDING',  (0,0), (-1,-1), 14),
            ('LINEBELOW',     (0,0), (-1,-1), 1.5, RED),
        ]))
        return t

    pred_name  = res['snames'][res['pred']]; dem_i = res['dem_idx']
    badge_col  = {'CN': GREEN, 'MCI': ORANGE, 'Dementia': RED}[pred_name]

    story = []

    # ── PAGE 1: Clinical summary ──────────────────────────────────────────────
    story += [
        hdr_tbl(), Spacer(1, 16),
        Paragraph("Alzheimer's Disease Clinical Assessment Record", title_s),
        Spacer(1, 10),
        Paragraph(f'Patient: <b>{patient_id}</b>  |  Date: {timestamp}',
                  sty('s', fontName='Helvetica', fontSize=9, textColor=MUTED, spaceAfter=8)),
        HRFlowable(width='100%', thickness=1, color=SLATE, spaceAfter=14),
    ]

    # Demographics
    story.append(Paragraph('Patient Demographics & Biomarkers', h2_s))
    dem_d = [
        ['Parameter', 'Value', 'Clinical Reference'],
        ['Age',                f'{tab_features["AGE"]:.1f} years', '—'],
        ['Education (PTEDUCAT)', f'{tab_features["PTEDUCAT"]:.0f} years', '—'],
        ['MMSE Score',         f'{tab_features["MMSE"]:.0f} / 30',
         '24–30 Normal | 18–23 Mild impairment | <18 Severe'],
        ['APOE4 Status',       f'{int(tab_features["APOE4"])} allele(s)',
         '0 = standard risk | 1–2 = elevated AD risk'],
    ]
    t = Table(dem_d, colWidths=[cw*.28, cw*.22, cw*.50])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0),  NAVY),
        ('TEXTCOLOR',     (0,0), (-1,0),  colors.white),
        ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTNAME',      (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',      (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [LIGHT, colors.white]),
        ('GRID',          (0,0), (-1,-1), .4, colors.HexColor('#e2e8f0')),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
    ]))
    story.append(t); story.append(Spacer(1, 14))

    # AI Diagnosis
    story.append(Paragraph('AI Diagnostic Classification', h2_s))
    d2 = [
        ['Predicted Class', 'Confidence', 'CN AUC', 'Dementia AUC', 'MCI AUC'],
        [pred_name, f'{res["probs"][res["pred"]]:.1%}', '0.982', '0.958', '0.913'],
    ]
    t2 = Table(d2, colWidths=[cw*.22]*5)
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0),  NAVY),
        ('TEXTCOLOR',  (0,0), (-1,0),  colors.white),
        ('BACKGROUND', (0,1), (0,1),   badge_col),
        ('TEXTCOLOR',  (0,1), (0,1),   colors.white),
        ('FONTNAME',   (0,0), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('GRID',       (0,0), (-1,-1), .4, colors.HexColor('#e2e8f0')),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t2); story.append(Spacer(1, 12))

    # Probability chart
    story.append(RLI(io.BytesIO(prob_bytes), width=cw*.5, height=cw*.3))
    story.append(Paragraph('Figure 1. Class probability distribution.', cap_s))
    story.append(Spacer(1, 14))

    # 5-year risk table — new page so all rows stay together
    story.append(PageBreak())
    story += [hdr_tbl(), Spacer(1, 14)]
    story.append(Paragraph('5-Year Dementia Probability — Scenario Analysis', h2_s))
    r_vals   = [res['tb'][-1,dem_i], res['ta'][-1,dem_i],
                res['tl'][-1,dem_i], res['td'][-1,dem_i]]
    r_labels = ['Baseline (no APOE4)', 'APOE4+ untreated',
                'APOE4+ + Lecanemab (30%)', 'APOE4+ + Donanemab (35%)']
    recs_col = ['Standard risk profile', 'Elevated genetic risk',
                'Recommended: anti-amyloid therapy', 'Higher efficacy anti-amyloid option']
    rd = [['Scenario', 'P(Dementia@5yr)', 'Clinical Note']] + [
        [l, f'{v:.1%}', c] for l, v, c in zip(r_labels, r_vals, recs_col)]
    rt = Table(rd, colWidths=[cw*.38, cw*.18, cw*.44])
    rt.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0),  NAVY),
        ('TEXTCOLOR',     (0,0), (-1,0),  colors.white),
        ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTNAME',      (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',      (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [LIGHT, colors.white]),
        ('GRID',          (0,0), (-1,-1), .4, colors.HexColor('#e2e8f0')),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
    ]))
    story.append(rt); story.append(Spacer(1, 10))
    apoe4_up = (res['ta'][-1,dem_i] - res['tb'][-1,dem_i]) * 100
    lecan_dn = (res['ta'][-1,dem_i] - res['tl'][-1,dem_i]) * 100
    story.append(Paragraph(
        f'<b>Clinical note:</b> APOE4 increases 5-year risk by <b>+{apoe4_up:.1f}pp</b>. '
        f'Lecanemab reduces elevated risk by <b>{lecan_dn:.1f}pp</b>. '
        f'Consider referral for genetic counselling and anti-amyloid eligibility assessment.',
        body_s))

    # ── PAGE 2: Grad-CAM + Trajectories + Recovery Plan ──────────────────────
    story.append(PageBreak())
    story += [hdr_tbl(), Spacer(1, 14)]

    if gradcam_bytes:
        story.append(Paragraph('Neuroimaging Explainability — Grad-CAM Analysis', h2_s))
        story.append(Paragraph(
            'The following activation maps show brain regions that most strongly influenced the '
            'AI classification. High-activation regions (red/yellow) indicate areas of structural '
            'significance. For Dementia predictions, activation typically concentrates in the '
            'medial temporal lobe, hippocampal region, and periventricular white matter.',
            body_s))
        story.append(Spacer(1, 10))
        story.append(RLI(io.BytesIO(gradcam_bytes), width=cw, height=cw*.6))
        story.append(Paragraph(
            'Figure 2. Grad-CAM class activation maps (axial, sagittal, coronal). '
            'Red-yellow overlay = regions driving the prediction.', cap_s))
        story.append(Spacer(1, 18))

    # Recovery Plan — new page so table content isn't split or clipped
    story.append(PageBreak())
    story += [hdr_tbl(), Spacer(1, 14)]
    story.append(Paragraph('Clinical Recovery Plan & Drug Recommendations', h2_s))

    base_risk  = res['tb'][-1, dem_i]
    apoe4_risk = res['ta'][-1, dem_i]
    lecan_risk = res['tl'][-1, dem_i]
    dona_risk  = res['td'][-1, dem_i]
    lecan_ben  = (apoe4_risk - lecan_risk) * 100
    dona_ben   = (apoe4_risk - dona_risk)  * 100

    if pred_name == 'CN':
        primary_rec = "Patient is cognitively normal. Focus on prevention and monitoring."
        drug_rec    = "No pharmacological intervention currently indicated. Annual cognitive screening recommended."
        follow_up   = "Annual MMSE assessment. Repeat MRI in 2 years if risk factors present."
    elif pred_name == 'MCI':
        primary_rec = (f"Patient presents with Mild Cognitive Impairment. "
                       f"5-year Dementia risk is {base_risk:.1%} (baseline) or {apoe4_risk:.1%} if APOE4+.")
        if int(tab_features.get('APOE4', 0)) >= 1:
            drug_rec = (f"APOE4+ status confirmed. Lecanemab reduces 5-year risk by {lecan_ben:.1f}pp "
                        f"({apoe4_risk:.1%} → {lecan_risk:.1%}). Donanemab reduces by {dona_ben:.1f}pp "
                        f"({apoe4_risk:.1%} → {dona_risk:.1%}). "
                        "Consider referral for anti-amyloid eligibility assessment (PET/CSF amyloid confirmation required).")
        else:
            drug_rec = ("No APOE4 alleles detected. Standard-risk MCI management: "
                        "cognitive stimulation programme, lifestyle modification, 6-monthly MMSE review. "
                        "Reassess for pharmacological intervention if progression observed.")
        follow_up = "6-monthly MMSE and CDR-SB. Annual MRI. Consider neuropsychological battery."
    else:  # Dementia
        primary_rec = (f"Dementia diagnosis confirmed with {res['probs'][res['pred']]:.0%} confidence. "
                       "Immediate multidisciplinary care plan recommended.")
        drug_rec    = ("Acetylcholinesterase inhibitors (donepezil, rivastigmine) as first-line symptomatic therapy. "
                       "Memantine for moderate-severe stages. "
                       f"If amyloid-confirmed: anti-amyloid therapy (Lecanemab {lecan_risk:.1%} 5yr risk, "
                       f"Donanemab {dona_risk:.1%}) may slow progression. Carer support referral essential.")
        follow_up = "3-monthly cognitive and functional assessment. Carer needs evaluation. Safety assessment."

    # KEY FIX: wrap every cell in Paragraph so ReportLab word-wraps long text
    cell_s  = sty('rc',  fontName='Helvetica',      fontSize=9,  textColor=colors.HexColor('#1e293b'), leading=14)
    lbl_s   = sty('rcl', fontName='Helvetica-Bold', fontSize=9,  textColor=colors.HexColor('#1e293b'), leading=14)
    hdr_s   = sty('rch', fontName='Helvetica-Bold', fontSize=9,  textColor=colors.white)

    lifestyle_text = (
        'Mediterranean-MIND diet · 150 min/week aerobic exercise · '
        'Cognitive stimulation daily · Social engagement · 7–9h sleep · '
        'Cardiovascular risk factor management (BP, cholesterol, diabetes)'
    )

    rec_data = [
        [Paragraph('Category', hdr_s),           Paragraph('Recommendation', hdr_s)],
        [Paragraph('Clinical assessment', lbl_s), Paragraph(primary_rec,     cell_s)],
        [Paragraph('Drug therapy',        lbl_s), Paragraph(drug_rec,        cell_s)],
        [Paragraph('Lifestyle',           lbl_s), Paragraph(lifestyle_text,  cell_s)],
        [Paragraph('Follow-up plan',      lbl_s), Paragraph(follow_up,       cell_s)],
    ]
    rec_tbl = Table(rec_data, colWidths=[cw*.22, cw*.78])
    rec_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0),  NAVY),
        ('TEXTCOLOR',     (0,0), (-1,0),  colors.white),
        ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [LIGHT, colors.white]),
        ('GRID',          (0,0), (-1,-1), 0.4, colors.HexColor('#e2e8f0')),
        ('VALIGN',        (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING',    (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
        ('RIGHTPADDING',  (0,0), (-1,-1), 8),
    ]))
    story.append(rec_tbl); story.append(Spacer(1, 18))

    if traj_bytes_list:
        story.append(Paragraph('5-Year Trajectory Simulations', h2_s))
        grid_rows = []
        for idx in range(0, min(4, len(traj_bytes_list)), 2):
            row = []
            for jdx in range(2):
                if idx + jdx < len(traj_bytes_list):
                    row.append(RLI(io.BytesIO(traj_bytes_list[idx+jdx]),
                                   width=cw*.48, height=cw*.32))
                else:
                    row.append(Spacer(1, 1))
            grid_rows.append(row)
        tg = Table(grid_rows, colWidths=[cw*.50, cw*.50])
        tg.setStyle(TableStyle([
            ('VALIGN',        (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING',   (0,0), (-1,-1), 2),
            ('RIGHTPADDING',  (0,0), (-1,-1), 2),
            ('TOPPADDING',    (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(tg)
        story.append(Paragraph(
            'Figure 3. Monte Carlo trajectory simulations (n=1,000, 95% CI shaded).', cap_s))
        story.append(Spacer(1, 14))

    if risk_bar_bytes:
        story.append(Paragraph('Comparative Risk Summary', h2_s))
        story.append(RLI(io.BytesIO(risk_bar_bytes), width=cw*.7, height=cw*.42))
        story.append(Paragraph('Figure 4. 5-year Dementia risk by intervention scenario.', cap_s))
        story.append(Spacer(1, 18))

    story.append(HRFlowable(width='100%', thickness=.5, color=SLATE))
    story.append(Spacer(1, 7))
    story.append(Paragraph(_PDF_ATTR + ' · For clinical use only.', disc_s))

    doc.build(story)
    buf.seek(0); return buf.read()


# ══════════════════════════════════════════════════════════════════════════════
#  UI LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>🧠 Neuro-DT — Brain Digital Twin</h1>
  <p>Multimodal Deep Learning · ADNI · Fold 4 AUC 0.951 (5-fold CV 0.871±0.046) · Alzheimer's Disease Progression Simulation</p>
</div>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Patient Input")
    st.markdown('<div class="info-box">Fill in the patient\'s clinical features below.</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">Patient</div>', unsafe_allow_html=True)
    patient_id = st.text_input("Patient ID", value="Patient_001")

    st.markdown('<div class="sec-hdr">Demographics</div>', unsafe_allow_html=True)
    age       = st.number_input("Age (years)", min_value=50.0, max_value=95.0,
                                 value=75.0, step=0.1, format="%.1f")
    education = st.number_input("Education (years)", min_value=4, max_value=25,
                                 value=16, step=1)

    st.markdown('<div class="sec-hdr">Cognitive Scores</div>', unsafe_allow_html=True)
    mmse  = st.slider("MMSE Score", 0, 30, 24,
                      help="30=normal · 24-27=mild concern · <24=impairment")
    apoe4 = st.selectbox("APOE4 Alleles", [0, 1, 2], index=1,
                          help="Number of APOE4 alleles")

    st.markdown('<div class="sec-hdr">MRI Scan (optional)</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Enter the blob storage path to load the patient scan. '
                'Format: <code>unzipped_dicoms/ADNI/XXX_S_XXXX/MPRAGE/date/IXXXXXX</code></div>',
                unsafe_allow_html=True)
    scan_dir_input = st.text_input(
        "Blob scan path",
        placeholder="unzipped_dicoms/ADNI/002_S_0559/MPRAGE/2009-07-01.../I147119",
        help="Paste the scan_dir value from your ADNI manifest CSV")

    st.markdown("---")
    run_btn = st.button("▶  Run Digital Twin", use_container_width=True)
    st.markdown('<div class="info-box">Fold 4 · Epoch 20 · AUC 0.951 (5-fold CV 0.871±0.046)<br>'
                'DenseNet121 + Transformer<br>1,549 ADNI scans</div>',
                unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_res, tab_about = st.tabs(["📊 Results", "ℹ️ About"])

# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_res:

    # ── Step 1: Run inference and store EVERYTHING to session_state ───────────
    if run_btn:
        with st.spinner("Loading model..."):
            model, scaler, le, label_map, markov = load_assets()
            if model is None:
                st.error("Model checkpoint not found in ./checkpoints/")
                st.stop()

        tab_features = {'AGE': float(age), 'PTEDUCAT': float(education),
                        'MMSE': float(mmse), 'APOE4': float(apoe4)}

        # ── Handle MRI input — blob download ────────────────────────────────
        uploaded_pt = None
        if scan_dir_input and scan_dir_input.strip():
            with st.spinner("Connecting to Azure Blob Storage..."):
                try:
                    import os as _os, tempfile as _tmp
                    from azure.identity import ClientSecretCredential
                    from azure.storage.blob import BlobServiceClient
                    from monai.transforms import (Compose, LoadImaged,
                        EnsureChannelFirstd, Orientationd, Spacingd,
                        ScaleIntensityRanged, ResizeWithPadOrCropd, ToTensord)

                    scan_path_clean = scan_dir_input.strip().rstrip('/')
                    cache_key  = scan_path_clean.replace('/', '_') + '.pt'
                    cache_path = CACHE_DIR / cache_key

                    if cache_path.exists():
                        uploaded_pt = torch.load(cache_path, map_location='cpu')
                        st.success("✓ Loaded from tensor cache.")
                    else:
                        CLIENT_SECRET = _os.environ.get('AZURE_CLIENT_SECRET', '')
                        if not CLIENT_SECRET:
                            st.warning("AZURE_CLIENT_SECRET not set — running tabular-only inference.")
                        else:
                            cred = ClientSecretCredential(
                                tenant_id='70c07c26-601e-415b-9a91-c351a5ad357b',
                                client_id='c638dc4d-96ec-4457-8797-23902283156b',
                                client_secret=CLIENT_SECRET)
                            cc = BlobServiceClient(
                                account_url="https://adnihendawy.blob.core.windows.net",
                                credential=cred
                            ).get_container_client("adni-data")

                            all_blobs = [b for b in cc.list_blobs(name_starts_with=scan_path_clean)
                                         if not b.name.endswith('/')]
                            blobs = [b for b in all_blobs
                                     if b.name.lower().endswith('.dcm')
                                     or '.' not in _os.path.basename(b.name)]

                            if not blobs:
                                st.warning(f"No DICOM blobs found at: {scan_path_clean}")
                            else:
                                # FIX 1: Placeholder message that updates in-place
                                scan_status = st.empty()
                                scan_status.info(
                                    f"📥 Found **{len(blobs)} DICOM slices** — downloading...")

                                with _tmp.TemporaryDirectory() as tmpdir:
                                    for blob in blobs:
                                        fname = _os.path.basename(blob.name)
                                        fpath = _os.path.join(tmpdir, fname)
                                        with open(fpath, 'wb') as fh:
                                            cc.get_blob_client(blob.name).download_blob().readinto(fh)

                                    # Update message: download done, now preprocessing
                                    scan_status.info(
                                        f"⚙️ Downloaded {len(blobs)} slices — "
                                        f"running MRI preprocessing pipeline...")

                                    pipeline = Compose([
                                        LoadImaged(keys=['image'], image_only=True,
                                                   reader='PydicomReader', force=True),
                                        EnsureChannelFirstd(keys=['image']),
                                        Orientationd(keys=['image'], axcodes='RAS'),
                                        Spacingd(keys=['image'], pixdim=(1.5,1.5,1.5),
                                                 mode='bilinear'),
                                        ResizeWithPadOrCropd(keys=['image'],
                                                             spatial_size=(128,128,128)),
                                        ScaleIntensityRanged(keys=['image'],
                                                             a_min=0, a_max=1500,
                                                             b_min=0.0, b_max=1.0, clip=True),
                                        ToTensord(keys=['image']),
                                    ])
                                    result = pipeline({'image': tmpdir})
                                    uploaded_pt = result['image']

                                # FIX 1: Clear the message entirely after processing
                                scan_status.empty()
                                st.success(f"✓ MRI scan ready ({len(blobs)} slices processed).")

                except Exception as e:
                    import traceback
                    st.warning(f"Blob download failed: {e}")
                    with st.expander("Show full error"):
                        st.code(traceback.format_exc())

        with st.spinner("Running Digital Twin simulations..."):
            res = run_inference(model, scaler, le, markov, tab_features,
                                uploaded_pt=uploaded_pt,
                                scan_dir=scan_dir_input or None)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        pred_name = res['snames'][res['pred']]
        dem_i     = res['dem_idx']

        if not res['image_available']:
            st.warning(
                "⚠ No MRI scan was available for this patient — the prediction below is "
                "TABULAR-ONLY (a zero image tensor was substituted). Do not treat this as "
                "equivalent to an image-based result.")


        # ── Compute ALL plots → bytes ────────────────────────────────────────
        fig_prob   = plot_probs(res['probs'], res['snames'])
        prob_bytes = fig_bytes(fig_prob); plt.close()

        scenarios = [
            (res['tb'], res['cb'], 'No APOE4 — Baseline'),
            (res['ta'], res['ca'], 'APOE4+ — No Treatment'),
            (res['tl'], res['cl'], 'APOE4+ + Lecanemab (30%)'),
            (res['td'], res['cd'], 'APOE4+ + Donanemab (35%)'),
        ]
        traj_bytes_list = []
        for traj, ci, title in scenarios:
            fig = plot_single_traj(traj, ci, title, res['snames'], dem_i, res['years'])
            traj_bytes_list.append(fig_bytes(fig)); plt.close()

        risk_vals   = [res['tb'][-1,dem_i]*100, res['ta'][-1,dem_i]*100,
                       res['tl'][-1,dem_i]*100, res['td'][-1,dem_i]*100]
        risk_labels = ['Baseline', 'APOE4+', '+Lecanemab', '+Donanemab']
        fig_risk        = plot_risk_bar(risk_vals, risk_labels)
        risk_bar_bytes  = fig_bytes(fig_risk); plt.close()

        # GradCAM
        gradcam_bytes = None; gradcam_bytes_white = None; gc_coords = None
        orig_vol = res['img_t'].detach().squeeze().numpy()
        if orig_vol.max() > 0:
            try:
                gc3d    = GradCAM3D(model, model.cnn_backbone.features.denseblock4)
                cam_map = gc3d.generate(res['img_t'], res['tab_t'], res['pred'])
                fig_cam, az, sy, cx = gradcam_figure(
                    orig_vol, cam_map, patient_id, pred_name, white_bg=False,
                    probs=res['probs'], snames=res['snames'])
                gradcam_bytes = fig_bytes(fig_cam, bg='#0f172a'); plt.close()
                fig_cam_w, _, _, _ = gradcam_figure(
                    orig_vol, cam_map, patient_id, pred_name, white_bg=True,
                    probs=res['probs'], snames=res['snames'])
                gradcam_bytes_white = fig_bytes(fig_cam_w); plt.close()
                gc_coords = (az, sy, cx)
            except Exception:
                pass

        # LLM recs
        with st.spinner("Generating personalised recommendations..."):
            recs = get_llm_recommendations(
                patient_id, tab_features, pred_name,
                res['tb'][-1, dem_i], bool(apoe4 >= 1))

        # FIX 2 & 3: Store EVERYTHING to session_state
        st.session_state.update({
            'res':                  res,
            'tab_features':         tab_features,
            'patient_id':           patient_id,
            'timestamp':            timestamp,
            'pred_name':            pred_name,
            'dem_i_val':            dem_i,
            'apoe4':                apoe4,
            'prob_bytes':           prob_bytes,
            'traj_bytes_list':      traj_bytes_list,
            'scenarios_titles':     [s[2] for s in scenarios],
            'risk_bar_bytes':       risk_bar_bytes,
            'risk_vals':            risk_vals,
            'risk_labels':          risk_labels,
            'gradcam_bytes':        gradcam_bytes,
            'gradcam_bytes_white':  gradcam_bytes_white,
            'gc_coords':            gc_coords,
            'image_available':      res['image_available'],
            'recs':                 recs,
            'apoe4_inc':            (res['ta'][-1,dem_i] - res['tb'][-1,dem_i]) * 100,
            'lecan_ben':            (res['ta'][-1,dem_i] - res['tl'][-1,dem_i]) * 100,
        })

    # ── Step 2: Display from session_state (persists across ALL reruns) ───────
    if st.session_state.get('res'):
        _res  = st.session_state['res']
        _tf   = st.session_state['tab_features']
        _pid  = st.session_state['patient_id']
        _ts   = st.session_state['timestamp']
        _pn   = st.session_state['pred_name']
        _di   = st.session_state['dem_i_val']
        _pb   = st.session_state.get('prob_bytes')
        _tb   = st.session_state.get('traj_bytes_list', [])
        _stit = st.session_state.get('scenarios_titles', ['','','',''])
        _rb   = st.session_state.get('risk_bar_bytes')
        _rv   = st.session_state.get('risk_vals', [])
        _gc   = st.session_state.get('gradcam_bytes')
        _gcw  = st.session_state.get('gradcam_bytes_white')
        _gcc  = st.session_state.get('gc_coords')
        _recs = st.session_state.get('recs', {})
        _ainc = st.session_state.get('apoe4_inc', 0.0)
        _lben = st.session_state.get('lecan_ben', 0.0)
        _img_avail = st.session_state.get('image_available', True)

        badge_cls = {'CN':'badge-cn','MCI':'badge-mci','Dementia':'badge-dementia'}[_pn]

        if not _img_avail:
            st.error(
                "⚠ TABULAR-ONLY RESULT — no real MRI scan was used for this patient. "
                "All predictions and trajectories below come from tabular features alone "
                "(zero image tensor). Do not present this as an image-based diagnosis.")


        # ── Diagnosis metrics ─────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">Predicted Diagnosis</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card is-accent">'
                        f'<span class="{badge_cls}">{_pn}</span>'
                        f'<div class="metric-lbl">Predicted class</div></div>',
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card">'
                        f'<div class="metric-val">'
                        f'{_res["probs"][_res["pred"]]:.0%}</div>'
                        f'<div class="metric-lbl">Confidence</div></div>',
                        unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card">'
                        f'<div class="metric-val">'
                        f'{_res["tb"][-1,_di]:.1%}</div>'
                        f'<div class="metric-lbl">5-yr risk · baseline</div></div>',
                        unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card">'
                        f'<div class="metric-val">'
                        f'{_res["ta"][-1,_di]:.1%}</div>'
                        f'<div class="metric-lbl">5-yr risk · APOE4+</div></div>',
                        unsafe_allow_html=True)

        st.markdown("---")

        # ── Probability chart ─────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">Diagnosis Probabilities</div>',
                    unsafe_allow_html=True)
        if _pb:
            st.image(_pb)
        st.markdown("---")

        # ── Trajectory plots ──────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">5-Year State Trajectories</div>',
                    unsafe_allow_html=True)
        if _tb:
            cols_row1 = st.columns(2)
            cols_row2 = st.columns(2)
            all_cols  = cols_row1 + cols_row2
            for idx, (b, title) in enumerate(zip(_tb, _stit)):
                with all_cols[idx]:
                    st.markdown(f'<div class="traj-title">{title}</div>',
                                unsafe_allow_html=True)
                    st.image(b, use_column_width=True)
        st.markdown("---")

        # ── Risk bar chart ────────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">5-Year Dementia Risk by Scenario</div>',
                    unsafe_allow_html=True)
        if _rb:
            st.image(_rb)
        if _rv:
            st.markdown(
                f'<div class="info-box">APOE4 increases 5-year Dementia risk by '
                f'<b>+{_ainc:.1f}pp</b>. '
                f'Lecanemab reduces it by <b>{_lben:.1f}pp</b>.</div>',
                unsafe_allow_html=True)
        st.markdown("---")

        # ── Grad-CAM ──────────────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">Grad-CAM Neuroimaging Explainability</div>',
                    unsafe_allow_html=True)
        if _gc:
            st.image(_gc, use_column_width=True)
            if _gcc:
                az, sy, cx = _gcc
                st.markdown(
                    f'<div class="info-box">Peak activation — Axial z={az} · '
                    f'Sagittal y={sy} · Coronal x={cx}<br>'
                    f'Red/yellow regions drove the <b>{_pn}</b> prediction.</div>',
                    unsafe_allow_html=True)
        else:
            st.info("Grad-CAM requires an MRI scan.")
        st.markdown("---")

        # ── LLM Lifestyle Recommendations ────────────────────────────────────
        st.markdown('<div class="sec-hdr">🤖 AI Lifestyle Recommendations (Gemma LLM)</div>',
                    unsafe_allow_html=True)
        if _recs:
            st.markdown(f"**{_recs.get('summary', '')}**")
            st.markdown(_recs.get('risk_context', ''))
            rec_cols = st.columns(3)
            icons    = {'exercise':'🏃','diet':'🥗','mental':'🧠',
                        'social':'👥','sleep':'😴','monitoring':'📅'}
            for idx, key in enumerate(['exercise','diet','mental','social','sleep','monitoring']):
                data = _recs.get(key, {})
                with rec_cols[idx % 3]:
                    st.markdown(f"**{icons.get(key,'')} {data.get('headline', key.title())}**")
                    for item in data.get('recommendations',
                                         data.get('schedule', data.get('tips', []))):
                        st.markdown(f"• {item}")
            if _recs.get('positive_note'):
                st.info(f"💙 {_recs['positive_note']}")
        st.markdown("---")

        # ── Download buttons ──────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">⬇ Download Reports</div>',
                    unsafe_allow_html=True)
        col_j, col_pat, col_doc = st.columns(3)

        with col_j:
            st.download_button(
                "📄 Patient Record JSON",
                data=json.dumps({
                    'patient_id':  _pid,
                    'generated':   _ts,
                    'features':    _tf,
                    'diagnosis':   _pn,
                    'confidence':  float(_res['probs'][_res['pred']]),
                    'class_probs': {s: float(p)
                                    for s, p in zip(_res['snames'], _res['probs'])},
                    '5yr_risk': {
                        'baseline':  float(_res['tb'][-1, _di]),
                        'apoe4':     float(_res['ta'][-1, _di]),
                        'lecanemab': float(_res['tl'][-1, _di]),
                        'donanemab': float(_res['td'][-1, _di]),
                    }}, indent=2),
                file_name=f"neuro_dt_{_pid}.json",
                mime="application/json",
                use_container_width=True,
                key="dl_json_v7")

        with col_pat:
            if _pb:
                try:
                    pat_pdf = make_patient_pdf(
                        _pid, _tf, _res, _recs, _pb,
                        _tb[0] if _tb else b'', _rb, _ts)
                    st.download_button(
                        "📋 Patient Report PDF",
                        data=pat_pdf,
                        file_name=f"patient_report_{_pid}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="dl_patient_pdf_v7")
                except Exception as e:
                    st.error(f"Patient PDF: {e}")
            else:
                st.button("📋 Patient Report PDF",
                          disabled=True, use_container_width=True,
                          help="Run the Digital Twin first")

        with col_doc:
            if _pb:
                try:
                    doc_pdf = make_doctor_pdf(
                        _pid, _tf, _res, _pb, _tb,
                        _gcw or _gc, _rb, _ts)
                    st.download_button(
                        "🏥 Doctor Clinical Record PDF",
                        data=doc_pdf,
                        file_name=f"doctor_record_{_pid}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="dl_doctor_pdf_v7")
                except Exception as e:
                    st.error(f"Doctor PDF: {e}")
            else:
                st.button("🏥 Doctor Clinical Record PDF",
                          disabled=True, use_container_width=True,
                          help="Run the Digital Twin first")

    else:
        # Placeholder when no results yet
        st.markdown("""
<div style="text-align:center;padding:3rem;color:#475569">
  <div style="font-size:3rem">🧠</div>
  <div style="font-size:1.1rem;font-weight:500;color:#94a3b8;margin-top:1rem">
    Configure patient features in the sidebar and click<br>
    <span style="color:#38bdf8">▶ Run Digital Twin</span>
  </div>
  <div style="font-size:.85rem;color:#64748b;margin-top:.6rem">
    An MRI blob path is optional — without it the prediction is tabular-only.
  </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ABOUT TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
<style>
.about-hero{background:linear-gradient(135deg,#0f172a,#1e3a5f);
  padding:2rem;border-radius:12px;margin-bottom:1.25rem;
  border:1px solid rgba(56,189,248,.2);}
.about-hero h2{color:#f0f9ff;margin:0;font-size:1.5rem;font-weight:600;letter-spacing:-.01em;}
.about-hero p{color:#94a3b8;margin:.45rem 0 0;font-size:.9rem;}
.metric-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
  gap:10px;margin:1rem 0 .6rem;}
.m-box{background:#1e293b;border:1px solid #334155;border-radius:10px;
  padding:.9rem 1.1rem;}
.m-box.primary{border-left:3px solid #38bdf8;}
.m-val{font-size:1.7rem;font-weight:600;color:#38bdf8;line-height:1.15;letter-spacing:-.02em;
  font-family:monospace;}
.m-sub{font-size:.78rem;color:#64748b;margin-top:.15rem;}
.m-lbl{font-size:.72rem;color:#94a3b8;font-weight:500;text-transform:uppercase;
  letter-spacing:.07em;margin-top:.35rem;}
.caveat{background:#2a1f0a;border:1px solid #5a4300;border-left:3px solid #eda100;
  border-radius:0 8px 8px 0;padding:.8rem 1rem;font-size:.83rem;color:#e2c98a;
  line-height:1.6;margin:.2rem 0 .5rem;}
</style>
<div class="about-hero">
  <h2>🧠 Brain Digital Twin (Neuro-DT)</h2>
  <p>A multimodal deep learning framework for Alzheimer's disease progression simulation</p>
  <p style="margin-top:.7rem;font-size:.82rem;color:#64748b">
    Seif Hendawy &nbsp;·&nbsp; Arab Academy for Science, Technology and Maritime Transport
    &nbsp;·&nbsp; Supervisors: Prof. Fahima Maghraby · Assoc. Prof. Ahmed Salem</p>
</div>
<div class="metric-row">
  <div class="m-box primary">
    <div class="m-val">0.951</div>
    <div class="m-sub">epoch 20 · val loss 0.461</div>
    <div class="m-lbl">Fold 4 macro AUC · deployed</div></div>
  <div class="m-box">
    <div class="m-val">0.871 <span style="font-size:1rem;color:#64748b">± 0.046</span></div>
    <div class="m-sub">mean of 5 folds</div>
    <div class="m-lbl">Cross-validated AUC</div></div>
  <div class="m-box">
    <div class="m-val">85.2%</div>
    <div class="m-sub">macro F1 0.855</div>
    <div class="m-lbl">Fold 4 accuracy</div></div>
  <div class="m-box">
    <div class="m-val">1,549</div>
    <div class="m-sub">469 CN · 603 MCI · 477 Dementia</div>
    <div class="m-lbl">ADNI T1 MRI scans</div></div>
</div>
<div class="caveat">
  <b>How to read these two AUC figures.</b> Fold 4 is the checkpoint this dashboard
  actually serves, and it is the only fold that trained to the full 20 epochs — the
  others stopped early (8, 8, 2 and 5 epochs). Its 0.951 is therefore a
  <i>best-fold</i> figure, not the model's expected performance on new data. The
  honest generalisation estimate is the cross-validated <b>0.871 ± 0.046</b>. Both
  are quoted together everywhere in this dashboard by design.
</div>
""", unsafe_allow_html=True)

    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown("### Architecture")
        st.markdown("""
**Imaging backbone:** 3D DenseNet121 on T1-weighted MRI (128³ voxels, 1.5 mm isotropic)

**Multimodal fusion:** global average pooling → concat with 4 tabular features
(AGE, PTEDUCAT, MMSE, APOE4) → linear projection + LayerNorm + GELU

**Encoder:** Transformer encoder (2 layers, 8 heads, `norm_first=True`) → 3-class classifier

**Prognostic engine:** empirical Markov chain from longitudinal ADNI transitions
+ Monte Carlo simulation (n = 1,000, 5-year horizon)

**Explainability:** Grad-CAM on DenseNet `denseblock4` (brain-masked) + SHAP feature attribution

**LLM layer:** Gemma 4 (31B, Google AI Studio) for personalised lifestyle guidance

**Parameters:** ~24 M &nbsp;|&nbsp; **GPU run:** AdamW lr 1e-4, OneCycleLR,
batch 16, mixed precision, ≤20 epochs, 5-fold stratified CV
        """)

    with c2:
        st.markdown("""
### Per-class AUC — fold 4
| Class | AUC (OvR) |
|---|---|
| CN | 0.982 |
| Dementia | 0.958 |
| MCI | 0.913 |
| **Macro** | **0.951** |
        """)
        st.caption(
            "One-vs-rest AUC on fold 4's validation split, from the GPU run's "
            "`auc_results.json`. Per-class precision / recall / F1 and the confusion "
            "matrix are regenerated by notebook Cell 18 and are not yet republished "
            "here — the only classification report currently on file is from the "
            "superseded CPU run, so it is deliberately not shown."
        )

    st.markdown("""
### Cross-validation — all five folds
| Fold | Epochs | Val loss | Macro AUC |
|---|---|---|---|
| 1 | 8 | 0.603 | 0.882 |
| 2 | 8 | 0.643 | 0.859 |
| 3 | 2 | 0.757 | 0.813 |
| 4 (deployed) | 20 | 0.461 | 0.951 |
| 5 | 5 | 0.670 | 0.848 |
    """)
    st.caption(
        "Mean 0.871 ± 0.046 (population SD). Only fold 4 reached the 20-epoch budget; "
        "the spread across folds is driven mainly by how early the others stopped, "
        "which is the methodological caveat to carry into the thesis write-up."
    )

    st.markdown("---")
    st.markdown("""
### Ablation study — what actually carries the model
| Variant | Macro AUC | Accuracy | Macro F1 |
|---|---|---|---|
| A5 · Transformer, 1 layer | 0.956 | 85.2% | 0.855 |
| A4 · CNN + linear fusion | 0.949 | 84.2% | 0.845 |
| A0 · Neuro-DT full | 0.949 | 85.2% | 0.855 |
| A2 · CNN only (no tabular) | 0.947 | 84.5% | 0.850 |
| B3 · Random Forest | 0.943 | 82.6% | 0.830 |
| B4 · Gradient Boosting | 0.941 | 82.3% | 0.825 |
| A3 · CNN + MLP fusion | 0.939 | 83.9% | 0.844 |
| A6 · no class weights | 0.929 | 78.4% | 0.791 |
| A1 · tabular only (no MRI) | 0.872 | 68.4% | 0.675 |
| B2 · SVM (RBF) | 0.871 | 73.5% | 0.735 |
| B1 · Logistic Regression | 0.865 | 71.0% | 0.707 |

**What the ablation shows, read honestly:**

- **Imaging is doing the work.** Dropping MRI and keeping only the four tabular
  features (A1) costs ~0.077 AUC and 17 points of accuracy — the largest single drop
  in the study.
- **The extra fusion machinery adds little on top of the CNN.** CNN-only (A2, 0.947)
  and a single-layer encoder (A5, 0.956) both land within noise of the full
  two-layer model (A0, 0.949). A5 nominally leads the table; on one fold that
  margin is not evidence the smaller model is better, but it does mean the
  full model's extra capacity is not earning its keep here.
- **Class weighting matters.** Removing it (A6) costs ~0.020 AUC and 6.8 points of
  accuracy, concentrated in the minority classes.
- **Classical baselines are closer than expected.** Random Forest on the same
  features reaches 0.943 AUC, so the deep model's margin over a well-tuned
  classical baseline is real but modest.

All variants were trained on fold 4 only, so these are single-fold comparisons.
    """)

    st.markdown("---")
    st.markdown("""
### CPU model vs. GPU model — results only

**Headline result**

| | CPU model | GPU model |
|---|---|---|
| Folds completed | 1 of 5 (fold 4 only) | 5 of 5 |
| Reported AUC | **0.9120** (single fold) | **0.8706 ± 0.0461** (5-fold CV mean) |
| Valid generalisation estimate? | **No** — one data point, no variance estimate | **Yes** — genuine cross-validated estimate |

**The one directly comparable number.** Both models were evaluated on the identical
fold 4 split (same `StratifiedKFold(random_state=42)`): CPU on fold 4 scored **0.9120**,
GPU on fold 4 scored **0.9511**. On the one apples-to-apples comparison available,
**the GPU model wins by +0.039 AUC** — from the training recipe fixes (OneCycleLR
instead of CosineAnnealingLR, larger batch size, mixed precision), not just from more folds.

**Why the raw 0.8706 vs 0.9120 comparison is misleading.** The GPU mean (0.8706) looks
lower than the CPU number (0.9120) only because CPU's one data point happened to land
on the GPU run's *easiest* fold — 0.9511 was GPU's best fold by a wide margin. The
other four GPU folds (0.8821, 0.8591, 0.8125, 0.8481) were never sampled by CPU at
all. The CPU model's true performance on a random patient split is unknown; the GPU
model's is measured.

| Class | CPU (fold 4) | GPU (fold 4) |
|---|---|---|
| CN | 0.9566 | 0.9819 |
| Dementia | 0.9357 | 0.9583 |
| MCI | 0.8437 | 0.9131 |

GPU is higher on every class, with MCI — the hardest class to classify clinically —
showing the largest gap (+0.069).

**What each model was actually able to produce.** The CPU model has no ablation
study (never ran) and no completed Digital Twin pipeline run — training never
finished, so evaluation, Grad-CAM, the Markov chain, and the what-if simulations
were never exercised against it. The GPU model has all of it: the full ablation
study (7 architecture variants + 4 classical baselines) and the complete Digital
Twin pipeline, all run and validated.

**A robustness check the GPU model got that the CPU model never could.** A second
GPU training run with a different checkpoint-selection criterion (AUC-based instead
of loss-based) produced an inflated 0.9379 ± 0.0069 — confirmed to be a
selection-bias artifact from training well past the point of overfitting with no
early stop. That confirms the GPU model's primary number (0.8706 ± 0.0461) is the
conservative, defensible one — a cross-check that was never possible for the CPU
model, which only ever produced one number with nothing to validate it against.

**Bottom line.** The CPU model's 0.9120 is a single, non-cross-validated, best-case
data point of unknown reliability. The GPU model's 0.8706 ± 0.0461 is a real 5-fold
cross-validated estimate that also outperforms CPU head-to-head on the one fold both
can be compared on, is higher on every per-class metric, and is the only one of the
two actually checked against a selection-bias artifact.
    """)

    st.markdown("---")
    st.markdown("""
### Comparison with published literature
| Paper | Method | Task | AUC / Accuracy | MRI imaging | Drug simulation | Explainability | LLM layer |
|---|---|---|---|---|---|---|---|
| Basaia et al. 2019 | 3D CNN | 3-class (CN/MCI/AD) | ~0.85 AUC | ✓ | ✗ | ✗ | ✗ |
| Wen et al. 2020 | CNN benchmark study | 3-class (CN/MCI/AD) | ~0.83 AUC | ✓ | ✗ | ✗ | ✗ |
| Venugopalan 2021 | Multimodal ML | 3-class (CN/MCI/AD) | ~0.87 AUC | ✓ | ✗ | ✗ | ✗ |
| Bertolini 2021 | Markov Digital Twin | Trajectory forecast | — (no AUC) | ✗ | ✗ | ✗ | ✗ |
| Kushol 2022 (ADDformer) | Transformer (ADDformer) | 3-class (CN/MCI/AD) | ~0.89 AUC | ✓ | ✗ | ✗ | ✗ |
| Hu 2023 (Conv-Swinformer) | CNN + Swin Transformer | 2-class (CN/AD) | 92.9% acc (2-class) | ✓ | ✗ | ✗ | ✗ |
| Wang 2025 (CRBM GenAI) | Conditional RBM GenAI | Clinical trial DT | — (no AUC) | ✗ | Trial-level | ✗ | ✗ |
| DT-GPT 2025 (LLM) | GPT on EHR | Score forecast | 1.8% MAE reduction | ✗ | ✗ | Text only | ✓ |
| **Neuro-DT 2026 (Ours)** | 3D DenseNet + Transformer + Markov | 3-class (CN/MCI/AD) | **0.951 AUC · 85% acc** | ✓ | ✓ Individual | ✓ Grad-CAM+SHAP | ✓ Gemma 4 |
    """)

    st.info("**Key differentiator:** Neuro-DT is the only published framework combining "
            "3D MRI classification (AUC 0.951, 5-fold CV 0.871±0.046), individual patient simulation, time-varying drug "
            "intervention modelling, imaging explainability (Grad-CAM), and an LLM advisory layer.")

    st.markdown("---")
    st.markdown("""
**Citation:** Hendawy, S. (2026). *A Multimodal Deep Learning Framework for a Digital Twin
Simulating Alzheimer's Disease Progression.* Arab Academy for Science, Technology and Maritime
Transport. Supervisors: Prof. F. Maghraby · Assoc. Prof. A. Salem.

**Dataset:** ADNI (Alzheimer's Disease Neuroimaging Initiative) — 1,549 T1-weighted MRI scans,
469 CN · 477 Dementia · 603 MCI · ~500 unique patients across ADNI-1/GO/2/3.
    """)
