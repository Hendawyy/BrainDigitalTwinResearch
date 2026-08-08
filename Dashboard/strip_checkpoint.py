"""
Shrink a training checkpoint down to what the dashboard actually needs.

The GPU-run checkpoints are ~289 MB because they embed the optimizer, scheduler
and AMP-scaler states so training could resume. Inference needs none of that —
only the weights, the fitted StandardScaler and the label map. Dropping the rest
takes the file from ~289 MB to ~97 MB (a 67% cut), which matters twice over on
the B2 App Service plan: a smaller image pulls inside the container start-time
limit, and torch.load() no longer has to deserialise ~190 MB of optimizer
tensors into a 3.5 GB RAM budget on the first request.

Usage (from the dashboard folder, on the compute instance):

    python strip_checkpoint.py checkpoints/best_model_fold4.pth

Writes alongside the input as <name>_slim.pth and leaves the original in place.
Verify, then swap the slim file in as best_model_fold4.pth before rebuilding.
"""
import os
import sys
from pathlib import Path

import torch

# Everything the dashboard's load_assets() reads, plus the provenance fields that
# let you confirm which run a checkpoint came from.
KEEP = (
    "model_state_dict",   # the weights
    "scaler",             # fitted StandardScaler for the 4 tabular features
    "label_map",          # class-name <-> index mapping
    "fold",
    "epoch",
    "val_loss",
    "val_auc",
    "fold_complete",
    "training_config",
)


def strip(src: Path) -> Path:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise SystemExit(f"{src}: expected a dict checkpoint, got {type(ckpt).__name__}")

    missing = [k for k in ("model_state_dict", "scaler", "label_map") if k not in ckpt]
    if missing:
        raise SystemExit(f"{src}: missing required key(s) {missing} — refusing to write")

    slim = {k: ckpt[k] for k in KEEP if k in ckpt}
    dropped = sorted(set(ckpt) - set(slim))

    dest = src.with_name(src.stem + "_slim" + src.suffix)
    torch.save(slim, dest)

    before = os.path.getsize(src) / 1e6
    after = os.path.getsize(dest) / 1e6
    print(f"source   : {src}  ({before:.1f} MB)")
    print(f"stripped : {dest}  ({after:.1f} MB)")
    print(f"saved    : {before - after:.1f} MB ({100 * (1 - after / before):.0f}%)")
    print(f"dropped  : {', '.join(dropped) if dropped else '(nothing)'}")
    print(f"kept     : {', '.join(sorted(slim))}")
    print(
        f"\nprovenance: fold={slim.get('fold')} epoch={slim.get('epoch')} "
        f"val_auc={slim.get('val_auc')}"
    )
    return dest


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: python {Path(__file__).name} <checkpoint.pth>")
    path = Path(sys.argv[1])
    if not path.exists():
        raise SystemExit(f"{path}: not found")
    strip(path)
