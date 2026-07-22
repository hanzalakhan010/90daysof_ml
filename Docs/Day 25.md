**Day 25 — Resumed after compute barrier (new GPU workstation)**

Retrained the pneumonia CNN from Day 24 locally on a Quadro T2000 (4GB VRAM) instead of Kaggle's free-tier compute.

**Fixes vs Day 24:**
- Day 24 had `data_val_dir == data_dir` (both pointed at `train/`) — "validation" during training was actually re-scoring training data. Fixed: proper train/val split (stratified, 15%) + the official `test/` folder held out untouched until final evaluation.
- Hardcoded `class_weight = {0: 1, 1: 5}` replaced with `compute_class_weight('balanced', ...)` from actual class counts.
- Added back Dropout + EarlyStopping (Day 24 had dropped these from Day 23).

**Setup gotchas (new machine):**
- `pip install tensorflow[and-cuda]` installs CUDA as pip packages, but TF can't `dlopen` them without `LD_LIBRARY_PATH` pointing at `site-packages/nvidia/*/lib` — not obvious from the error message.
- `batch_size=32` OOM'd on the 4GB card (activation memory for 150x150 images across 3 conv layers adds up fast) — dropped to `batch_size=16`, trained fine.
- Kaggle's local dataset auth changed to a single API token (`~/.kaggle/access_token`), not the old `kaggle.json` username+key format.

**Result:**
| Split | Recall | Precision |
|---|---|---|
| Train | 0.981 | 0.993 |
| Val | 0.993 | 0.981 |
| **Test (held-out)** | 0.99 (PNEUMONIA) / **0.45 (NORMAL)** | 0.75 (PNEUMONIA) / 0.97 (NORMAL) |

Test accuracy 79%. Train/val looked near-perfect but real held-out test recall for NORMAL collapsed to 45% — despite the corrected class weights favoring it. Val being a stratified split of the *same* pool as train doesn't guarantee it reflects the differently-sourced test set. Banked as-is; augmentation (rotation/zoom) is the next lever if this needs revisiting.

**Not done (next candidate)**: TFLite export for edge deployment.
