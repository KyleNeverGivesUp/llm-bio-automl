# CheMeleon fine-tune on the A5000 — run steps

Goal: fine-tune the CheMeleon graph model end-to-end on PXR pEC50 (the heavy path the
leaderboard leaders used), produce leak-free 5-fold OOF + 513 test predictions, send
2 CSVs back. **Set 1 is never touched** — training is broad rows only, split by our
calibrated cluster folds.

## 1. Copy 4 files to the GPU box (from this repo)

```bash
# from your local machine (repo root):
scp data/pxr_activity/train.csv \
    data/pxr_activity/test.csv \
    data/pxr_activity/folds_calibrated.json \
    scripts/finetune_cheme.py \
    USER@A5000_HOST:~/cheme/
```

## 2. On the A5000 (SSH in)

```bash
cd ~/cheme
python -m venv venv && source venv/bin/activate
pip install 'chemprop>=2.2.0'

# confirm GPU is visible (should print: cuda True ... A5000)
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. (optional, 2 min) sanity check the plumbing first

```bash
python finetune_cheme.py --max-rows 150 --folds 2 --epochs 5 --accelerator gpu
# -> should finish and write cheme_ft_out/oof_cheme_ft.csv + test_cheme_ft.csv
```

## 4. The real run (~20–40 min on A5000)

```bash
python finetune_cheme.py --epochs 50 --accelerator gpu --out-dir cheme_ft_out
```

CheMeleon auto-downloads its weights (~MB) to `~/.chemprop/` on first run. Output:
```
cheme_ft_out/oof_cheme_ft.csv    # row_id, SMILES, y_true, y_pred  (4139 OOF rows)
cheme_ft_out/test_cheme_ft.csv   # Molecule Name, SMILES, pEC50    (513 test preds)
```

## 5. Send the 2 CSVs back

```bash
scp cheme_ft_out/oof_cheme_ft.csv cheme_ft_out/test_cheme_ft.csv USER@LOCAL:~/...
```
Then I judge them on Set 1 and stack into the ensemble (vs the current ~0.62).

## Notes / knobs
- `--epochs 50` is a good start; try 30–80. More epochs ≠ always better (overfit) — the judge decides.
- For a stronger result, run it 2–3× with different seeds (add `--epochs` runs to separate `--out-dir`s) and send all; I'll average.
- If `--accelerator gpu` errors, use `--accelerator auto`.
- Next step after this works: multi-task fine-tune adding the 8,135 `single_concentration` molecules (bigger lever) — separate script.
