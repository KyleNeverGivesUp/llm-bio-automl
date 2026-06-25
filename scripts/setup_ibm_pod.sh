#!/bin/bash
# Isolated install of IBM SmallMoleculeMultiView (biomed-multi-view) on the DSMLP pod.
# Uses its OWN venv so it never touches the unimol torch (2.4.1) in ~/.local.
# Linux builds fast_transformers more reliably than macOS.
set -e
cd ~
export PYTHONNOUSERSITE=1   # CRITICAL: ignore ~/.local (unimol's torch 2.4.1) so ibm_venv stays isolated

echo "=== [1/5] create isolated venv ==="
python3 -m venv ~/ibm_venv
~/ibm_venv/bin/pip install -q --upgrade pip

echo "=== [2/5] install biomed-multi-view (pulls its own torch) ==="
~/ibm_venv/bin/pip install -q "git+https://github.com/BiomedSciAI/biomed-multi-view.git"

TV=$(~/ibm_venv/bin/python -c "import torch; print(torch.__version__)")
BASE=${TV%%+*}
SUFFIX=$(echo "$TV" | grep -o '+cu[0-9]*' || echo '+cpu')
echo "    torch in ibm_venv = $TV  -> pyg index torch-${BASE}${SUFFIX}"

echo "=== [3/5] install torch_scatter / torch_sparse (matched) ==="
~/ibm_venv/bin/pip install -q torch_scatter torch_sparse \
  -f "https://data.pyg.org/whl/torch-${BASE}${SUFFIX}.html"

echo "=== [4/5] install pytorch-fast-transformers (text view) ==="
~/ibm_venv/bin/pip install -q --no-build-isolation pytorch-fast-transformers \
  || ~/ibm_venv/bin/pip install -q pytorch-fast-transformers

echo "=== [5/5] import test ==="
~/ibm_venv/bin/python -c "from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel; print('IMPORT OK')"
echo "DONE. Next: ~/ibm_venv/bin/python ibm_embed.py --smoke   (then full)"
