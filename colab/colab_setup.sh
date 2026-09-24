#!/bin/bash
set -e
# Paste me in the FIRST Colab cell (with !):
#   !bash colab/colab_setup.sh
# Then run the PREFLIGHT cell, then one experiment command. See colab/RUN.md.

cd /content || exit 1

if [ ! -d FairDerm ]; then
  git clone https://github.com/FairMedAI/fairderm-audit FairDerm
fi
cd FairDerm

echo "== installing Python deps =="
pip install -q numpy pandas scikit-learn albumentations \
  opencv-python-headless torch torchvision

echo "== linking data + model weights from Drive (if mounted) =="
DRIVE=/content/drive/MyDrive/fairderm_assets
if [ -d "$DRIVE" ]; then
  mkdir -p models data
  cp -r "$DRIVE/models/." models/ 2>/dev/null || echo "  (no models in Drive; you must upload fairderm_ham_baseline.pth)"
  cp -r "$DRIVE/data/ddi" data/ 2>/dev/null || echo "  (no data/ddi in Drive; cannot run experiments)"
else
  echo "WARNING: /content/drive not found. Mount Drive:"
  echo "  from google.colab import drive; drive.mount('/content/drive')"
  echo "  and place assets under MyDrive/fairderm_assets/ as described in colab/RUN.md"
fi

echo "== repo contents =="
ls
echo "SETUP DONE"