#!/bin/bash
#SBATCH --job-name=vibescholar-index
#SBATCH --output=vibescholar-index_%j.out
#SBATCH --error=vibescholar-index_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a40:1

set -euo pipefail

WORKDIR="$HOME/vibescholar-mcp"
CORPUS="$HOME/cvprscrape"
DATA_DIR="$HOME/vibescholar-data"
SIF="$HOME/vibescholar.sif"
VENV="$HOME/vibescholar-venv"

export SINGULARITY_TMPDIR="$HOME/.singularity/tmp"
export SINGULARITY_CACHEDIR="$HOME/.singularity/cache"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR" "$DATA_DIR"

# Step 1: Pull base Python image if it doesn't exist
if [ ! -f "$SIF" ]; then
    echo "=== Pulling Python 3.11 image ==="
    singularity pull "$SIF" docker://python:3.11-slim
    echo "=== Image pulled ==="
else
    echo "=== Image already exists at $SIF ==="
fi

# Step 2: Create a persistent venv with GPU-enabled deps
if [ ! -f "$VENV/bin/activate" ]; then
    echo "=== Creating virtual environment ==="
    singularity exec --nv "$SIF" python -m venv "$VENV"
    # Install onnxruntime-gpu instead of onnxruntime (fastembed pulls onnxruntime by default)
    # Install all deps, then force-reinstall onnxruntime-gpu LAST
    # (fastembed pulls onnxruntime CPU as a dep and would overwrite GPU)
    singularity exec --nv "$SIF" "$VENV/bin/pip" install --no-cache-dir \
        'faiss-cpu>=1.8.0' \
        'fastembed>=0.5.0' \
        'pypdf>=5.0.0' \
        'FlashRank>=0.2.0' \
        numpy
    singularity exec --nv "$SIF" "$VENV/bin/pip" install --no-cache-dir \
        --force-reinstall --no-deps onnxruntime-gpu
    echo "=== Dependencies installed ==="
else
    echo "=== Venv already exists at $VENV ==="
fi

# Step 3: Show GPU info
echo "=== GPU info ==="
singularity exec --nv "$SIF" "$VENV/bin/python" -c "
import onnxruntime as ort
print('ONNX Runtime version:', ort.__version__)
print('Available providers:', ort.get_available_providers())
"

# Step 4: Index the corpus with 3 GPUs
echo "=== Starting indexing ==="
cd "$WORKDIR"
singularity exec --nv "$SIF" "$VENV/bin/python" -u index_corpus.py \
    "$CORPUS" --data-dir "$DATA_DIR" --cuda --device-ids 0
echo "=== Indexing complete ==="

echo "=== Output files ==="
ls -lh "$DATA_DIR"/
