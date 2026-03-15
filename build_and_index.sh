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
SIF="$HOME/vibescholar-gpu.sif"

export SINGULARITY_TMPDIR="$HOME/.singularity/tmp"
export SINGULARITY_CACHEDIR="$HOME/.singularity/cache"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR" "$DATA_DIR"

# Step 1: Build container if it doesn't exist
if [ ! -f "$SIF" ]; then
    echo "=== Building container from definition ==="
    singularity build --fakeroot "$SIF" "$WORKDIR/vibescholar.def"
    echo "=== Container built ==="
else
    echo "=== Container already exists at $SIF ==="
fi

# Step 2: Verify GPU support
echo "=== GPU info ==="
singularity exec --nv "$SIF" python -c "
import onnxruntime as ort
print('ONNX Runtime version:', ort.__version__)
print('Available providers:', ort.get_available_providers())
"
singularity exec --nv "$SIF" nvidia-smi --query-gpu=name,memory.total --format=csv

# Step 3: Index the corpus
echo "=== Starting indexing ==="
cd "$WORKDIR"
singularity exec --nv "$SIF" python -u index_corpus.py \
    "$CORPUS" --data-dir "$DATA_DIR" --cuda --device-ids 0 --workers 16
echo "=== Indexing complete ==="

echo "=== Output files ==="
ls -lh "$DATA_DIR"/
