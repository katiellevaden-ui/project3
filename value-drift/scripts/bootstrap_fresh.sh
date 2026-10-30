#!/usr/bin/env bash
# Fallback bootstrap for a pod when the archived environment freezes under runs/ are not
# available (bootstrap_stage2.sh needs runs/training-environment.freeze.txt etc.).
# Builds the same two environments from the committed pins instead:
#   /workspace/venv       training (torch 2.14 cu130, transformers 5.17, bitsandbytes, FLA)
#   /workspace/venv-vllm  inference (vllm 0.29.0, transformers 5.17)
# and downloads the pinned Qwen3.5-9B. The causal-conv1d kernel is built when
# --with-causal-conv1d is given (optional speed-up; without it transformers falls back to
# its torch implementation, which is slower but numerically equivalent).
#   bash scripts/bootstrap_fresh.sh [--with-causal-conv1d] [--skip-download]
set -euo pipefail
with_cc=0; skip_download=0
for argument in "$@"; do
  case "$argument" in
    --with-causal-conv1d) with_cc=1 ;;
    --skip-download) skip_download=1 ;;
    *) echo "usage: $0 [--with-causal-conv1d] [--skip-download]" >&2; exit 2 ;;
  esac
done
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$project_root"
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/pip-cache}"
train_env="${STAGE2_TRAIN_ENV:-/workspace/venv}"
inference_env="${STAGE2_INFERENCE_ENV:-/workspace/venv-vllm}"
python_bin="${STAGE2_PYTHON:-python3}"
record_dir="${STAGE2_RECORD_DIR:-$project_root/runs/fresh-bootstrap}"
mkdir -p "$record_dir"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Linux x86_64 required' >&2; exit 2; }
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 required (apt install python3.12 python3.12-venv; STAGE2_PYTHON=python3.12)"'
command -v g++ >/dev/null || { echo 'g++ missing: apt-get install -y build-essential' >&2; exit 2; }

# Host driver must expose the CUDA 13 driver API (R580+). Checked before any install.
"$python_bin" - "$record_dir" <<'PY'
import ctypes,json,sys
from pathlib import Path
driver=ctypes.CDLL('libcuda.so.1'); assert driver.cuInit(0)==0, 'cuInit failed'
version=ctypes.c_int(); driver.cuDriverGetVersion(ctypes.byref(version))
if version.value<13000: raise SystemExit(f'CUDA 13 driver API required; host reports {version.value}')
Path(sys.argv[1],'driver-readiness.json').write_text(json.dumps({'driver_api_version':version.value})+'\n')
print('driver api',version.value)
PY

if [[ ! -x "$train_env/bin/python" ]]; then "$python_bin" -m venv "$train_env"; fi
if [[ ! -x "$inference_env/bin/python" ]]; then "$python_bin" -m venv "$inference_env"; fi
"$train_env/bin/python" -m pip install --upgrade pip wheel setuptools
"$train_env/bin/python" -m pip install -r requirements-gpu.txt
"$train_env/bin/python" -m pip install flash-linear-attention==0.5.2 \
  nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88 nvidia-cuda-cccl==13.0.85 ninja==1.13.2
"$inference_env/bin/python" -m pip install --upgrade pip
"$inference_env/bin/python" -m pip install -r requirements-inference.txt ninja==1.13.2

# Same compiler/runtime layout the recorded build used (FlashInfer JIT looks under CUDA_HOME).
export CUDA_HOME="$train_env/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$train_env/bin:$CUDA_HOME/bin:$PATH"
"$train_env/bin/python" - "$CUDA_HOME" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])
assert (root/'lib/libcudart.so.13').is_file(), 'cu13 runtime not found under the training venv'
for link,target in [(root/'lib/libcudart.so','libcudart.so.13'),(root/'lib64','lib')]:
    if not (link.exists() or link.is_symlink()): link.symlink_to(target)
PY

if [[ "$with_cc" == 1 ]]; then
  export CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS:-4}" TORCH_CUDA_ARCH_LIST="9.0"
  "$train_env/bin/python" -m pip install --no-build-isolation --no-deps causal-conv1d==1.7.0 \
    || echo 'causal-conv1d build failed; continuing without it (slower kernels)' >&2
fi

"$train_env/bin/python" -m pip freeze > "$record_dir/training-environment.freeze.txt"
"$inference_env/bin/python" -m pip freeze > "$record_dir/inference-environment.freeze.txt"

# Real CUDA checks in both environments.
"$train_env/bin/python" - "$record_dir" <<'PY'
import json,sys,torch,transformers
from pathlib import Path
assert torch.cuda.is_available(), 'training env cannot initialise CUDA'
x=torch.ones((16,16),device='cuda',dtype=torch.bfloat16,requires_grad=True); loss=(x@x).float().mean(); loss.backward()
assert loss.item()==16 and torch.isfinite(x.grad).all().item()
try:
    import causal_conv1d; cc=True
except Exception: cc=False
try:
    import fla; fla_ok=True
except Exception: fla_ok=False
device=torch.cuda.get_device_properties(0)
result={'torch':torch.__version__,'transformers':transformers.__version__,'causal_conv1d':cc,'fla':fla_ok,
        'device':device.name,'memory_gb':device.total_memory/1e9}
Path(sys.argv[1],'training-imports.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))
if 'cu130' not in torch.__version__: print('WARNING: torch is not a cu130 build; expected 2.14.0+cu130', file=sys.stderr)
if device.total_memory/1e9 < 140: print('WARNING: less than 140 GB device memory; DPO peaks at 135 GB', file=sys.stderr)
PY
PATH="$inference_env/bin:$PATH" "$inference_env/bin/python" - "$record_dir" <<'PY'
import json,shutil,sys,torch,vllm
from pathlib import Path
assert torch.cuda.is_available(), 'inference env cannot initialise CUDA'
result={'torch':torch.__version__,'vllm':vllm.__version__,'ninja':shutil.which('ninja')}
Path(sys.argv[1],'inference-imports.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))
PY

if [[ "$skip_download" == 0 ]]; then
  "$train_env/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
path=snapshot_download('Qwen/Qwen3.5-9B',revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a')
print('MODEL_DOWNLOAD_COMPLETE',path)
PY
fi
echo "Fresh bootstrap complete; records in $record_dir. Next: bash scripts/bootstrap_bench.sh $train_env $(dirname "$project_root")"
