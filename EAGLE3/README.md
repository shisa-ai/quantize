# EAGLE3 Training with SpecForge

Train EAGLE3 speculative decoding models using SpecForge:
- https://github.com/sgl-project/SpecForge
- https://docs.sglang.ai/SpecForge/

## Quick Start (CUDA/NVIDIA)

```bash
# Create environment
mamba create -n quantize python=3.12
mamba activate quantize

# Install PyTorch
pip install torch torchvision

# Install SpecForge
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge
pip install -e .
```

## ROCm/MI300X Installation

### Prerequisites
- ROCm 6.4+ installed
- AMD Instinct MI300X GPU(s)

### Step 1: Create Environment

```bash
mamba create -n quantize python=3.12
mamba activate quantize
```

### Step 2: Install PyTorch for ROCm 6.4

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4 -U
```

Verify installation:
```bash
python -c "import torch; print('torch:', torch.__version__); print('hip:', torch.version.hip); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected output:
```
torch: 2.9.1+rocm6.4
hip: 6.4.43484-123eb5128
device: AMD Instinct MI300X VF
```

### Step 3: Clone/Update SpecForge

```bash
cd /data/quantize/EAGLE3
git clone https://github.com/sgl-project/SpecForge.git
# or if already cloned:
cd SpecForge && git pull
```

### Step 4: Install Dependencies from ROCm Requirements

The `requirements-rocm.txt` has been updated to avoid pinning PyTorch versions (since we pre-installed it):

```bash
cd SpecForge
pip install -r requirements-rocm.txt
```

This installs:
- sglang[all]>=0.5.4
- transformers>=4.57.1
- accelerate, wandb, datasets, etc.

### Step 5: Install SpecForge in Editable Mode

```bash
pip install -e . --no-deps
```

Using `--no-deps` avoids re-reading `requirements.txt` which has pinned CUDA PyTorch versions.

### Step 6: Install AMD AITER (Required for sglang on ROCm)

AITER (AI Tensor Engine for ROCm) provides optimized kernels for AMD GPUs:

```bash
cd /data/quantize
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
python setup.py develop
```

This will JIT-compile modules on first use (~10s per module).

### Step 7: Install vLLM for ROCm (Required by sglang)

sglang has a hard dependency on vLLM for some code paths. Upstream vLLM now has full ROCm support:

```bash
cd /data/quantize
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Install ROCm-specific dependencies
pip install -r requirements/rocm.txt

# Build for MI300X (gfx942)
export PYTORCH_ROCM_ARCH="gfx942"
python setup.py develop
```

**Note:** Build takes 5-10 minutes. See `docs/getting_started/installation/gpu.rocm.inc.md` for full details.

**Known working commit:** `1b0482b9d` (2025-12-07). If you encounter issues with HEAD, try:
```bash
git checkout 1b0482b9d
```

**TODO:** Confirm minimal requirements for ROCm build. vLLM has `rocm.txt`, `rocm-build.txt`, and `rocm-test.txt` - may only need `rocm.txt` for runtime.

**Alternative architectures:**
- MI200/MI250: `export PYTORCH_ROCM_ARCH="gfx90a"`
- Multiple: `export PYTORCH_ROCM_ARCH="gfx90a;gfx942"`

### Step 8: Build sgl-kernel for ROCm (Required by sglang)

The pip-installed `sgl-kernel` is CUDA-only. For ROCm, build from source using `setup_rocm.py`:

```bash
cd /data/quantize/sglang/sgl-kernel

# Uninstall CUDA version if installed
pip uninstall sgl-kernel -y

# Build for MI300X
export PYTORCH_ROCM_ARCH="gfx942"
python setup_rocm.py develop
```

This compiles HIP kernels for allreduce, MoE, EAGLE speculative decoding, etc.

### Step 9: Verify MI300X Patch (SpecForge loss.py)

The SpecForge log-softmax kernel includes an MI300X fix (halves `num_warps` on ROCm to respect the 1024 thread-per-block limit). Verify with:

```bash
cd /data/quantize/EAGLE3
./patch-specforge-mi300x.sh
```

Expected output:
```
Editable install detected at: /data/quantize/EAGLE3/SpecForge
Repo file:      /data/quantize/EAGLE3/SpecForge/specforge/core/loss.py
Installed file: /data/quantize/EAGLE3/SpecForge/specforge/core/loss.py
Editable install: repo and installed files are the same.
MI300X num_warps fix is present.
Patch verification complete.
```

### Step 10: Test the Loss Kernel

Test the Triton log-softmax kernel on MI300X:

```bash
SGLANG_USE_AITER=1 python -c "
import sys
sys.path.insert(0, '/data/quantize/EAGLE3/SpecForge/specforge/core')
import torch
from loss import LogSoftmaxLoss, _compute_loss

device = 'cuda'
B, T, V = 2, 512, 16000
print(f'Testing on {torch.cuda.get_device_name(0)}')

logits = torch.randn(B, T, V, device=device, requires_grad=True)
logits2 = logits.clone().detach().requires_grad_(True)
target = torch.randn(B, T, V, device=device)
position_mask = torch.ones((B, T, 1), dtype=torch.bool, device=device)

output1 = LogSoftmaxLoss.apply(logits, target, position_mask)
output2 = _compute_loss(logits2, target, position_mask)
torch.testing.assert_close(output1, output2, rtol=1e-4, atol=1e-4)
print('Forward: PASSED')

output1.backward()
output2.backward()
torch.testing.assert_close(logits.grad, logits2.grad, rtol=1e-4, atol=1e-4)
print('Backward: PASSED')
print('MI300X LogSoftmaxLoss: SUCCESS')
"
```

### Step 11: Verify Full SpecForge Import

Confirm all components work together:

```bash
cd /data/quantize/EAGLE3/SpecForge
SGLANG_USE_AITER=1 python -c "
import torch
print('torch:', torch.__version__)
print('hip:', torch.version.hip)
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')

import sgl_kernel
print('sgl_kernel:', sgl_kernel)

import specforge
print('specforge:', specforge)

from specforge import OfflineEagle3Model
print('OfflineEagle3Model:', OfflineEagle3Model)
print('SUCCESS: All imports work!')
"
```

## Integration Tests

After installation, run these tests to verify SpecForge works correctly on MI300X:

### Loss Kernel Test (~20s)

Tests the Triton log-softmax kernel with various batch/sequence/vocab sizes. This validates the MI300X `num_warps` fix:

```bash
cd /data/quantize/EAGLE3/SpecForge
SGLANG_USE_AITER=1 python -m pytest tests/test_utils/test_loss.py -v
```

Expected output:
```
tests/test_utils/test_loss.py::TestLogSoftmaxLoss::test_loss PASSED
tests/test_utils/test_loss.py::TestLogSoftmaxLoss::test_ttt_loss_accumulation PASSED
======================== 2 passed, 5 warnings in 20.44s ========================
```

### Model Loading Test (~20s)

Tests loading an EAGLE3 draft model from HuggingFace:

```bash
cd /data/quantize/EAGLE3/SpecForge
SGLANG_USE_AITER=1 python -m pytest tests/test_modeling/test_auto_model.py -v
```

Expected output:
```
tests/test_modeling/test_auto_model.py::TestAutoModelForCausalLM::test_automodel PASSED
======================== 1 passed, 7 warnings in 20.84s ========================
```

### Full Test Suite (Optional)

Run all non-training tests:

```bash
cd /data/quantize/EAGLE3/SpecForge
SGLANG_USE_AITER=1 python -m pytest tests/ -v --ignore=tests/test_scripts/
```

**Note:** `tests/test_scripts/` contains full training tests that require model downloads and take longer to run.

## Training Scripts

### Environment Variable

All scripts use `mamba run -n $ENV` where `ENV` defaults to `quantize`:

```bash
# Override if needed:
ENV=my_env ./run-shisa-v2.1-unphi4-14b.1xMI300.sh
```

### 1xMI300 Training

```bash
./run-shisa-v2.1-unphi4-14b.1xMI300.sh
```

### 8xMI300X Setup & Verification

If setting up on a new 8xMI300X node, follow the same Steps 1-11 as above. The installation is identical.

**Quick verification on 8xMI300X:**

```bash
# 1. Verify all 8 GPUs visible
rocm-smi --showid

# 2. Test imports and GPU count
SGLANG_USE_AITER=1 python -c "
import torch
print(f'GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  {i}: {torch.cuda.get_device_name(i)}')
from specforge import OfflineEagle3Model
print('SUCCESS')
"

# 3. Loss kernel test (uses single GPU)
cd /data/quantize/EAGLE3/SpecForge
SGLANG_USE_AITER=1 python -m pytest tests/test_utils/test_loss.py -v

# 4. Multi-GPU NCCL/RCCL communication test
torchrun --standalone --nproc_per_node 8 -m torch.distributed.run --help  # verify torchrun works
```

**8xMI300X training script:** `run-chotto-202501013.8xMI300.sh`

Or edit `run-shisa-v2.1-unphi4-14b.1xMI300.sh` and uncomment the 8xMI300 sections.

| Setting | 1xMI300 | 8xMI300 |
|---------|---------|---------|
| `NUM_GPUS` | 1 | 8 |
| `BATCH_SIZE` | 8 | 16 |
| `LEARNING_RATE` | 1e-4 | 3e-4 |
| `--draft-global-batch-size` | (default) | 32 |
| `--draft-micro-batch-size` | (default) | 4 |

## Create Dataset

```bash
python prepare_data.py
```

## AMD MI300X Technical Notes

### Thread Limits
- ROCm on MI300 enforces a 1024 thread-per-block limit
- Triton kernels requesting 2048 threads will fail with `triton.runtime.errors.OutOfResources`
- The SpecForge log-softmax kernel caps `num_warps` based on the active device (lines 42-44 in `loss.py`)

### Attention Backend
- MI300X: Use `--attention-backend sdpa` (flex_attention may not work)
- H100: Can use `--attention-backend flex_attention`

### Environment Variables for ROCm
```bash
export SGLANG_USE_AITER=1           # Use AMD AITER optimizations
export PYTORCH_ROCM_ARCH="gfx942"   # MI300X architecture
```

## Installed Versions (as of 2025-12-07)

| Package | Version |
|---------|---------|
| torch | 2.9.1+rocm6.4 |
| torchvision | 0.24.1+rocm6.4 |
| pytorch-triton-rocm | 3.5.1 |
| specforge | 0.1.1 (editable) |
| sglang | 0.5.6 |
| sgl-kernel | 0.3.18.post3 (editable, built from source) |
| aiter | 0.1.7 (editable) |
| vllm | 0.11.2.dev635+rocm641 (built from source) |
| transformers | 4.57.1 |

## Troubleshooting

### "vllm is required when SGLANG_USE_AITER is set to False"
Set `SGLANG_USE_AITER=1` and ensure AMD AITER is installed.

### "aiter is required when SGLANG_USE_AITER is set to True"
Install AMD AITER from source (Step 6).

### "No module named 'vllm'"
Install vLLM from ROCm fork (Step 7).

### triton.runtime.errors.OutOfResources
The MI300X patch is not applied. Run `./patch-specforge-mi300x.sh`.

### "No module named 'sgl_kernel.common_ops'"
The pip-installed sgl-kernel is CUDA-only. Build from source (Step 8):
```bash
cd /data/quantize/sglang/sgl-kernel
pip uninstall sgl-kernel -y
export PYTORCH_ROCM_ARCH="gfx942"
python setup_rocm.py develop
```

## References

- [SpecForge GitHub](https://github.com/sgl-project/SpecForge)
- [SpecForge Docs](https://docs.sglang.ai/SpecForge/)
- [AMD AITER](https://github.com/ROCm/aiter)
- [vLLM GitHub](https://github.com/vllm-project/vllm) (has full ROCm support)
- [vLLM ROCm Install Docs](https://github.com/vllm-project/vllm/blob/main/docs/getting_started/installation/gpu.rocm.inc.md)
- [SGLang ROCm Docs](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/sglang.html)
