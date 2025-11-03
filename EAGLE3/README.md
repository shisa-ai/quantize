Use SpecForge for training EAGLE models:
- https://github.com/sgl-project/SpecForge
- https://docs.sglang.ai/SpecForge/


## Install
````
# source
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge
pip install -v .

# pypi
pip install specforge
````

## Create Dataset
python prepare_data.py

## AMD MI300 Notes
- ROCm on MI300 enforces a 1024 thread-per-block limit, so Triton kernels that request 2048 threads will fail with `triton.runtime.errors.OutOfResources`.
- The SpecForge log-softmax kernel now caps `num_warps` based on the active device; make sure your environment is using the patched `SpecForge/specforge/core/loss.py` (and reinstallations pick it up under `site-packages/specforge/core/loss.py`).
- After the patch, you can sanity-check the kernel with `python -m specforge.core.loss` or a small `LogSoftmaxLoss` call on `cuda` before launching `run-chotto-202501013.8xMI300.sh`.
