# LoFT-LLM

Code for **LoFT-LLM: Low-Frequency Time-Series Forecasting with Large Language Models**.
[Paper](https://arxiv.org/abs/2512.20002v3) · [中文使用说明](README_zh.md) · [Method mapping](docs/METHOD.md)

This release retains the original research implementation. The PLFM, iTransformer, attention layers, data loaders, metrics, prompts and original run entry points are unchanged. Configuration helpers supply public model/adapter paths and launch the existing code. A standalone QLoRA script completes the paper's supervised LLM calibration stage.

Use Python 3.10 and a suitable CUDA/PyTorch installation, then install dependencies:

```bash
pip install -r requirements.txt
```

Prepare data in the original format described in [docs/DATA.md](docs/DATA.md). Edit `configs/fundar.json` or `configs/solar.json` to set dataset paths, visible CUDA devices and the Qwen3-8B base/adapter paths. Run from this directory:

```bash
# Stage 1: original PLFM low-frequency training
python run_release.py --config configs/fundar.json --stage plfm
# Stage 2: original residual training with fixed PLFM
python run_release.py --config configs/fundar.json --stage residual
# Stage 3: original prompt/target export, then supervised QLoRA
python run_release.py --config configs/fundar.json --stage sft
python run_release.py --config configs/fundar.json --stage qlora
# Original LLM calibration and evaluation
python run_release.py --config configs/fundar.json --stage infer
```

Replace `fundar` with `solar` for Solar. Add the same `--horizon 3` to each stage to change forecast length. `--dry-run` prints the existing entry point and its arguments without loading models. All original `run*.py` entry points remain available.

The base model is [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), or a configured local copy. Data and trained adapters are supplied or generated separately. QLoRA reads the original `messages` JSON and supervises the assistant's numerical reply and EOS, masking the prompt and padding. Its configurable hyperparameters are implementation defaults, not recovered historical experiment settings.

Only five original experiment Python files were adjusted: restoring the original Qwen financial branch, configuring paths/export checkpoint selection, removing a Solar debug exception, and restoring odd forecast lengths in two IFFT calls. [CHANGES.md](docs/CHANGES.md), [FILE_CHANGES.json](FILE_CHANGES.json) and the included diffs document every modification. The original TSLib README and dependency list are retained as `README_TSLib.md` and `requirements-original.txt`; the MIT [LICENSE](LICENSE) is retained.

```bash
python -m unittest discover -s tests -v
# After exporting SFT examples:
python train_qlora.py --config configs/fundar.json --check-data
```

This preparation targets consistency with the paper's proposed method, without requiring numerical-table reproduction. [Validation coverage](docs/VALIDATION.md) distinguishes source/configuration checks from GPU training and inference.
