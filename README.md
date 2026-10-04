# Environment Setup

### Clone repo and initialize submodules

```bash
git clone <REPO_URL>
cd ood_detect
git submodule update --init --recursive
```

### Create and activate Conda environment

```bash
conda create -n ood_detect python=3.10 -y
conda activate ood_detect
```

### Install PyTorch

Choose one depending on your hardware:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Install Anomalib and dependencies

Install the local submodule in editable mode with OpenVINO support:

```bash
pip install -e "libs/anomalib[openvino]"
```

### Verify installation

```bash
python -c "import torch, openvino, anomalib; print('Setup successful! CUDA available:', torch.cuda.is_available())"
```