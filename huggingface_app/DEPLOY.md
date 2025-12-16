# Hugging Face Spaces Deployment Guide
=====================================

This guide explains how to deploy the EO AI Demo to Hugging Face Spaces.

## Prerequisites

1. A Hugging Face account (https://huggingface.co/join)
2. Git LFS installed (`git lfs install`)
3. Trained models (optional - placeholders work for demo)

---

## Method 1: Web Upload

1. **Create a new Space**
   - Go to https://huggingface.co/new-space
   - Choose a name: `eo-ai-demo`
   - Select **Gradio** as the SDK
   - Choose visibility (Public/Private)
   - Click "Create Space"

2. **Upload files**
   - Click "Files" tab
   - Upload all files from `huggingface_app/`:
     - `app.py`
     - `interface.py`
     - `requirements.txt`
     - `README.md`
     - `utils/` folder (all .py files)
     - `models/` folder (if you have trained models)
     - `samples/` folder (sample images)

3. **Wait for build**
   - The Space will automatically build and deploy
   - Check "Logs" tab for build status
   - Access your demo at: `https://huggingface.co/spaces/yourusername/eo-ai-demo`

---

## Method 2: Git Push

```bash
# Clone your Space (empty at first)
git clone https://huggingface.co/spaces/yourusername/eo-ai-demo
cd eo-ai-demo

# Copy files from huggingface_app
cp -r /path/to/octo/huggingface_app/* .

# Add files
git add .
git commit -m "Initial deployment"

# Push to Hugging Face
git push
```

---

## Method 3: Hugging Face CLI

```bash
# Install CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create and upload Space
python -c "
from huggingface_hub import HfApi
api = HfApi()

# Create Space
api.create_repo(
    repo_id='yourusername/eo-ai-demo',
    repo_type='space',
    space_sdk='gradio',
    private=False
)

# Upload all files
api.upload_folder(
    folder_path='./huggingface_app',
    repo_id='yourusername/eo-ai-demo',
    repo_type='space'
)
"
```

---

## Adding Trained Models

### Option A: Upload with Space

Put your model files in the `models/` directory and push:

```
huggingface_app/
├── models/
│   ├── lulc_classifier.pt
│   ├── lst_estimator.pt
│   └── plastic_detector.pt
```

⚠️ **Note**: HF Spaces has a 10GB limit for free tier.

### Option B: Host on Hugging Face Hub

1. Create a model repository:
   ```bash
   huggingface-cli repo create eo-lulc-classifier --type model
   ```

2. Upload your model:
   ```bash
   huggingface-cli upload yourusername/eo-lulc-classifier ./model.pt
   ```

3. Load in your app:
   ```python
   from huggingface_hub import hf_hub_download
   
   model_path = hf_hub_download(
       repo_id="yourusername/eo-lulc-classifier",
       filename="model.pt"
   )
   model.load_state_dict(torch.load(model_path))
   ```

---

## Adding Sample Images

Create sample images and place in `samples/` folder:

```
samples/
├── uhi/
│   └── thermal_sample.png
├── lulc/
│   ├── before_2020.png
│   └── after_2023.png
├── plastic/
│   └── water_body.png
└── xai/
    └── satellite_rgb.png
```

For large datasets, use Hugging Face Datasets:

```python
from datasets import load_dataset

dataset = load_dataset("yourusername/eo-samples")
```

---

## Environment Variables

For sensitive data (API keys), use Secrets:

1. Go to Space Settings → Repository secrets
2. Add secrets:
   - `SENTINELHUB_API_KEY`
   - `GEE_SERVICE_ACCOUNT`

3. Access in code:
   ```python
   import os
   api_key = os.environ.get("SENTINELHUB_API_KEY")
   ```

---

## Hardware Selection

For GPU inference:

1. Go to Space Settings
2. Select "Hardware": GPU (T4, A10G, A100)
3. Note: GPU requires paid subscription

---

## Troubleshooting

### Build Fails

- Check `Logs` tab for errors
- Verify `requirements.txt` has correct versions
- Test locally first: `python app.py`

### Out of Memory

- Reduce model size or use quantization
- Use GPU hardware tier
- Implement batch processing

### Slow Loading

- Lazy load models on first request
- Use `@functools.lru_cache` for model loading
- Host large models on Hub separately

---

## Example Deployment Script

```python
#!/usr/bin/env python
"""Deploy to Hugging Face Spaces"""

import subprocess
import os

SPACE_ID = "yourusername/eo-ai-demo"
APP_DIR = "./huggingface_app"

def deploy():
    # Create fresh clone
    subprocess.run(["git", "clone", f"https://huggingface.co/spaces/{SPACE_ID}", "temp_deploy"])
    
    # Copy files
    subprocess.run(["cp", "-r", f"{APP_DIR}/.", "temp_deploy/"])
    
    # Push
    os.chdir("temp_deploy")
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", "Update deployment"])
    subprocess.run(["git", "push"])
    
    # Cleanup
    os.chdir("..")
    subprocess.run(["rm", "-rf", "temp_deploy"])
    
    print(f"✓ Deployed to https://huggingface.co/spaces/{SPACE_ID}")

if __name__ == "__main__":
    deploy()
```

---

## Links

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://gradio.app/docs/)
- [Your Space Dashboard](https://huggingface.co/settings/spaces)
