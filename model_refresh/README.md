# Model Refresh Automation

This automates the process of detecting and downloading new datasets for model training from 17lands. Efforts have been made to make the minimal number of requests.

## Training Workflow

### 1. Email Notification (Automated)

GitHub Actions runs nightly to check for new data on 17lands.com. When new data is detected, you'll receive an email notification.

**What gets trained:**
- Premier Draft models (preferred)
- PickTwo Draft models (fallback when Premier is not available)
- Traditional Draft models are **no longer trained**

### 2. Train Models (Manual)

When you receive the email notification:

```bash
cd model_refresh
python ./refresh_models.py
```

This script will:
- Download new draft data if needed
- Train the appropriate model (Premier or PickTwo)
- Save the trained model to `../data/models/`
- Convert to ONNX format in `../data/onnx/`

**Training typically takes 10-20 minutes per model.**

### 3. Deploy to Website (Manual)

After training completes:

1. Copy the new ONNX model files from `data/onnx/` to the website repo
2. Commit and push the changes to the website repo

```bash
# Example:
cp ../data/onnx/*.onnx ../../statistical-drafting-website/data/onnx/
cd ../../statistical-drafting-website
git add data/onnx/*.onnx
git commit -m "Update models for [SET_NAME]"
git push
```

## Files

- `refresh_models.py` - Main automation script (downloads data and trains models)
- `ci_check_updates.py` - Automated check script (used by GitHub Actions)
- `get_latest_set.py` - Fetches latest set information from 17lands
- `data_tracker.json` - Tracks update dates to minimize requests
- `requirements.txt` - Additional dependencies
