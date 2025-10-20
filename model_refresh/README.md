# Model Refresh Automation

This automates the process of detecting and downloading new datasets for model training from 17lands. Efforts have been made to make the minimal number of requests.

## Usage

```bash
cd model_refresh
python ./refresh_models.py
```

The script automatically:
1. Checks for new MTG sets and updated draft data
2. Downloads new data when detected
3. Retrains models for updated datasets

## Files

- `refresh_models.py` - Main automation script
- `get_latest_set.py` - Fetches latest set information from 17lands
- `data_tracker.json` - Tracks update dates to minimize requests
- `requirements.txt` - Additional dependencies
