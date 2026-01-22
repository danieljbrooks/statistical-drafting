#!/bin/bash
set -e

# Configuration
WEBSITE_REPO="../statistical-drafting-website"

# Step 1: Train models
echo "=== Training models ==="
cd model_refresh
python refresh_models.py
cd ..

# Step 2: Deploy to website
echo "=== Deploying to website ==="
cp data/onnx/*.onnx "$WEBSITE_REPO/data/onnx/"

cd "$WEBSITE_REPO"
git add data/onnx/*.onnx
git commit -m "Update ONNX models"
git push

echo "=== Done ==="
