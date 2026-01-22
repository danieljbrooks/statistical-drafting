# statisticaldrafting
MTG drafting model trained on human draft data from 20+ sets.

This repo contains the python model training code. 

## Quickstart

Website frontend http://statisticaldrafting.com

Code for website fronted https://github.com/danieljbrooks/statistical-drafting-website

Older frontend [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danieljbrooks/statistical-drafting/blob/main/notebooks/colab_draft_assistant.ipynb) [Video Demo](https://www.youtube.com/watch?v=RnHF_pV1Zhw)

## Summary
- Models for sets released since 2021 (training data from 17lands datasets)
- Provides a recommended pick order based on the current collection (of previous picks)
- Recommendations for pack-1 pick-1 pick order is similar to GIH winrate from 17lands 
- Synergy adjustment accounts for color, archetype, speculation, and splash potential
- Models are portable, with a training time of <20 minutes/dataset and inference time of <1ms/pick

## Model Training & Updates

For training new models when new sets are released, see the [model_refresh](./model_refresh) directory.

GitHub Actions automatically checks for new data nightly. When updates are detected, train new models with `python model_refresh/refresh_models.py`, then manually deploy ONNX models to the website repository.
