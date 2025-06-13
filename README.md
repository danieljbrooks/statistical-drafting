# statisticaldrafting

MTG draft assistant trained on human draft data from 18+ sets.

## Quickstart

Run the draft assistant in your browser using colab [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danieljbrooks/statistical-drafting/blob/main/notebooks/colab_draft_assistant.ipynb)

[Video Demo](https://www.youtube.com/watch?v=RnHF_pV1Zhw)

## Current features

- Models for sets released since 2021 (training data from 17lands datasets)
- Provides a recommended pick order based on the current collection (of previous picks)
- Recommendations for pack-1 pick-1 pick order is similar to GIH winrate from 17lands
- Synergy adjustment accounts for color, archetype, speculation, and splash potential
- Models are portable, with a training time of <20 minutes/dataset and inference time of <1ms/pick

## Running Flask Server

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
