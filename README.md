# Perceived Stress Score

Based on the dataset: <https://www.kaggle.com/datasets/harishyadav0506/impact-of-social-media-on-life>

We predict the `Perceived_Stress_Score`, binned into 4 classes from 7 categorical features using Categorical Naive Bayes.

## Setup

Requires Python 3.14+.

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Using uv

```bash
uv sync
```

## Run

```bash
python stratifydata.py # split, tune, alpha, report F1
```

