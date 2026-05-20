# ADC Project

> [!NOTE]  
> This project was conducted as part of Stanford Unviersity's CS231N Deep Learning for Computer Vision course.

## How to Use

Setup using these commands:
```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Usage:


## Structure


## Background

### ViT for SEM defect classification.

Forward pass:

1. Split image into patches and embed each.

2. Prepend a learnable [CLS] token.

3. Add learnable positional embeddings.

4. Pass through `depth` transformer blocks (last N use V-V attention).

5. Apply final LayerNorm.

6. Classify from the [CLS] token via a linear head.