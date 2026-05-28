# ADC Project

> [!NOTE]  
> This project was conducted as part of Stanford Unviersity's CS231N Deep Learning for Computer Vision course.

## How to Use
First, download the Carinthia SEM defect data from [here](https://zenodo.org/records/10715190).
Place into the `./data/` folder, so that it contains the doneloaded .csv and the `\images` folder with the Carinthia JPGs.


Then, setup using these commands:
```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Finally, run using:
```
python3 train.py
```

For the Grad-CAM vizualization, user:
```
python gradcam.py --checkpoint ./checkpoints/best.pt --n 12
```


## Structure
```
./
 ├-- data/
 |     ├-- carinthia.csv
 |     ├-- images/
 ├-- requirements.txt
 ├-- preprocessing.py
 ├-- visualize_tensor.py
 ├-- model.py
 ├-- train.py
 ├-- gradcam.py
 ├-- README.md
```


## Background

### ViT for SEM defect classification.

Forward pass:

1. Split image into patches and embed each.

2. Prepend a learnable [CLS] token.

3. Add learnable positional embeddings.

4. Pass through `depth` transformer blocks (last N use V-V attention).

5. Apply final LayerNorm.

6. Classify from the [CLS] token via a linear head.

### Training loop for ViT-Small on the Carinthia SEM defect dataset.

Pipeline:

1. Load preprocessed + augmented tensors from preprocess.build_tensors()


2. Wrap in TensorDataset / DataLoader

3. Train vit_small with AdamW + cosine LR schedule + label smoothing

4. Log per-epoch train/val loss and accuracy

5. Save best checkpoint by val accuracy