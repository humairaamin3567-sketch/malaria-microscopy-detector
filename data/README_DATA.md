# Data instructions

Place your raw microscopy images like this (do NOT commit raw data):
```
data/raw/infected/*.png
data/raw/uninfected/*.png
```

The preprocessing script will generate `data/processed/train`, `data/processed/val`, and `data/processed/test` folders with subfolders `infected` and `uninfected` ready for torchvision.datasets.ImageFolder.
