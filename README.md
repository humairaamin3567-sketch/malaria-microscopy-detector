# Malaria Microscopy Detector

Detect malaria parasites from microscopy images using deep learning (PyTorch + transfer learning).

## Quickstart

1. Create virtual env and install:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Prepare dataset: place raw images under `data/raw/{infected,uninfected}`. See `data/README_DATA.md` for formatting.
3. Convert raw -> processed (simple script included) then train:
   ```bash
   python src/train.py --data_dir data/processed --model efficientnet_b0 --epochs 12 --batch-size 32
   ```
4. Evaluate:
   ```bash
   python src/evaluate.py --ckpt experiments/baseline/checkpoint.pth --data_dir data/processed
   ```
5. Inference single image:
   ```bash
   python src/infer.py --ckpt experiments/baseline/checkpoint.pth --image-path examples/sample.jpg
   ```

## Structure

- `src/` : code (data, model, training, inference, utils)
- `notebooks/` : EDA, training demo, Grad-CAM
- `data/` : instructions and processed dataset layout (do not commit raw data)
- `experiments/` : saved checkpoints and reports (gitignored)
- `Dockerfile`, `requirements.txt`, `.github/` for CI

## License
MIT
