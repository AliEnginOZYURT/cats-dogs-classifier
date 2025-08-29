# Cats vs Dogs Classifier 🐱🐶

A simple and reproducible deep-learning project that classifies **cat vs dog** images with PyTorch.  
The repository contains both a **Jupyter Notebook** (`.ipynb`) for exploration and a **Python script** (`.py`) for straightforward execution.

---

## 📁 Project Structure
```
cats-dogs-classifier/
├── notebooks/
│   └── Cats_Dogs.ipynb
├── src/
│   └── Cats_Dogs.py
├── requirements.txt
└── README.md
```

## 📦 Setup
Python 3.10+ is recommended.
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 🗂️ Data Layout
Place your dataset like this (example):
```
data/
├── train/
│   ├── cat/
│   └── dog/
└── valid/
    ├── cat/
    └── dog/
```
> Any similar folder structure supported by `torchvision.datasets.ImageFolder` will work.

## ▶️ Run
**Notebook:**
```bash
jupyter notebook notebooks/Cats_Dogs.ipynb
```

**Script:**
```bash
python src/Cats_Dogs.py --data_dir data --epochs 10 --batch_size 32
```
> Adjust CLI arguments according to your script; if not implemented, simply run `python src/Cats_Dogs.py`.

## ⚙️ Training Defaults (example)
- Optimizer: Adam / SGD  
- Loss: CrossEntropyLoss  
- Epochs: 10–30 (depending on hardware)  
- Batch size: 32–64  
- Learning rate: 1e-3 to start

## 📊 Results
- **Accuracy:** _XX%_ (fill with your best run)  
- Add training/validation plots or a confusion matrix screenshot here for clarity.

## 🔁 Reproducibility
- Pin versions via `requirements.txt`  
- Set a random seed (e.g., `torch.manual_seed(42)`) for repeatable results

## 📥 Inference (optional)
If your script supports single-image prediction, document it here, e.g.:
```bash
python src/Cats_Dogs.py --predict path/to/image.jpg
```

## 🧾 License
MIT (or another license of your choice).

---
**Author:** Ali Engin Özyurt  
**Purpose:** Portfolio-ready example for CV and internship/job applications.
