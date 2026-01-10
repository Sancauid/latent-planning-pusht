# Latent Action Planning with JEPA World Models

This repository contains a PyTorch implementation of a **Joint-Embedding Predictive Architecture (JEPA)** applied to the **Push-T** robotics dataset.

It implements State-of-the-Art (SOTA) recommendations from recent World Modeling research (Terver et al., 2025), focusing on stable long-horizon prediction and latent-space planning without pixel reconstruction.

## 🚀 Key Features

*   **Visual Backbone:** Frozen **DINOv2** (ViT-Small/14) for high-level semantic embedding.
*   **Architecture:** Transformer Predictor with **AdaLN (Adaptive Layer Norm)** conditioning for efficient action modulation.
*   **Multi-Modal:** Fuses Visual Embeddings with **Proprioception** (Robot State) for precise physics modeling.
*   **Training Objective:** Multi-step auto-regressive rollout (Horizon=6) to enforce temporal stability.
*   **Planner:** **Cross-Entropy Method (CEM)** for iterative trajectory optimization in latent space.

## 📊 Performance

The model was trained on an NVIDIA A100 cluster.

*   **Final Validation Loss:** `0.29` (MSE in Latent Space)
*   **Convergence:** Achieved in 40 Epochs.
*   **Optimization:** Uses `torch.compile`, mixed-precision (BF16), and full VRAM-resident caching for maximum throughput.

## 🛠️ Usage

### 1. Precompute Embeddings
Extracts DINOv2 features and proprioception states to Zarr for high-speed training.
```bash
python 2_precompute_sota.py
```

### 2. Train World Model
Trains the AdaLN-conditioned predictor with multi-step consistency loss.
```bash
python 3_train.py
```

### 3. Run Planner (Inference)
Loads the trained model and executes the CEM planner to solve the Push-T task.
```bash
python 4_simulate.py
```

## 📄 References

This implementation is based on findings from:
1.  **"What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?"** (Terver et al., 2025)
2.  **"Joint-Embedding Predictive Architecture"** (LeCun, 2022)