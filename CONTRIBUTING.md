# Contributing to Neural Network Foundations from Scratch

Thank you for your interest in contributing! This repository is dedicated to educational, clean, first-principles implementations of deep learning foundations.

## How Can You Contribute?

1. **Typo & Explanation Fixes:** Improving clarity in markdown cells, documentation, or mathematical derivations.
2. **Exercise Solutions:** Adding implementations for extra exercises (e.g., adding ReLU/GELU activations, Adam optimizer, or cross-entropy variants).
3. **Visualization Improvements:** Enhancing plots, activation histograms, or Graphviz renderings.
4. **Code Quality:** Improving readability and adherence to clean Python / PyTorch coding idioms without obscuring the pedagogical intent.

## Guidelines

- **Pedagogy First:** The goal of this code is education and clarity, not raw micro-optimization. Code should be straightforward to trace and understand.
- **Maintain Independence:** Avoid introducing heavy third-party dependencies beyond standard scientific Python (`torch`, `numpy`, `matplotlib`, `graphviz`, `scipy`).
- **Verify Parity:** If modifying mathematical operations or manual gradient calculations, verify that manual gradients match PyTorch autograd (`torch.allclose` with `maxdiff < 1e-7`).

## Submitting Changes

1. Fork the repository.
2. Create a descriptive feature branch (`git checkout -b feature/improved-viz`).
3. Commit your changes with clear messages (`git commit -m 'Add GELU activation to micrograd'`).
4. Push to your branch and open a Pull Request.
