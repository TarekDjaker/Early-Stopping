# Early Stopping for Iterative Learning

This repository provides reference implementations of early stopping strategies for iterative algorithms such as Landweber iterations, conjugate gradients, L2 boosting, truncated SVD and regression trees. The project originates from academic work on implicit regularisation for inverse problems and is designed for experimentation and teaching.

## Features

- Landweber iterations with bias/variance tracking
- Conjugate Gradients and L2 boosting with discrepancy-based stopping rules
- Utilities for truncated SVD and regression trees
- A simulation wrapper to reproduce experiments

## Installation

The algorithms depend on a scientific Python stack. Install the required
packages and the project itself with:

```bash
git clone https://github.com/TarekDjaker/Early-Stopping.git
cd Early-Stopping
pip install -r requirements.txt
pip install -e .
```

## Quick start

Execute the example script to see a minimal Landweber run on synthetic data
(`numpy` is required):

```bash
python example.py
```

The script prints the iteration at which the discrepancy principle triggers early stopping.

## Documentation

Further theoretical background and API details can be found in the [project documentation](https://earlystop.github.io/EarlyStopping/).

## License

This project is released under the [MIT License](LICENSE).

