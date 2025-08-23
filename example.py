"""Minimal example for the Landweber early stopping algorithm."""

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised in docs
    raise SystemExit(
        "NumPy is required to run this example. Install it with `pip install numpy`."
    ) from exc

from landweber import Landweber


def main():
    """Run a small Landweber demo on synthetic data."""
    rng = np.random.default_rng(0)
    design = rng.normal(size=(100, 20))
    true_signal = rng.normal(size=20)
    noise_level = 0.1
    response = design @ true_signal + noise_level * rng.normal(size=100)

    lw = Landweber(
        design=design,
        response=response,
        learning_rate=0.5,
        true_signal=true_signal,
        true_noise_level=noise_level,
    )
    lw.iterate(number_of_iterations=50)
    stop = lw.get_discrepancy_stop(
        critical_value=noise_level**2 * design.shape[0], max_iteration=50
    )
    print(f"Discrepancy principle stop: {stop} iterations")

if __name__ == "__main__":
    main()
