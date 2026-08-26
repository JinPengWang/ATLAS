"""PyTorch GPU acceleration engine for massively parallel population evaluation."""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np


def is_torch_available() -> bool:
    """Check if PyTorch is installed and importable."""
    try:
        import torch

        return True
    except ImportError:
        return False


def get_default_device() -> str:
    """Return best available compute device ('cuda', 'mps', or 'cpu')."""
    if not is_torch_available():
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TorchBenchmarkProblem:
    """GPU-accelerated vectorized benchmark problem evaluator.

    Accepts a 2-D PyTorch tensor of shape (N, dim) and computes the objective
    values in a single vectorized kernel on GPU/CPU.

    Supported function names:
        'sphere', 'rastrigin', 'rosenbrock', 'ackley', 'griewank'
    """

    def __init__(
        self,
        function_name: str = "sphere",
        dim: int = 30,
        device: Optional[str] = None,
    ) -> None:
        if not is_torch_available():
            raise ImportError(
                "PyTorch is required for TorchBenchmarkProblem. Install with `pip install torch`."
            )
        import torch

        self.function_name = function_name.lower()
        self.dim = dim
        self.device = torch.device(device if device is not None else get_default_device())

    def evaluate_tensor(self, X: Any) -> Any:
        """Evaluate a batch tensor of candidate solutions.

        Args:
            X: torch.Tensor of shape (N, dim).

        Returns:
            torch.Tensor of shape (N,) containing objective values.
        """
        import torch

        X = X.to(self.device)

        if self.function_name == "sphere":
            return torch.sum(X**2, dim=-1)

        elif self.function_name == "rastrigin":
            return torch.sum(X**2 - 10.0 * torch.cos(2.0 * np.pi * X) + 10.0, dim=-1)

        elif self.function_name == "rosenbrock":
            x_curr = X[..., :-1]
            x_next = X[..., 1:]
            return torch.sum(100.0 * (x_next - x_curr**2) ** 2 + (x_curr - 1.0) ** 2, dim=-1)

        elif self.function_name == "ackley":
            d = X.shape[-1]
            term1 = -20.0 * torch.exp(-0.2 * torch.sqrt(torch.sum(X**2, dim=-1) / d))
            term2 = -torch.exp(torch.sum(torch.cos(2.0 * np.pi * X), dim=-1) / d)
            return term1 + term2 + 20.0 + np.e

        elif self.function_name == "griewank":
            d = X.shape[-1]
            indices = torch.arange(1, d + 1, device=self.device, dtype=X.dtype)
            sum_term = torch.sum(X**2, dim=-1) / 4000.0
            prod_term = torch.prod(torch.cos(X / torch.sqrt(indices)), dim=-1)
            return sum_term - prod_term + 1.0

        else:
            raise ValueError(f"Unknown GPU benchmark function: '{self.function_name}'")

    def evaluate_numpy(self, X_np: np.ndarray) -> np.ndarray:
        """Evaluate a numpy batch and return numpy array."""
        import torch

        X_t = torch.as_tensor(X_np, dtype=torch.float32, device=self.device)
        res_t = self.evaluate_tensor(X_t)
        return res_t.detach().cpu().numpy()
