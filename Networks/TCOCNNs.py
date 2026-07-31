"""Pooling-based TCOCNN variant implemented with PyTorch."""

from typing import Any, Optional, Sequence

try:
    from ._torch_tcocnn import TCOCNNBase
except ImportError:
    from _torch_tcocnn import TCOCNNBase


class TCOCNNsClass(TCOCNNBase):
    """TCOCNN variant using max pooling and global max pooling."""

    architecture = "pooled"
    l2_reg = 0.0

    def __init__(
        self,
        input_size: Sequence[int],
        output_size: int,
        regression: bool = True,
        optim_params: Optional[dict[str, Any]] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(
            input_size,
            output_size,
            regression,
            optim_params,
            device,
        )


# Backward compatibility for code that imported the old class name from this file.
TCOCNNClass = TCOCNNsClass
