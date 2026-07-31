"""Notebook-compatible TCOCNN implemented with PyTorch."""

from typing import Any, Optional, Sequence

try:
    from ._torch_tcocnn import TCOCNNBase
except ImportError:
    from _torch_tcocnn import TCOCNNBase


class TCOCNNClass(TCOCNNBase):
    """Original strided-convolution TCOCNN architecture."""

    architecture = "strided"
    l2_reg = 0.0001

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
