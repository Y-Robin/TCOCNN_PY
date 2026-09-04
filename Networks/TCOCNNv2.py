"""TCOCNN V2 with paired convolutions, max pooling, and global average pooling."""

from typing import Any, Optional, Sequence

try:
    from ._torch_tcocnn import TCOCNNBase
except ImportError:
    from _torch_tcocnn import TCOCNNBase


class TCOCNNv2Class(TCOCNNBase):
    """V2 architecture sharing the complete notebook-compatible TCOCNN API.

    Every feature block contains two stride-1 convolutions with the same filter
    depth, followed by max pooling. Global average pooling reduces the final
    feature map to one value per channel before the dense regression or
    classification head.

    The existing ``stride`` hyperparameter is retained for API and checkpoint
    compatibility, but in V2 it controls the max-pooling factor. Convolution
    strides are always one.
    """

    architecture = "v2"
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


# Keep the same import convention used by the original architecture wrappers.
TCOCNNv2 = TCOCNNv2Class
TCOCNNClass = TCOCNNv2Class
