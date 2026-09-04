"""TCOCNN V3 with configurable-depth residual convolution blocks."""

from typing import Any, Optional, Sequence

from skopt.space import Categorical, Integer, Real

try:
    from ._torch_tcocnn import TCOCNNBase
except ImportError:
    from _torch_tcocnn import TCOCNNBase


class TCOCNNv3Class(TCOCNNBase):
    """Deeper TCOCNN variant with optional residual connections.

    V3 keeps stride one for every convolution.  Each feature block contains a
    searchable number of convolutions, an optional residual shortcut, and one
    explicit max-pooling operation.  Global average pooling keeps the dense
    head compact even when the convolutional backbone becomes deeper.
    """

    architecture = "v3"
    l2_reg = 0.0001
    network_parameter_names = TCOCNNBase.network_parameter_names.union(
        {"convs_per_block", "channel_growth", "residual"}
    )
    integer_hyperparameter_names = TCOCNNBase.integer_hyperparameter_names.union(
        {"convs_per_block", "channel_growth"}
    )

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

    def default_search_space(self) -> list[Any]:
        """Return the expanded V3 architecture and training search space."""
        return [
            Categorical([8, 12, 16, 24, 32, 48, 64], name="n_filter"),
            Integer(2, 6, name="section_depth"),
            Categorical([3, 5, 7, 9, 11, 15, 21], name="kernel"),
            Categorical([2, 3, 4], name="stride"),
            Integer(1, 4, name="convs_per_block"),
            Categorical([0, 8, 16, 24, 32], name="channel_growth"),
            Categorical([False, True], name="residual"),
            Categorical([16, 32, 64, 128, 256, 512], name="num_neurons"),
            Real(0.0, 0.5, name="drop_out"),
            Real(
                5e-5,
                1e-2,
                prior="log-uniform",
                name="initial_learning_rate",
            ),
            Categorical([16, 32, 64, 128], name="batch_size"),
        ]


TCOCNNv3 = TCOCNNv3Class
TCOCNNClass = TCOCNNv3Class
