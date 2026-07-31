"""Shared PyTorch backend for the notebook-compatible TCOCNN wrappers."""

from __future__ import annotations

import copy
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from scipy.ndimage import uniform_filter1d
from skopt import gp_minimize
from skopt.space import Integer, Real
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class History:
    """Small Keras-like history container kept for notebook compatibility."""

    history: dict[str, list[float]]


class SamePadConv2d(nn.Module):
    """Conv2d with TensorFlow/Keras-compatible dynamic ``padding="same"``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_height, input_width = inputs.shape[-2:]
        output_height = math.ceil(input_height / self.stride[0])
        output_width = math.ceil(input_width / self.stride[1])

        pad_height = max(
            (output_height - 1) * self.stride[0]
            + self.kernel_size[0]
            - input_height,
            0,
        )
        pad_width = max(
            (output_width - 1) * self.stride[1]
            + self.kernel_size[1]
            - input_width,
            0,
        )
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        if pad_height or pad_width:
            inputs = functional.pad(
                inputs,
                (pad_left, pad_right, pad_top, pad_bottom),
            )
        return self.conv(inputs)


class TCOCNNModule(nn.Module):
    """Internal network; public input conversion lives in ``TCOCNNBase``."""

    def __init__(
        self,
        input_size: Sequence[int],
        output_size: int,
        optim_params: dict[str, Any],
        architecture: str,
    ) -> None:
        super().__init__()
        height, width, channels = (int(value) for value in input_size)
        n_filters = int(optim_params["n_filter"])
        first_kernel_size = int(optim_params["kernel"])
        first_stride = int(optim_params["stride"])
        dropout_rate = float(optim_params["drop_out"])
        num_convs = int(optim_params["section_depth"])
        width_fc = int(optim_params["num_neurons"])

        if architecture not in {"strided", "pooled"}:
            raise ValueError(f"Unknown TCOCNN architecture: {architecture}")
        if min(height, width, channels, n_filters, num_convs, width_fc) < 1:
            raise ValueError("Network dimensions must be positive integers.")
        if first_kernel_size < 1 or first_stride < 1:
            raise ValueError("Kernel size and stride must be positive.")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("drop_out must be in the interval [0, 1).")

        feature_layers: list[nn.Module] = []
        if architecture == "strided":
            feature_layers.extend(
                [
                    SamePadConv2d(
                        channels,
                        n_filters,
                        kernel_size=(1, first_kernel_size),
                        stride=(1, first_stride),
                    ),
                    nn.BatchNorm2d(n_filters, momentum=0.01),
                    nn.ReLU(),
                ]
            )
        else:
            feature_layers.extend(
                [
                    SamePadConv2d(
                        channels,
                        n_filters,
                        kernel_size=(1, first_kernel_size),
                    ),
                    nn.MaxPool2d(kernel_size=(1, first_stride)),
                    nn.BatchNorm2d(n_filters, momentum=0.01),
                    nn.ReLU(),
                ]
            )

        feature_layers.extend(
            [
                SamePadConv2d(
                    n_filters,
                    n_filters,
                    kernel_size=(1, first_kernel_size),
                ),
                nn.BatchNorm2d(n_filters, momentum=0.01),
                nn.ReLU(),
            ]
        )

        in_channels = n_filters
        for section_index in range(1, num_convs):
            out_channels = n_filters * (section_index + 1)
            if architecture == "strided":
                feature_layers.extend(
                    [
                        SamePadConv2d(
                            in_channels,
                            out_channels,
                            kernel_size=(1, 2),
                            stride=(1, 2),
                        ),
                        nn.BatchNorm2d(out_channels, momentum=0.01),
                        nn.ReLU(),
                    ]
                )
            else:
                feature_layers.extend(
                    [
                        SamePadConv2d(
                            in_channels,
                            out_channels,
                            kernel_size=(1, 2),
                        ),
                        nn.MaxPool2d(kernel_size=(1, 2)),
                        nn.BatchNorm2d(out_channels, momentum=0.01),
                        nn.ReLU(),
                    ]
                )

            feature_layers.extend(
                [
                    SamePadConv2d(
                        out_channels,
                        out_channels,
                        kernel_size=(1, 2),
                    ),
                    nn.BatchNorm2d(out_channels, momentum=0.01),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*feature_layers)
        self.global_pool: nn.Module
        if architecture == "pooled":
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.global_pool = nn.Identity()

        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            feature_count = int(
                self.global_pool(self.features(dummy)).flatten(start_dim=1).shape[1]
            )

        self.fc1 = nn.Linear(feature_count, width_fc)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(width_fc, int(output_size))
        self._initialize_like_keras()

    def _initialize_like_keras(self) -> None:
        for module in self.modules():
            if isinstance(module, SamePadConv2d):
                nn.init.xavier_uniform_(module.conv.weight)
                if module.conv.bias is not None:
                    nn.init.zeros_(module.conv.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.features(inputs)
        outputs = self.global_pool(outputs)
        outputs = outputs.flatten(start_dim=1)
        outputs = functional.relu(self.fc1(outputs))
        outputs = self.dropout(outputs)
        return self.fc2(outputs)


class TCOCNNBase:
    """Keras-shaped public API implemented with PyTorch.

    The notebooks continue to pass and receive NumPy arrays in NHWC format.
    Tensor conversion, device placement, batching, and NCHW conversion are
    intentionally hidden behind this compatibility layer.
    """

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
        if len(input_size) != 3:
            raise ValueError("input_size must use the Keras-style (height, width, channels) format.")

        self.input_size = tuple(int(value) for value in input_size)
        self.output_size = int(output_size)
        self.regression = bool(regression)
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: Optional[TCOCNNModule] = None
        self.built_flag = False
        self.optim_params = copy.deepcopy(optim_params)
        self.history: Optional[History] = None
        self.trainFlag = False
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler: Optional[torch.optim.lr_scheduler.StepLR] = None
        self.criterion: Optional[nn.Module] = None
        self.initial_learning_rate = 1e-3

        if optim_params is not None:
            self.build_net(optim_params)

    def build_net(self, optim: dict[str, Any]) -> None:
        required = {
            "n_filter",
            "section_depth",
            "kernel",
            "stride",
            "num_neurons",
            "drop_out",
        }
        missing = sorted(required.difference(optim))
        if missing:
            raise KeyError(f"Missing network parameters: {', '.join(missing)}")

        self.optim_params = copy.deepcopy(optim)
        self.model = TCOCNNModule(
            self.input_size,
            self.output_size,
            self.optim_params,
            self.architecture,
        ).to(self.device)
        self.built_flag = True
        self.optimizer = None
        self.lr_scheduler = None
        self.criterion = None

    def _require_model(self) -> TCOCNNModule:
        if not self.built_flag or self.model is None:
            raise RuntimeError("Model is not built. Call build_net(...) first.")
        return self.model

    def _create_optimizer(self, learning_rate: float) -> torch.optim.Optimizer:
        model = self._require_model()
        trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
        if not trainable:
            raise RuntimeError("The model has no trainable parameters.")

        decay = [value for _, value in trainable if value.ndim > 1]
        no_decay = [value for _, value in trainable if value.ndim <= 1]
        parameter_groups: list[dict[str, Any]] = []
        if decay:
            parameter_groups.append(
                {
                    "params": decay,
                    "weight_decay": self.l2_reg if self.architecture == "strided" else 0.0,
                }
            )
        if no_decay:
            parameter_groups.append({"params": no_decay, "weight_decay": 0.0})
        return torch.optim.Adam(
            parameter_groups,
            lr=float(learning_rate),
            eps=1e-7,
        )

    def compile_model(self, initial_learning_rate: float = 1e-3) -> None:
        self._require_model()
        self.initial_learning_rate = float(initial_learning_rate)
        self.optimizer = self._create_optimizer(self.initial_learning_rate)
        self.criterion = nn.MSELoss() if self.regression else nn.CrossEntropyLoss()
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=2,
            gamma=0.9,
        )

    def _input_tensor(self, data: Any) -> torch.Tensor:
        tensor = torch.as_tensor(np.asarray(data), dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(
                "Expected input data with shape (samples, height, width, channels)."
            )

        height, width, channels = self.input_size
        if tuple(tensor.shape[1:]) == (height, width, channels):
            tensor = tensor.permute(0, 3, 1, 2)
        elif tuple(tensor.shape[1:]) != (channels, height, width):
            raise ValueError(
                "Input shape does not match input_size. "
                f"Expected (*, {height}, {width}, {channels}), got {tuple(tensor.shape)}."
            )
        return tensor.contiguous()

    def _target_tensor(self, target: Any) -> torch.Tensor:
        if self.regression:
            tensor = torch.as_tensor(np.asarray(target), dtype=torch.float32)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(1)
            tensor = tensor.reshape(tensor.shape[0], -1)
            if tensor.shape[1] != self.output_size:
                raise ValueError(
                    f"Regression targets need {self.output_size} value(s) per sample."
                )
            return tensor

        tensor = torch.as_tensor(np.asarray(target), dtype=torch.long)
        return tensor.reshape(-1)

    def _loader(
        self,
        data: Any,
        target: Optional[Any] = None,
        batch_size: int = 50,
        shuffle: bool = False,
    ) -> DataLoader:
        inputs = self._input_tensor(data)
        dataset: TensorDataset
        if target is None:
            dataset = TensorDataset(inputs)
        else:
            targets = self._target_tensor(target)
            if inputs.shape[0] != targets.shape[0]:
                raise ValueError("Data and target must contain the same number of samples.")
            dataset = TensorDataset(inputs, targets)

        return DataLoader(
            dataset,
            batch_size=max(1, int(batch_size)),
            shuffle=shuffle,
            pin_memory=self.device.type == "cuda",
        )

    def _evaluate(
        self,
        data: Any,
        target: Any,
        batch_size: int,
    ) -> Tuple[float, float]:
        model = self._require_model()
        if self.criterion is None:
            raise RuntimeError("Model is not compiled. Call compile_model(...) first.")

        model.eval()
        total_loss = 0.0
        total_metric = 0.0
        total_samples = 0
        with torch.no_grad():
            for inputs, targets in self._loader(data, target, batch_size):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                sample_count = inputs.shape[0]
                total_loss += float(loss.item()) * sample_count
                if self.regression:
                    total_metric += float(torch.abs(outputs - targets).mean(dim=1).sum().item())
                else:
                    total_metric += float((outputs.argmax(dim=1) == targets).sum().item())
                total_samples += sample_count

        if total_samples == 0:
            raise ValueError("Cannot evaluate an empty dataset.")
        return total_loss / total_samples, total_metric / total_samples

    def _fit(
        self,
        data: Any,
        target: Any,
        validation_data: Optional[Tuple[Any, Any]],
        epochs: int,
        batch_size: int,
    ) -> History:
        model = self._require_model()
        if self.optimizer is None or self.criterion is None or self.lr_scheduler is None:
            raise RuntimeError("Model is not compiled. Call compile_model(...) first.")

        metric_name = "mae" if self.regression else "accuracy"
        values: dict[str, list[float]] = {"loss": [], metric_name: []}
        if validation_data is not None:
            if not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2:
                raise ValueError("validation_data must be a (data, target) tuple.")
            values["val_loss"] = []
            values[f"val_{metric_name}"] = []

        for _ in range(max(0, int(epochs))):
            model.train()
            total_loss = 0.0
            total_metric = 0.0
            total_samples = 0
            for inputs, targets in self._loader(data, target, batch_size, shuffle=True):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                sample_count = inputs.shape[0]
                total_loss += float(loss.detach().item()) * sample_count
                if self.regression:
                    total_metric += float(
                        torch.abs(outputs.detach() - targets).mean(dim=1).sum().item()
                    )
                else:
                    total_metric += float(
                        (outputs.detach().argmax(dim=1) == targets).sum().item()
                    )
                total_samples += sample_count

            if total_samples == 0:
                raise ValueError("Cannot train on an empty dataset.")
            values["loss"].append(total_loss / total_samples)
            values[metric_name].append(total_metric / total_samples)

            if validation_data is not None:
                validation_loss, validation_metric = self._evaluate(
                    validation_data[0],
                    validation_data[1],
                    batch_size,
                )
                values["val_loss"].append(validation_loss)
                values[f"val_{metric_name}"].append(validation_metric)

            self.lr_scheduler.step()

        self.trainFlag = True
        self.history = History(values)
        return self.history

    def train(
        self,
        data: Any,
        target: Any,
        validation_data: Optional[Tuple[Any, Any]] = None,
        epochs: int = 75,
        batch_size: int = 50,
    ) -> None:
        self._fit(data, target, validation_data, epochs, batch_size)

    def predict(self, data: Any, batch_size: int = 256) -> np.ndarray:
        model = self._require_model()
        model.eval()
        batches: list[torch.Tensor] = []
        with torch.no_grad():
            for (inputs,) in self._loader(data, batch_size=batch_size):
                outputs = model(inputs.to(self.device, non_blocking=True))
                if not self.regression:
                    outputs = torch.softmax(outputs, dim=1)
                batches.append(outputs.cpu())
        if not batches:
            return np.empty((0, self.output_size), dtype=np.float32)
        return torch.cat(batches, dim=0).numpy()

    def retrain(
        self,
        new_data: Any,
        new_target: Any,
        validation_data: Optional[Tuple[Any, Any]] = None,
        epochs: int = 75,
        batch_size: int = 50,
        fine_tune: bool = False,
        new_learning_rate: Optional[float] = None,
    ) -> None:
        model = self._require_model()
        for parameter in model.parameters():
            parameter.requires_grad = not fine_tune
        if fine_tune:
            for layer in (model.fc1, model.fc2):
                for parameter in layer.parameters():
                    parameter.requires_grad = True

        current_learning_rate = (
            float(new_learning_rate)
            if new_learning_rate is not None
            else (
                float(self.optimizer.param_groups[0]["lr"])
                if self.optimizer is not None
                else self.initial_learning_rate
            )
        )
        self.optimizer = self._create_optimizer(current_learning_rate)
        self.criterion = nn.MSELoss() if self.regression else nn.CrossEntropyLoss()
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=2,
            gamma=0.9,
        )
        self._fit(
            new_data,
            new_target,
            validation_data,
            epochs,
            batch_size,
        )
        print("Retraining completed.")

    def optimize_model(
        self,
        data: Any,
        target: Any,
        validation_data: Any,
        validation_target: Any,
        num_epochs: int = 50,
    ) -> Any:
        n_calls = int(num_epochs)
        if n_calls < 1:
            raise ValueError("num_epochs must be at least 1.")

        def objective(params: Sequence[float]) -> float:
            try:
                self.build_net(
                    {
                        "n_filter": int(params[0]),
                        "section_depth": int(params[1]),
                        "kernel": int(params[2]),
                        "stride": int(params[3]),
                        "num_neurons": int(params[4]),
                        "drop_out": float(params[5]),
                    }
                )
                self.compile_model(float(params[6]))
                self.train(data, target)
                validation_loss, _ = self._evaluate(
                    validation_data,
                    validation_target,
                    batch_size=256,
                )
                return validation_loss
            except (torch.cuda.OutOfMemoryError, MemoryError):
                return float("inf")
            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise
                return float("inf")
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        space = [
            Integer(50, 150, name="n_filter"),
            Integer(3, 5, name="section_depth"),
            Integer(5, 15, name="kernel"),
            Integer(2, 5, name="stride"),
            Integer(800, 1500, name="num_neurons"),
            Real(0.1, 0.5, name="drop_out"),
            Real(
                1e-5,
                1e-3,
                prior="log-uniform",
                name="initial_learning_rate",
            ),
        ]
        result = gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            n_initial_points=min(10, n_calls),
            random_state=42,
        )
        best = result.x
        best_params = {
            "n_filter": int(best[0]),
            "section_depth": int(best[1]),
            "kernel": int(best[2]),
            "stride": int(best[3]),
            "num_neurons": int(best[4]),
            "drop_out": float(best[5]),
            "initial_learning_rate": float(best[6]),
        }
        self.build_net(best_params)
        self.compile_model(best_params["initial_learning_rate"])
        self.train(data, target)

        output_path = (
            Path(__file__).resolve().parents[1]
            / "Evaluation"
            / "Results"
            / "bestParams.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as parameter_file:
            json.dump(best_params, parameter_file, indent=2)

        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(result.func_vals) + 1),
            result.func_vals,
            marker="o",
        )
        plt.xlabel("Iteration")
        plt.ylabel("Validation Error")
        plt.title("Validation Errors Over Iterations")
        plt.grid(True)
        plt.show()
        return result

    def custom_occlusion(
        self,
        data_train: np.ndarray,
        data_test: np.ndarray,
        method: str = "custom",
    ) -> np.ndarray:
        data_train = np.asarray(data_train)
        data_test = np.asarray(data_test)
        if data_train.ndim != 4 or data_test.ndim != 4:
            raise ValueError("Occlusion data must use NHWC format.")

        activation_map = np.zeros_like(data_test, dtype=np.float32)
        mean_train_sample = np.mean(data_train, axis=0)
        kernel_half = 15
        stride = 10
        number_of_steps = int(math.ceil(data_train.shape[2] / stride))

        for sample_index in range(data_test.shape[0]):
            original_sample = data_test[sample_index]
            original_prediction = self.predict(np.expand_dims(original_sample, axis=0))
            occluded_samples = np.zeros(
                (
                    number_of_steps * data_train.shape[1],
                    data_train.shape[1],
                    data_train.shape[2],
                    data_train.shape[3],
                ),
                dtype=np.float32,
            )

            output_index = 0
            for sensor_index in range(data_train.shape[1]):
                for step_index in range(number_of_steps):
                    occluded = original_sample.copy()
                    start = max(0, step_index * stride - kernel_half)
                    end = min(
                        data_train.shape[2],
                        step_index * stride + kernel_half + 1,
                    )
                    if method == "subsensor":
                        replacement = np.mean(
                            data_test[:, sensor_index, :, :],
                            axis=(0, 1),
                        )
                    elif method == "custom":
                        replacement = mean_train_sample[sensor_index, start:end]
                    else:
                        replacement = np.mean(data_test, axis=(0, 1, 2))
                    occluded[sensor_index, start:end] = replacement
                    occluded_samples[output_index] = occluded
                    output_index += 1

            predictions = self.predict(occluded_samples)
            importance = (
                np.abs(self.zscore(predictions - original_prediction))
                / (np.abs(original_prediction) + 1.0)
            ).squeeze()
            importance = np.atleast_1d(importance).reshape(
                data_train.shape[1],
                number_of_steps,
            )

            source_x = np.linspace(
                0,
                data_train.shape[2] - 1,
                number_of_steps,
            )
            target_x = np.arange(data_train.shape[2])
            for sensor_index in range(data_train.shape[1]):
                activation_map[sample_index, sensor_index, :, 0] = np.interp(
                    target_x,
                    source_x,
                    importance[sensor_index],
                )
        return activation_map

    @staticmethod
    def zscore(array: np.ndarray, axis: int = 0, ddof: int = 0) -> np.ndarray:
        array = np.asarray(array)
        mean = np.mean(array, axis=axis)
        std = np.std(array, axis=axis, ddof=ddof)
        return np.divide(
            array - mean,
            std,
            out=np.zeros_like(array, dtype=np.float64),
            where=std != 0,
        )

    def get_gradient_map(
        self,
        data: Any,
        layer_name: Optional[str] = None,
        window_size: int = 5,
    ) -> np.ndarray:
        model = self._require_model()
        model.eval()
        inputs = self._input_tensor(data).to(self.device)
        inputs.requires_grad_(True)

        selected_output: list[torch.Tensor] = []
        hook = None
        if layer_name:
            modules = dict(model.named_modules())
            if layer_name not in modules:
                available = ", ".join(name for name in modules if name)
                raise ValueError(
                    f"Unknown layer '{layer_name}'. Available layers: {available}"
                )
            hook = modules[layer_name].register_forward_hook(
                lambda _module, _inputs, output: selected_output.append(output)
            )

        try:
            predictions = model(inputs)
            output = selected_output[0] if selected_output else predictions
            output.mean().backward()
        finally:
            if hook is not None:
                hook.remove()

        gradients = inputs.grad.detach().abs().permute(0, 2, 3, 1).cpu().numpy()
        smoothed = np.zeros_like(gradients)
        for sample_index in range(gradients.shape[0]):
            for sensor_index in range(gradients.shape[1]):
                smoothed[sample_index, sensor_index, :, 0] = uniform_filter1d(
                    gradients[sample_index, sensor_index, :, 0],
                    size=max(1, int(window_size)),
                )
        return smoothed

    def copy(self) -> "TCOCNNBase":
        new_instance = self.__class__(
            self.input_size,
            self.output_size,
            self.regression,
            device=str(self.device),
        )
        if self.model is None or self.optim_params is None:
            return new_instance

        new_instance.build_net(copy.deepcopy(self.optim_params))
        new_instance.compile_model(self.initial_learning_rate)
        new_instance.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        if self.optimizer is not None and new_instance.optimizer is not None:
            new_instance.optimizer.load_state_dict(
                copy.deepcopy(self.optimizer.state_dict())
            )
        if self.lr_scheduler is not None and new_instance.lr_scheduler is not None:
            new_instance.lr_scheduler.load_state_dict(
                copy.deepcopy(self.lr_scheduler.state_dict())
            )
        new_instance.history = copy.deepcopy(self.history)
        new_instance.trainFlag = self.trainFlag
        return new_instance
