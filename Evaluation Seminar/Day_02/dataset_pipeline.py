"""Datenbasis fuer den schrittweisen Neuaufbau des ML-Seminars."""

from __future__ import annotations

import csv
import hashlib
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

ZENODO_DOI = "10.5281/zenodo.6821340"
DATASET_URL = "https://zenodo.org/records/6821340/files/fullData.mat?download=1"
DATASET_MD5 = "6ce3fefe1852219e1a66b3fcc24db045"
PHYSICAL_SENSOR = "sensorA"
SUB_SENSOR_INDEX = 0
SENSOR_DIMENSION = 1
SAMPLE_RATE_HZ = 10
CYCLE_SAMPLES = 1440
HIGH_TEMPERATURE_C = 400
HIGH_PHASE_SAMPLES = 50
LOW_PHASE_SAMPLES = 70
LOW_TEMPERATURES_C = tuple(range(100, 400, 25))
BASE_SEGMENT_MAX_UGM = 500
EXTRA_TEST_MIN_UGM = 501
BASE_TEST_SIZE = 0.2
VALIDATION_SIZE_OF_DEVELOPMENT = 0.2
RANDOM_STATE = 42
SPLIT_NAMES = ("train", "val", "test", "test_extra")
TRANSFORMS = ("stored", "log1p")

GAS_TARGETS = (
    "acetic_acid", "acetone", "carbon_monoxide", "ethanol",
    "ethyl_acetate", "formaldehyde", "hydrogen", "isopropanol",
    "toluene", "xylene",
)
INTERFERENCE_TARGETS = (*GAS_TARGETS, "water")


@dataclass(frozen=True)
class BoundaryPoint:
    index: int
    temperature_c: int
    phase: str
    cycle_step: int
    edge: str

    @property
    def point_id(self) -> str:
        return (
            f"t{self.temperature_c:03d}_{self.phase}_"
            f"step{self.cycle_step:02d}_{self.edge}_i{self.index:04d}"
        )


@dataclass(frozen=True)
class PointScore:
    target: str
    index: int
    temperature_c: int
    phase: str
    cycle_step: int
    edge: str
    target_correlation: float
    absolute_target_correlation: float
    strongest_interferer: str
    interferer_correlation: float
    absolute_interferer_correlation: float
    selectivity_margin: float

    @property
    def point_id(self) -> str:
        return (
            f"t{self.temperature_c:03d}_{self.phase}_"
            f"step{self.cycle_step:02d}_{self.edge}_i{self.index:04d}"
        )


def project_root(start: str | Path | None = None) -> Path:
    here = Path(start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "README.md").exists() and (candidate / "Data").exists():
            return candidate
    raise FileNotFoundError("Repository root with README.md and Data/ not found")


def dataset_path() -> Path:
    return project_root() / "Data" / "fullData.mat"


def _md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dataset(download: bool = True, verify: bool = False) -> Path:
    path = dataset_path()
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} is missing")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".mat.part")
        urllib.request.urlretrieve(DATASET_URL, partial)
        partial.replace(path)
    if verify and _md5(path) != DATASET_MD5:
        raise ValueError("The Zenodo dataset MD5 checksum does not match")
    return path


def temperature_boundary_points() -> tuple[BoundaryPoint, ...]:
    """All 48 start/end points of the 24 temperature phases."""

    points: list[BoundaryPoint] = []
    block_samples = HIGH_PHASE_SAMPLES + LOW_PHASE_SAMPLES
    for step, low_temperature in enumerate(LOW_TEMPERATURES_C):
        base = step * block_samples
        points.extend(
            (
                BoundaryPoint(base, HIGH_TEMPERATURE_C, "high", step, "start"),
                BoundaryPoint(base + 49, HIGH_TEMPERATURE_C, "high", step, "end"),
                BoundaryPoint(base + 50, low_temperature, "low", step, "start"),
                BoundaryPoint(base + 119, low_temperature, "low", step, "end"),
            )
        )
    if points[-1].index != CYCLE_SAMPLES - 1:
        raise AssertionError("Temperature profile and cycle length are inconsistent")
    return tuple(points)


def temperature_profile() -> np.ndarray:
    """Temperature setpoint profile for one cycle with 1,440 samples."""

    blocks: list[np.ndarray] = []
    for low_temperature in LOW_TEMPERATURES_C:
        blocks.extend(
            (
                np.full(HIGH_PHASE_SAMPLES, HIGH_TEMPERATURE_C, dtype=np.int16),
                np.full(LOW_PHASE_SAMPLES, low_temperature, dtype=np.int16),
            )
        )
    profile = np.concatenate(blocks)
    if profile.shape != (CYCLE_SAMPLES,):
        raise AssertionError("Unerwartete Laenge des Temperaturprofils")
    return profile


def _load_single_channel(part: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if part not in {"train", "test"}:
        raise ValueError("part must be 'train' or 'test'")
    with h5py.File(ensure_dataset(), "r") as file:
        references = file[f"{PHYSICAL_SENSOR}_{part}"]
        raw = np.asarray(file[references[SUB_SENSOR_INDEX, 0]]).T.astype(np.float32)
        targets_group = file[f"targets_{part}"]
        targets = {
            name: np.asarray(targets_group[name]).ravel().astype(np.float64)
            for name in targets_group
        }
    data = raw[:, np.newaxis, :]
    if data.shape[1] != SENSOR_DIMENSION or data.shape[2] != CYCLE_SAMPLES:
        raise AssertionError(f"Unerwartete Datenform: {data.shape}")
    return data, targets


def _take_targets(
    targets: dict[str, np.ndarray], indices: np.ndarray
) -> dict[str, np.ndarray]:
    return {name: values[indices] for name, values in targets.items()}


def _transform_data(data: np.ndarray, transform: str) -> np.ndarray:
    if transform == "stored":
        return np.asarray(data, dtype=np.float32)
    if transform == "log1p":
        if np.any(data < 0):
            raise ValueError("log1p is not defined for negative sensor values")
        return np.log1p(data).astype(np.float32)
    raise ValueError(f"Unknown transformation: {transform}")


def prepare_splits(transform: str = "stored") -> dict[str, dict[str, object]]:
    """Teilt UGM 1-500 gruppiert; UGM 501-906 bleiben exklusiver Extratest."""

    official_train_x, official_train_targets = _load_single_channel("train")
    official_test_x, official_test_targets = _load_single_channel("test")
    all_x = _transform_data(
        np.concatenate((official_train_x, official_test_x), axis=0), transform
    )
    all_targets = {
        name: np.concatenate(
            (official_train_targets[name], official_test_targets[name])
        )
        for name in official_train_targets
    }
    groups = all_targets["range"].astype(np.int64)
    base_indices = np.flatnonzero(groups <= BASE_SEGMENT_MAX_UGM)
    extra_indices = np.flatnonzero(groups >= EXTRA_TEST_MIN_UGM)

    test_splitter = GroupShuffleSplit(
        n_splits=1, test_size=BASE_TEST_SIZE, random_state=RANDOM_STATE
    )
    development_pos, test_pos = next(
        test_splitter.split(
            np.zeros(len(base_indices)), groups=groups[base_indices]
        )
    )
    development_indices = base_indices[development_pos]
    test_indices = base_indices[test_pos]

    validation_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE_OF_DEVELOPMENT,
        random_state=RANDOM_STATE,
    )
    train_pos, validation_pos = next(
        validation_splitter.split(
            np.zeros(len(development_indices)),
            groups=groups[development_indices],
        )
    )
    index_sets = {
        "train": development_indices[train_pos],
        "val": development_indices[validation_pos],
        "test": test_indices,
        "test_extra": extra_indices,
    }
    result: dict[str, dict[str, object]] = {}
    for split, indices in index_sets.items():
        result[split] = {
            "X": all_x[indices],
            "targets": _take_targets(all_targets, indices),
            "source_indices": indices.astype(np.int64),
            "source": "merged_official_train_and_test",
            "transform": transform,
        }
    validate_splits(result)
    return result


def validate_splits(splits: dict[str, dict[str, object]]) -> None:
    group_sets: dict[str, set[int]] = {}
    for split in SPLIT_NAMES:
        X = np.asarray(splits[split]["X"])
        targets = splits[split]["targets"]
        if X.ndim != 3 or X.shape[1] != SENSOR_DIMENSION:
            raise AssertionError(f"{split}: sensor dimension must have length one: {X.shape}")
        group_sets[split] = set(np.asarray(targets["range"], dtype=int))
    if any(
        group_sets[left] & group_sets[right]
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    ):
        raise AssertionError("Splits enthalten gemeinsame UGM-Gruppen")
    for split in ("train", "val", "test"):
        if max(group_sets[split]) > BASE_SEGMENT_MAX_UGM:
            raise AssertionError(f"{split} contains a UGM from the final segment")
    if min(group_sets["test_extra"]) < EXTRA_TEST_MIN_UGM:
        raise AssertionError("test_extra contains a UGM from the first segments")


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], dtype=np.float64)
    y = np.asarray(y[finite], dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    x -= x.mean()
    y -= y.mean()
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / denominator) if denominator else float("nan")


def score_boundary_points(
    splits: dict[str, dict[str, object]], target: str
) -> list[PointScore]:
    """Scores allowed measurement points using training data only."""

    if target not in GAS_TARGETS:
        raise ValueError(f"Unknown gas: {target}")
    train = splits["train"]
    X = np.asarray(train["X"])[:, 0, :]
    targets = train["targets"]
    scores: list[PointScore] = []
    for point in temperature_boundary_points():
        values = X[:, point.index]
        target_corr = _pearson(values, np.asarray(targets[target]))
        interferer_correlations = {
            name: _pearson(values, np.asarray(targets[name]))
            for name in INTERFERENCE_TARGETS
            if name != target
        }
        strongest = max(
            interferer_correlations,
            key=lambda name: abs(interferer_correlations[name]),
        )
        interferer_corr = interferer_correlations[strongest]
        abs_target = abs(target_corr)
        abs_interferer = abs(interferer_corr)
        scores.append(
            PointScore(
                target, point.index, point.temperature_c, point.phase,
                point.cycle_step, point.edge, target_corr, abs_target,
                strongest, interferer_corr, abs_interferer,
                abs_target - abs_interferer,
            )
        )
    return scores


def selected_point_scores(
    scores: Iterable[PointScore],
) -> dict[str, PointScore]:
    """Overall best, most selective, and per-temperature best individual points."""

    scores = list(scores)
    if not scores:
        raise ValueError("Keine Punktbewertungen vorhanden")
    selected = {
        "best_correlation": max(
            scores, key=lambda item: item.absolute_target_correlation
        ),
        "best_selectivity": max(scores, key=lambda item: item.selectivity_margin),
    }
    for temperature in sorted({item.temperature_c for item in scores}):
        candidates = [
            item for item in scores if item.temperature_c == temperature
        ]
        selected[f"temperature_{temperature:03d}C"] = max(
            candidates, key=lambda item: item.absolute_target_correlation
        )
    return selected


def build_point_dataset(
    splits: dict[str, dict[str, object]], target: str, score: PointScore
) -> dict[str, np.ndarray]:
    """Extract exactly one raw measurement point; X has shape (n, 1, 1)."""

    result: dict[str, np.ndarray] = {}
    for split in SPLIT_NAMES:
        split_data = splits[split]
        result[f"X_{split}"] = np.asarray(split_data["X"])[
            :, :, score.index : score.index + 1
        ].astype(np.float32)
        targets = split_data["targets"]
        result[f"y_{split}"] = np.asarray(targets[target], dtype=np.float32)
        result[f"acetone_{split}"] = np.asarray(
            targets["acetone"], dtype=np.float32
        )
        result[f"groups_{split}"] = np.asarray(
            targets["range"], dtype=np.int64
        )
        result[f"source_indices_{split}"] = np.asarray(
            split_data["source_indices"], dtype=np.int64
        )
        if result[f"X_{split}"].shape[1:] != (1, 1):
            raise AssertionError("Single-point dataset must have shape (n, 1, 1)")
    return result


def build_multi_point_dataset(
    splits: dict[str, dict[str, object]],
    target: str,
    point_indices: Iterable[int],
) -> dict[str, np.ndarray]:
    """Combines several raw time points without changing the sensor axis."""

    indices = np.asarray(list(point_indices), dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("At least one point index is required")
    if np.any(indices < 0) or np.any(indices >= CYCLE_SAMPLES):
        raise ValueError("Punktindex ausserhalb des Temperaturzyklus")
    result: dict[str, np.ndarray] = {}
    for split in SPLIT_NAMES:
        split_data = splits[split]
        result[f"X_{split}"] = np.asarray(split_data["X"])[
            :, :, indices
        ].astype(np.float32)
        targets = split_data["targets"]
        result[f"y_{split}"] = np.asarray(targets[target], dtype=np.float32)
        result[f"groups_{split}"] = np.asarray(
            targets["range"], dtype=np.int64
        )
        if result[f"X_{split}"].shape[1:] != (1, len(indices)):
            raise AssertionError("Sensor dimension must remain one with multiple points")
    result["point_indices"] = indices
    return result


def _all_targets_for_export(
    splits: dict[str, dict[str, object]]
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for split in SPLIT_NAMES:
        targets = splits[split]["targets"]
        for name, values in targets.items():
            arrays[f"target_{name}_{split}"] = np.asarray(values)
        arrays[f"source_indices_{split}"] = np.asarray(
            splits[split]["source_indices"], dtype=np.int64
        )
    return arrays


def export_base_datasets(
    output_dir: str | Path, transform: str = "stored"
) -> tuple[Path, Path]:
    """Exports the full cycle and all allowed boundary points once each."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    splits = prepare_splits(transform)
    common = _all_targets_for_export(splits)
    metadata = {
        "zenodo_doi": np.asarray(ZENODO_DOI),
        "physical_sensor": np.asarray(PHYSICAL_SENSOR),
        "sub_sensor_index": np.asarray(SUB_SENSOR_INDEX, dtype=np.int64),
        "sensor_dimension": np.asarray(SENSOR_DIMENSION, dtype=np.int64),
        "transform": np.asarray(transform),
        "base_segment_max_ugm": np.asarray(
            BASE_SEGMENT_MAX_UGM, dtype=np.int64
        ),
        "extra_test_min_ugm": np.asarray(
            EXTRA_TEST_MIN_UGM, dtype=np.int64
        ),
    }

    raw_path = destination / f"all_raw_values__{transform}__single_sensor.npz"
    np.savez_compressed(
        raw_path,
        **{
            f"X_{split}": np.asarray(splits[split]["X"], dtype=np.float32)
            for split in SPLIT_NAMES
        },
        **common,
        **metadata,
    )

    points = temperature_boundary_points()
    point_indices = np.asarray([point.index for point in points], dtype=np.int64)
    boundary_path = destination / (
        f"all_temperature_boundaries__{transform}__single_sensor.npz"
    )
    np.savez_compressed(
        boundary_path,
        **{
            f"X_{split}": np.asarray(splits[split]["X"])[
                :, :, point_indices
            ].astype(np.float32)
            for split in SPLIT_NAMES
        },
        **common,
        **metadata,
        point_indices=point_indices,
        point_ids=np.asarray([point.point_id for point in points]),
        temperatures_c=np.asarray(
            [point.temperature_c for point in points], dtype=np.int64
        ),
        edges=np.asarray([point.edge for point in points]),
    )
    return raw_path, boundary_path


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def export_point_datasets(
    output_dir: str | Path, targets: Iterable[str] = GAS_TARGETS
) -> Path:
    """Exports separate single-point datasets per gas and a manifest."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    for transform in TRANSFORMS:
        splits = prepare_splits(transform)
        for target in targets:
            scores = score_boundary_points(splits, target)
            for selection, score in selected_point_scores(scores).items():
                dataset = build_point_dataset(splits, target, score)
                filename = (
                    f"{_slug(target)}__{transform}__{_slug(selection)}__"
                    f"{score.point_id}.npz"
                )
                np.savez_compressed(
                    destination / filename,
                    **dataset,
                    target=np.asarray(target),
                    transform=np.asarray(transform),
                    selection=np.asarray(selection),
                    point_index=np.asarray(score.index, dtype=np.int64),
                    temperature_c=np.asarray(
                        score.temperature_c, dtype=np.int64
                    ),
                    edge=np.asarray(score.edge),
                    zenodo_doi=np.asarray(ZENODO_DOI),
                    physical_sensor=np.asarray(PHYSICAL_SENSOR),
                    sub_sensor_index=np.asarray(
                        SUB_SENSOR_INDEX, dtype=np.int64
                    ),
                    base_segment_max_ugm=np.asarray(
                        BASE_SEGMENT_MAX_UGM, dtype=np.int64
                    ),
                    extra_test_min_ugm=np.asarray(
                        EXTRA_TEST_MIN_UGM, dtype=np.int64
                    ),
                )
                row = asdict(score)
                row.update(
                    {
                        "transform": transform,
                        "selection": selection,
                        "point_id": score.point_id,
                        "filename": filename,
                        "n_train": len(dataset["y_train"]),
                        "n_val": len(dataset["y_val"]),
                        "n_test": len(dataset["y_test"]),
                        "n_test_extra": len(dataset["y_test_extra"]),
                        "sensor_dimension": SENSOR_DIMENSION,
                        "base_ugm": "1-500",
                        "extra_test_ugm": "501-906",
                        "correlation_source": "train_only",
                    }
                )
                manifest_rows.append(row)
    manifest_path = destination / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_path


if __name__ == "__main__":
    output = project_root() / "Data" / "seminar_point_datasets"
    manifest = export_point_datasets(output)
    print(f"Single-point export: {manifest}")
    for transform in TRANSFORMS:
        raw_path, boundary_path = export_base_datasets(output, transform)
        print(f"Vollstaendiger Rohzyklus ({transform}): {raw_path}")
        print(f"All allowed boundary points ({transform}): {boundary_path}")
