"""Bayesian hyperparameter optimization for the six-source TCOCNNv3 model.

Only subsensor 0 from source devices 1-6 participates. Device 7 is excluded
from training, validation, checkpoint selection, and the optimization score.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from skopt import Optimizer
from skopt.space import Categorical, Real

ROOT = Path(__file__).resolve().parents[2]
for folder in ['Networks', 'Evaluation Seminar/Day_03', 'Evaluation Seminar/Day_04']:
    sys.path.insert(0, str(ROOT / folder))

from day4_utils import (fit_network, load_sensor_domains, predict_ppb,
                        save_checkpoint, scale_domain, source_scaler)
from transfer_workflow import metrics, pool_domains, save_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum epochs per candidate; its best source-validation checkpoint is kept.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'artifacts' / 'seminar_day4_global6_hpo')
    return parser.parse_args()


def search_space():
    """Named dimensions used by the reproducible Bayesian search."""
    return [
        Categorical([16, 24, 32, 48], name='n_filter'),
        Categorical([2, 3, 4], name='section_depth'),
        Categorical([1, 2, 3], name='convs_per_block'),
        Categorical([0, 8, 16, 32], name='channel_growth'),
        Categorical([5, 9, 15], name='kernel'),
        Categorical([2, 4], name='stride'),
        Categorical([64, 128, 256], name='num_neurons'),
        Real(0.0, 0.30, name='drop_out'),
        Real(1e-4, 3e-3, prior='log-uniform', name='initial_learning_rate'),
        Categorical([32, 64, 128], name='batch_size'),
        Categorical([False, True], name='residual'),
    ]


def native(value):
    return value.item() if isinstance(value, np.generic) else value


def validate_sources(raw):
    sensors = list(raw)
    if len(sensors) != 7:
        raise AssertionError(f'Expected seven devices, got {len(sensors)}.')
    sources, excluded_target = sensors[:6], sensors[6]
    for sensor in sources:
        for split in ['train', 'val']:
            shape = raw[sensor][split]['X'].shape[1:]
            if shape != (1, 1440, 1):
                raise AssertionError(
                    f'{sensor}/{split}: expected subsensor-0 shape (1,1440,1), got {shape}.'
                )
    return sources, excluded_target


def plot_progress(trials, output):
    scores = np.asarray([row['UGM_RMSE_ppb'] for row in trials])
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(np.arange(1, len(scores) + 1), scores, marker='o', alpha=.55,
              label='trial')
    axis.plot(np.arange(1, len(scores) + 1), np.minimum.accumulate(scores),
              color='black', linewidth=2, label='best so far')
    axis.set(xlabel='Bayesian-search trial', ylabel='Source validation UGM RMSE [ppb]',
             title='Global-6 TCOCNNv3 hyperparameter optimization')
    axis.grid(True, alpha=.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / 'hpo_progress.png', dpi=180)
    plt.close(figure)


def plot_best_history(history, scaler, best_epoch, output):
    epochs = np.arange(1, len(history['loss']) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(14, 4))
    for axis in axes:
        axis.plot(epochs, np.sqrt(history['loss']) * scaler['y_std'], label='train')
        axis.plot(epochs, np.sqrt(history['val_group_mse']) * scaler['y_std'],
                  label='source validation')
        axis.axvline(best_epoch, color='black', linestyle=':', label='saved checkpoint')
        axis.set(xlabel='Epoch', ylabel='RMSE [ppb]')
        axis.grid(True, which='both', alpha=.3)
        axis.legend()
    axes[0].set_title('Best Global-6 trial - linear')
    axes[1].set(title='Best Global-6 trial - logarithmic', yscale='log')
    figure.tight_layout()
    figure.savefig(output / 'best_training_curve.png', dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    if args.trials < 1 or args.epochs < 1:
        raise ValueError('trials and epochs must be positive.')
    args.output.mkdir(parents=True, exist_ok=True)

    raw, audit = load_sensor_domains(channel=0, calibration=1, gas='acetone')
    sources, excluded_target = validate_sources(raw)
    # The scaler and both optimization splits are derived from source devices only.
    source_raw = pool_domains({sensor: raw[sensor] for sensor in sources}, sources)
    scaler = source_scaler(source_raw)
    source_domains = {
        sensor: scale_domain(raw[sensor], scaler) for sensor in sources
    }
    pooled = pool_domains(source_domains, sources)
    print('HPO sources:', sources, flush=True)
    print('Excluded target:', excluded_target, flush=True)
    print('Only subsensor 0; model input:', pooled['train']['X_z'].shape[1:], flush=True)
    print('Train cycles:', len(pooled['train']['y']),
          '| validation sensor-UGMs:', len(np.unique(pooled['val']['eval_groups'])),
          flush=True)

    dimensions = search_space()
    optimizer = Optimizer(
        dimensions, n_initial_points=min(8, args.trials), random_state=args.seed
    )
    trials = []
    best_score = float('inf')
    best_model = best_history = best_info = best_params = None
    reference = pooled['train']['X_z'].mean(
        axis=0, keepdims=True, dtype=np.float64
    ).astype(np.float32)

    for trial_number in range(1, args.trials + 1):
        values = optimizer.ask()
        params = {dimension.name: native(value)
                  for dimension, value in zip(dimensions, values)}
        for name in ['n_filter', 'section_depth', 'convs_per_block',
                     'channel_growth', 'kernel', 'stride', 'num_neurons',
                     'batch_size']:
            params[name] = int(params[name])
        params['drop_out'] = float(params['drop_out'])
        params['initial_learning_rate'] = float(params['initial_learning_rate'])
        params['residual'] = bool(params['residual'])
        started = time.perf_counter()
        try:
            model, history, info = fit_network(
                pooled, params, epochs=args.epochs, seed=args.seed,
                restore_best=True, use_validation=True, verbose=False
            )
            prediction = predict_ppb(model, pooled['val'], scaler)
            score = metrics(
                pooled['val']['y'], prediction, pooled['val']['eval_groups']
            )
            objective = score['UGM_RMSE_ppb']
            parameter_count = int(sum(p.numel() for p in model.model.parameters()))
            row = {
                'trial': trial_number, **params, **score,
                'best_epoch': info['best_epoch'],
                'parameter_count': parameter_count,
                'seconds': time.perf_counter() - started,
                'status': 'ok',
            }
            optimizer.tell(values, objective)
            trials.append(row)
            if objective < best_score:
                best_score = objective
                best_model, best_history, best_info = model, history, info
                best_params = dict(params)
                save_checkpoint(
                    args.output / 'source_global6_hpo', best_model, scaler,
                    {'params': best_params, 'input_shape': [1, 1440, 1],
                     'gas': 'acetone', 'calibration': 1, 'channel': 0,
                     'sources': sources, 'excluded_target': excluded_target,
                     'seed': args.seed, 'hpo_trials': args.trials,
                     'trial_epochs': args.epochs,
                     'selected_trial': trial_number,
                     'selected_epoch': best_info['best_epoch'],
                     'validation_UGM_RMSE_ppb': best_score,
                     'parameter_count': parameter_count},
                    reference,
                )
                save_json(args.output / 'best_history.json', best_history)
            print(
                f'Trial {trial_number:02d}/{args.trials}: '
                f'val UGM RMSE={objective:.3f} ppb | epoch={info["best_epoch"]} | '
                f'parameters={parameter_count:,}', flush=True
            )
        except (RuntimeError, MemoryError) as error:
            if isinstance(error, RuntimeError) and 'out of memory' not in str(error).lower():
                raise
            optimizer.tell(values, 1e9)
            trials.append({
                'trial': trial_number, **params, 'status': 'out_of_memory',
                'error': str(error), 'seconds': time.perf_counter() - started,
            })
            print(f'Trial {trial_number:02d}/{args.trials}: out of memory', flush=True)
        finally:
            save_json(args.output / 'trials.json', trials)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if best_model is None:
        raise RuntimeError('Every HPO trial failed.')
    best_document = {
        'params': best_params,
        'best_validation_UGM_RMSE_ppb': best_score,
        'best_epoch': best_info['best_epoch'],
        'selected_trial': next(
            row['trial'] for row in trials
            if row.get('status') == 'ok'
            and row['UGM_RMSE_ppb'] == best_score
        ),
        'sources': sources,
        'excluded_target': excluded_target,
        'channel': 0,
        'input_shape': [1, 1440, 1],
        'selection_metric': 'sensor-specific UGM RMSE on pooled source validation',
        'test_or_target7_used': False,
        'search_space': {
            dimension.name: [native(value) for value in dimension.categories]
            if hasattr(dimension, 'categories')
            else {'low': native(dimension.low), 'high': native(dimension.high),
                  'prior': dimension.prior}
            for dimension in dimensions
        },
        'audit': audit,
    }
    save_json(args.output / 'best_hyperparameters.json', best_document)
    plot_progress([row for row in trials if row.get('status') == 'ok'], args.output)
    plot_best_history(best_history, scaler, best_info['best_epoch'], args.output)
    print('BEST', json.dumps(best_document['params'], indent=2), flush=True)
    print(f'Best source-validation UGM RMSE: {best_score:.3f} ppb', flush=True)
    print('Saved:', args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
