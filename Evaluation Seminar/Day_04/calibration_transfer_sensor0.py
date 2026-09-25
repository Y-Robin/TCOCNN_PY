"""LOSO-selected Global-6 -> sensor-7 transfer using subsensor 0 only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for folder in ['Networks', 'Evaluation Seminar/Day_03', 'Evaluation Seminar/Day_04']:
    sys.path.insert(0, str(ROOT / folder))

from day4_utils import (fit_network, load_sensor_domains, predict_ppb, save_checkpoint,
                        scale_domain, source_scaler, transfer_subset)
from transfer_workflow import (
    DirectStandardization, PiecewiseStandardization, choose_settings, copy_state,
    joint_training, mapped_predictions, metrics, paired_source_mean, pool_domains,
    run_transfer_search, save_json,
)

PARAMS = dict(
    n_filter=24, section_depth=3, convs_per_block=3, channel_growth=16,
    kernel=9, stride=4, num_neurons=128, drop_out=0.0,
    initial_learning_rate=3e-3, residual=True, batch_size=64,
)
TRANSFER_METHODS = ['Global + target', 'Scratch', 'Head-only', 'Fine-tune', 'DS', 'PDS']
ALL_METHODS = ['Global-6', *TRANSFER_METHODS]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budgets', type=int, nargs='+',
                        default=[5, 10, 20, 40, 80, 120, 137])
    parser.add_argument('--source-epochs', type=int, default=100,
                        help='Epochs for every LOSO and final Global model.')
    parser.add_argument('--candidate-epochs', type=int, nargs='+',
                        default=[10, 20, 40, 60])
    parser.add_argument('--lr-global-target', type=float, nargs='+',
                        default=[1e-4, 3e-4, 1e-3])
    parser.add_argument('--lr-scratch', type=float, nargs='+',
                        default=[1e-4, 3e-4, 1e-3])
    parser.add_argument('--lr-head', type=float, nargs='+',
                        default=[1e-5, 3e-5, 1e-4, 3e-4])
    parser.add_argument('--lr-finetune', type=float, nargs='+',
                        default=[1e-6, 3e-6, 1e-5, 3e-5, 1e-4])
    parser.add_argument('--alphas', type=float, nargs='+',
                        default=[.001, .01, .1, 1., 10., 1000.])
    parser.add_argument('--windows', type=int, nargs='+', default=[5, 9, 21])
    parser.add_argument('--loso-folds', type=int, default=6,
                        help='Use 6 for the real six-source LOSO; smaller values are for smoke tests.')
    parser.add_argument('--global-params', type=Path,
                        help='best_hyperparameters.json produced by script 00; fixed defaults are used if omitted.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'artifacts' / 'seminar_day4_global6_sensor0')
    return parser.parse_args()


def load_domains(gas='acetone', calibration=1):
    """Compatibility loader returning seven devices with subsensor 0 only."""
    domains, _ = load_sensor_domains(channel=0, calibration=calibration, gas=gas)
    return domains


def load_global_params(path):
    if path is None:
        return dict(PARAMS), 'built-in defaults'
    document = json.loads(path.read_text(encoding='utf-8'))
    params = document.get('params', document)
    required = set(PARAMS)
    missing = required.difference(params)
    if missing:
        raise ValueError(f'Global parameter file is missing: {sorted(missing)}')
    return {name: params[name] for name in required}, str(path.resolve())


def validate_experiment(raw, budgets, windows, loso_folds):
    sensors = list(raw)
    if len(sensors) != 7:
        raise AssertionError(f'Expected seven devices, got {len(sensors)}.')
    for sensor, domain in raw.items():
        for split, values in domain.items():
            if values['X'].shape[1:] != (1, 1440, 1):
                raise AssertionError(
                    f'{sensor}/{split}: expected subsensor-0 shape (1,1440,1), '
                    f'got {values["X"].shape[1:]}'
                )
    available = len(np.unique(raw[sensors[6]]['train']['groups']))
    if not budgets or min(budgets) < 1 or max(budgets) > available:
        raise ValueError(f'Budgets must be between 1 and {available} UGMs.')
    if any(window < 1 or window % 2 != 1 for window in windows):
        raise ValueError('PDS windows must be positive odd integers.')
    if not 1 <= loso_folds <= 6:
        raise ValueError('loso-folds must be between 1 and 6.')
    return sensors, available


def adapter_for(method, alpha, radius):
    if method == 'DS':
        return DirectStandardization(alpha)
    return PiecewiseStandardization(alpha, radius)


def selected_settings(records, budgets, expected_folds):
    selected = {}
    for budget in budgets:
        for method in TRANSFER_METHODS:
            selected[(method, budget)] = choose_settings(
                records, method, budget, expected_folds=expected_folds
            )
    return selected


def train_final_method(method, subset, pooled, params, source_state, setting, seed):
    if method == 'Global + target':
        domain = joint_training(pooled, subset)
        initial = None
    else:
        domain = subset
        initial = None if method == 'Scratch' else source_state
    model, history, info = fit_network(
        domain, {**params, 'initial_learning_rate': setting['lr']},
        epochs=int(setting['epoch']), initial=initial,
        head_only=(method == 'Head-only'),
        freeze_batchnorm=(method in {'Head-only', 'Fine-tune'}),
        restore_best=False, use_validation=False, lr_schedule='constant',
        seed=seed, verbose=False,
    )
    return model, history, info


def plot_source_history(history, scaler, best_epoch, output):
    epochs = np.arange(1, len(history['loss']) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(14, 4))
    for axis in axes:
        axis.plot(epochs, np.sqrt(history['loss']) * scaler['y_std'], label='Train')
        axis.plot(epochs, np.sqrt(history['val_group_mse']) * scaler['y_std'],
                  label='Validation')
        axis.axvline(best_epoch, color='black', ls=':', label='best checkpoint')
        axis.set(xlabel='Epoch', ylabel='RMSE [ppb]')
        axis.grid(True, which='both', alpha=.3)
        axis.legend()
    axes[0].set_title('Global-6 training - linear')
    axes[1].set(title='Global-6 training - logarithmic', yscale='log')
    figure.tight_layout()
    figure.savefig(output / 'source_training_curve.png', dpi=180)
    plt.close(figure)


def plot_loso_selection(records, selected, budgets, output):
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for axis, method in zip(axes.ravel(), TRANSFER_METHODS):
        means = []
        for budget in budgets:
            setting = selected[(method, budget)]
            matches = []
            for row in records:
                if row['method'] != method or row['budget'] != budget:
                    continue
                keys = ['lr', 'epoch'] if method not in {'DS', 'PDS'} else ['alpha', 'radius']
                if all(row.get(key) == setting.get(key) for key in keys):
                    matches.append(row['UGM_RMSE_ppb'])
            means.append(float(np.mean(matches)))
        axis.plot(budgets, means, marker='o')
        axis.set(title=method, xlabel='Transfer budget [UGMs]',
                 ylabel='mean LOSO UGM RMSE [ppb]')
        axis.grid(True, alpha=.3)
    figure.tight_layout()
    figure.savefig(output / 'loso_selected_settings.png', dpi=180)
    plt.close(figure)


def plot_final_results(results, budgets, output):
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for axis, split in zip(axes, ['val', 'test']):
        baseline = next(row['UGM_RMSE_ppb'] for row in results
                        if row['method'] == 'Global-6' and row['split'] == split)
        axis.axhline(baseline, color='black', ls='--', label='Global-6')
        for method in TRANSFER_METHODS:
            rows = sorted((row for row in results
                           if row['method'] == method and row['split'] == split),
                          key=lambda row: row['budget_UGMs'])
            axis.plot([row['budget_UGMs'] for row in rows],
                      [row['UGM_RMSE_ppb'] for row in rows],
                      marker='o', label=method)
        axis.set(title=f'{split.upper()}: Global-6 -> sensor 7, subsensor 0',
                 xlabel='Transfer budget [UGMs]', ylabel='UGM RMSE [ppb]')
        axis.grid(True, alpha=.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / 'all_methods_budget_curve.png', dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw, audit = load_sensor_domains(channel=0, calibration=1, gas='acetone')
    params, params_source = load_global_params(args.global_params)
    budgets = sorted(set(args.budgets))
    sensors, available = validate_experiment(raw, budgets, args.windows, args.loso_folds)
    sources, target = sensors[:6], sensors[6]
    fold_sensors = sources[:args.loso_folds]
    learning_rates = {
        'Global + target': sorted(set(args.lr_global_target)),
        'Scratch': sorted(set(args.lr_scratch)),
        'Head-only': sorted(set(args.lr_head)),
        'Fine-tune': sorted(set(args.lr_finetune)),
    }
    print('Global-6 sources:', sources, '| untouched final target:', target, flush=True)
    print('Only subsensor 0; input shape: (1, 1440, 1)', flush=True)
    print('LOSO pseudo-targets:', fold_sensors, '| budgets:', budgets, flush=True)
    print('Global hyperparameters:', params_source, flush=True)

    records = run_transfer_search(
        raw, sources, params, budgets, learning_rates,
        sorted(set(args.candidate_epochs)), args.source_epochs, args.output,
        seed=args.seed, alphas=sorted(set(args.alphas)),
        pds_windows=sorted(set(args.windows)), fold_sensors=fold_sensors,
    )
    selected = selected_settings(records, budgets, len(fold_sensors))
    selected_rows = [
        {'method': method, 'budget_UGMs': budget, **setting}
        for (method, budget), setting in selected.items()
    ]
    save_json(args.output / 'selected_loso_settings.json', selected_rows)
    plot_loso_selection(records, selected, budgets, args.output)
    print('LOSO selection is frozen; sensor 7 was not used.', flush=True)

    scaler = source_scaler(pool_domains(raw, sources))
    domains = {sensor: scale_domain(raw[sensor], scaler) for sensor in sensors}
    pooled = pool_domains(domains, sources)
    source, source_history, source_info = fit_network(
        pooled, params, epochs=args.source_epochs, seed=args.seed, verbose=True
    )
    source_state = copy_state(source)
    source_reference = pooled['train']['X_z'].mean(
        axis=0, keepdims=True, dtype=np.float64
    ).astype(np.float32)
    save_checkpoint(
        args.output / 'source_global6', source, scaler,
        {'params': params, 'input_shape': [1, 1440, 1], 'gas': 'acetone',
         'sources': sources, 'target': target, 'calibration': 1, 'seed': args.seed,
         'selected_epoch': source_info['best_epoch']},
        source_reference,
    )
    plot_source_history(
        source_history, scaler, source_info['best_epoch'], args.output
    )

    results = []
    predictions = {}
    for split in ['val', 'test']:
        pred = predict_ppb(source, domains[target][split], scaler)
        predictions[(0, 'Global-6', split)] = pred
        results.append({'method': 'Global-6', 'budget_UGMs': 0, 'split': split,
                        **metrics(domains[target][split]['y'], pred,
                                  domains[target][split]['groups'])})

    for budget in budgets:
        subset = transfer_subset(domains[target], budget, seed=args.seed)
        target_pairs = subset['train']['X_z']
        master_pairs = paired_source_mean(domains, sources, subset)
        fitted = {}
        for method in ['DS', 'PDS']:
            setting = selected[(method, budget)]
            adapter = adapter_for(
                method, setting['alpha'], setting['radius']
            ).fit(target_pairs, master_pairs)
            fitted[method] = adapter
        for method in ['Global + target', 'Scratch', 'Head-only', 'Fine-tune']:
            model, history, info = train_final_method(
                method, subset, pooled, params, source_state,
                selected[(method, budget)], args.seed
            )
            fitted[method] = model
            if method == 'Fine-tune' and (
                budget == 40 or (40 not in budgets and budget == max(budgets))
            ):
                save_checkpoint(
                    args.output / 'target_finetuned', model, scaler,
                    {'params': params, 'input_shape': [1, 1440, 1], 'gas': 'acetone',
                     'sources': sources, 'target': target, 'calibration': 1,
                     'seed': args.seed, 'budget_UGMs': budget,
                     'loso_setting': selected[(method, budget)]},
                    subset['train']['X_z'].mean(axis=0, keepdims=True),
                )
        for split in ['val', 'test']:
            for method in TRANSFER_METHODS:
                if method in {'DS', 'PDS'}:
                    pred = mapped_predictions(
                        source, fitted[method], domains[target][split], scaler
                    )
                else:
                    pred = predict_ppb(fitted[method], domains[target][split], scaler)
                predictions[(budget, method, split)] = pred
                results.append({
                    'method': method, 'budget_UGMs': budget, 'split': split,
                    **metrics(domains[target][split]['y'], pred,
                              domains[target][split]['groups']),
                })
        print(f'FINAL target budget {budget}/{available} UGMs completed.', flush=True)

    save_json(args.output / 'final_metrics.json', results)
    save_json(args.output / 'run_config.json', {
        'params': params, 'global_params_source': params_source,
        'sources': sources, 'target': target, 'channel': 0,
        'input_shape': [1, 1440, 1], 'budgets': budgets,
        'source_epochs': args.source_epochs,
        'candidate_epochs': sorted(set(args.candidate_epochs)),
        'learning_rates': learning_rates, 'alphas': sorted(set(args.alphas)),
        'pds_windows': sorted(set(args.windows)),
        'loso_folds': fold_sensors, 'seed': args.seed, 'audit': audit,
        'selection_rule': 'mean held-out UGM RMSE across source-device LOSO folds',
        'final_target_used_for_selection': False,
    })
    plot_final_results(results, budgets, args.output)
    print('Saved:', args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
