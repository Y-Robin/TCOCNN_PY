"""Auditable six-source calibration transfer using UGM-level evaluation only."""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from day4_utils import (fit_network, source_scaler, scale_domain, transfer_subset,
                        predict_ppb, predict_normalized, gas_label)

GAS = 'acetone'
GAS_LABEL = gas_label(GAS)
GAS_UNIT = 'ppb'
COLORS = {'Global-6':'#7f7f7f','Global + target':'#17becf','DS':'#9467bd','PDS':'#8c564b',
          'Scratch':'#ff7f0e','Head-only':'#2ca02c','Fine-tune':'#d62728'}
METHODS = list(COLORS)


def group_values(truth, prediction, groups):
    """Average truth and prediction across every cycle belonging to one UGM."""
    ids, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    yt = np.bincount(inverse, weights=np.asarray(truth).ravel()) / counts
    yp = np.bincount(inverse, weights=np.asarray(prediction).ravel()) / counts
    return yt, yp


def metrics(truth, prediction, groups):
    """Seminar metric: UGM-mean RMSE/R² only; no cycle-level RMSE is reported."""
    yt, yp = group_values(truth, prediction, groups)
    return {'UGM_RMSE_ppb': float(np.sqrt(np.mean((yt - yp) ** 2))),
            'UGM_R2': float(r2_score(yt, yp)),
            'UGMs': int(len(yt))}


def pool_domains(domains, sensors):
    """Pool sensor observations while retaining separate sensor-UGM validation groups."""
    result = {}
    for split in ['train', 'val']:
        result[split] = {key: np.concatenate([domains[sensor][split][key] for sensor in sensors])
                         for key in domains[sensors[0]][split] if key != 'eval_groups'}
        result[split]['eval_groups'] = np.concatenate([
            i * 10000 + domains[sensor][split]['groups'] for i, sensor in enumerate(sensors)
        ])
    return result


def joint_training(pooled, target_subset):
    """Add target TRAIN cycles only; source validation remains the checkpoint selector."""
    result = {name: dict(split) for name, split in pooled.items()}
    extra = {**target_subset['train'], 'eval_groups': 90000 + target_subset['train']['groups']}
    result['train'] = {key: np.concatenate([value, extra[key]]) for key, value in pooled['train'].items()}
    return result


def copy_state(wrapper):
    return {k: v.detach().cpu().clone() for k, v in wrapper.model.state_dict().items()}


def paired_source_mean(domains, source_sensors, subset, split='train'):
    """Virtual source reference from all available source sensors.

    For every target cycle retained by ``transfer_subset`` we retrieve the *same original
    measurement row* from every source sensor and average their standardized signals.
    Consequently all cycles from every selected UGM are used, while no unpaired source UGM
    can leak into the DS/PDS calibration mapping.
    """
    if not source_sensors:
        raise ValueError('At least one source sensor is required.')
    target_rows = np.asarray(subset[split]['rows'])
    target_groups = np.asarray(subset[split]['groups'])
    aligned = []
    for sensor in source_sensors:
        source_split = domains[sensor][split]
        lookup = {int(row): i for i, row in enumerate(source_split['rows'])}
        try:
            indices = np.asarray([lookup[int(row)] for row in target_rows], dtype=int)
        except KeyError as exc:
            raise KeyError(f'Row {exc.args[0]} is unavailable for source sensor {sensor}.') from exc
        if not np.array_equal(source_split['rows'][indices], target_rows):
            raise AssertionError(f'Row alignment failed for source sensor {sensor}.')
        if not np.array_equal(source_split['groups'][indices], target_groups):
            raise AssertionError(f'UGM alignment failed for source sensor {sensor}.')
        aligned.append(source_split['X_z'][indices])
    virtual = np.mean(np.stack(aligned, axis=0), axis=0, dtype=np.float64).astype(np.float32)
    if virtual.shape != subset[split]['X_z'].shape:
        raise AssertionError('Virtual-source shape differs from target shape.')
    return virtual


class DirectStandardization:
    """Ridge DS for every channel and time sample of one target device."""
    def __init__(self, alpha=.1):
        self.alpha=float(alpha)

    def fit(self,target,master):
        shape=target.shape[1:]
        x=np.asarray(target,dtype=float).reshape(len(target),-1)
        y=np.asarray(master,dtype=float).reshape(len(master),-1)
        if x.shape!=y.shape:
            raise ValueError("DS target/master pairs need identical shapes.")
        self.shape_=shape
        self.x_mean_=x.mean(0); self.y_mean_=y.mean(0)
        x=x-self.x_mean_; y=y-self.y_mean_
        self.x_=x
        self.dual_=np.linalg.solve(x@x.T+self.alpha*np.eye(len(x)),y)
        return self

    def transform(self,values):
        x=np.asarray(values,dtype=float).reshape(len(values),-1)-self.x_mean_
        return (x@self.x_.T@self.dual_+self.y_mean_).reshape((len(values),)+self.shape_).astype(np.float32)


class PiecewiseStandardization:
    """Local ridge PDS applied separately to all sub-sensor channels."""
    def __init__(self,alpha=.1,radius=2):
        self.alpha=float(alpha); self.radius=int(radius)

    def windows(self,values):
        x=np.asarray(values,dtype=float)[...,0]
        return np.lib.stride_tricks.sliding_window_view(
            np.pad(x,((0,0),(0,0),(self.radius,self.radius)),mode="edge"),
            2*self.radius+1,axis=2)

    def fit(self,target,master):
        x=self.windows(target)
        y=np.asarray(master,dtype=float)[...,0]
        if x.shape[:3]!=y.shape:
            raise ValueError("PDS target/master pairs need identical cycle/channel/time shapes.")
        self.x_mean_=x.mean(0); self.y_mean_=y.mean(0)
        x=x-self.x_mean_; y=y-self.y_mean_
        gram=np.einsum("nctw,nctv->ctwv",x,x,optimize=True)
        rhs=np.einsum("nctw,nct->ctw",x,y,optimize=True)
        gram+=self.alpha*np.eye(x.shape[-1])[None,None]
        self.coef_=np.linalg.solve(gram,rhs[...,None])[...,0]
        return self

    def transform(self,values):
        pred=np.einsum("nctw,ctw->nct",self.windows(values)-self.x_mean_,self.coef_,optimize=True)
        return (pred+self.y_mean_)[...,None].astype(np.float32)


def mapped_predictions(source, adapter, split, scaler):
    return predict_normalized(source, adapter.transform(split['X_z'])).ravel() * scaler['y_std'] + scaler['y_mean']


def choose_settings(rows, method, budget, expected_folds=6):
    """Choose one transfer setting by mean UGM RMSE across all six pseudo-targets."""
    candidates = {}
    for row in rows:
        if row['method'] != method or row['budget'] != budget:
            continue
        key = (row.get('lr'), row.get('epoch'), row.get('alpha'), row.get('radius'))
        candidates.setdefault(key, []).append(row['UGM_RMSE_ppb'])
    if not candidates:
        raise ValueError(f'No search rows for {method}, budget {budget}.')
    key = min(candidates, key=lambda item: np.mean(candidates[item]))
    if len(candidates[key]) != expected_folds:
        raise AssertionError(
            f'Each candidate must include {expected_folds} held-out development sensors.'
        )
    return dict(zip(['lr', 'epoch', 'alpha', 'radius'], key),
                development_RMSE=float(np.mean(candidates[key])))


def plot_predictions(split, predictions, title, gas_label_text=GAS_LABEL, unit=GAS_UNIT):
    """Only UGM-mean scatterplots are shown; individual-cycle RMSE is intentionally omitted."""
    import matplotlib.pyplot as plt
    truth = split['y']; groups = split['groups']
    values = {name: group_values(truth, pred, groups) for name, pred in predictions.items()}
    lo = min(min(yt.min(), yp.min()) for yt, yp in values.values())
    hi = max(max(yt.max(), yp.max()) for yt, yp in values.values())
    margin = max((hi - lo) * .04, 1)
    limits = [lo - margin, hi + margin]
    cols = min(4, len(values)); rows = int(np.ceil(len(values) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.6 * rows), squeeze=False)
    for ax, (name, (yt, yp)) in zip(axes.ravel(), values.items()):
        rmse = np.sqrt(np.mean((yt - yp) ** 2))
        ax.scatter(yt, yp, s=22, alpha=.65, color=COLORS.get(name, '#1f77b4'))
        ax.plot(limits, limits, 'k--')
        ax.grid(True, alpha=.3)
        ax.set(xlim=limits, ylim=limits, aspect='equal',
               xlabel=f'True UGM mean {gas_label_text} [{unit}]',
               ylabel=f'Predicted UGM mean {gas_label_text} [{unit}]',
               title=f'{name}\nUGM RMSE={rmse:.1f} {unit}')
    for ax in axes.ravel()[len(values):]:
        ax.set_visible(False)
    fig.suptitle(f'{title}\n{len(np.unique(groups))} held-out UGMs; predictions averaged within UGM')
    fig.tight_layout(); plt.show()


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2,
                        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')


def run_transfer_search(raw, sources, params, budgets, learning_rates, epochs,
                        source_epochs, output, seed=42, alphas=None,
                        pds_windows=None, fold_sensors=None):
    """Select transfer hyperparameters using six LOSO pseudo-target folds only.

    ``learning_rates`` is a dict with separate grids for Global + target,
    Scratch, Head-only and Fine-tune.
    Transfer candidates use a *constant* LR.  Fine-tuning freezes BatchNorm running
    statistics because the target adaptation sets can be small.
    """
    alphas = [.001, .01, .1, 1., 10., 1000.] if alphas is None else list(alphas)
    pds_windows = [5, 9, 21] if pds_windows is None else list(pds_windows)
    fold_sensors = list(sources if fold_sensors is None else fold_sensors)
    if not fold_sensors or any(sensor not in sources for sensor in fold_sensors):
        raise ValueError('LOSO fold sensors must be a non-empty subset of the sources.')
    records = []
    for fold, heldout in enumerate(fold_sensors):
        training = [s for s in sources if s != heldout]
        fold_raw = {s: raw[s] for s in training}
        scaler = source_scaler(pool_domains(fold_raw, training))
        domains = {s: scale_domain(raw[s], scaler) for s in sources}
        source, _, _ = fit_network(pool_domains(domains, training), params,
                                   epochs=source_epochs, lr_schedule='step', verbose=False)
        state = copy_state(source)
        for budget in budgets:
            subset = transfer_subset(domains[heldout], budget, seed=seed)
            n_cycles = len(subset['train']['y'])
            target_pairs = subset['train']['X_z']
            source_pairs = paired_source_mean(domains, training, subset)

            for method in ['Global + target', 'Scratch', 'Head-only', 'Fine-tune']:
                for lr in learning_rates[method]:
                    adaptation_domain = subset
                    if method == 'Global + target':
                        adaptation_domain = joint_training(
                            pool_domains(domains, training), subset
                        )
                        adaptation_domain['val'] = domains[heldout]['val']
                    model, history, _ = fit_network(
                        adaptation_domain, {**params, 'initial_learning_rate': lr},
                        epochs=max(epochs),
                        initial=None if method in {'Global + target', 'Scratch'} else state,
                        head_only=(method == 'Head-only'),
                        freeze_batchnorm=(method in {'Head-only', 'Fine-tune'}),
                        seed=seed, restore_best=False, use_validation=True,
                        lr_schedule='constant', verbose=False)
                    for epoch in epochs:
                        records.append({
                            'heldout': heldout, 'train_sensors': ','.join(training),
                            'budget': budget, 'adaptation_cycles': n_cycles,
                            'method': method, 'lr': lr, 'epoch': epoch,
                            'UGM_RMSE_ppb': float(np.sqrt(history['val_group_mse'][epoch - 1]) * scaler['y_std'])
                        })

            for alpha in alphas:
                settings = [('DS', None)]
                settings += [('PDS', (window - 1) // 2) for window in pds_windows]
                for method, radius in settings:
                    adapter = (DirectStandardization(alpha) if method == 'DS'
                               else PiecewiseStandardization(alpha, radius))
                    adapter.fit(target_pairs, source_pairs)
                    pred = mapped_predictions(source, adapter, domains[heldout]['val'], scaler)
                    records.append({
                        'heldout': heldout, 'train_sensors': ','.join(training),
                        'budget': budget, 'adaptation_cycles': n_cycles,
                        'virtual_master_sensors': len(training),
                        'calibration_pairs': len(target_pairs),
                        'method': method, 'alpha': alpha, 'radius': radius,
                        **metrics(domains[heldout]['val']['y'], pred, domains[heldout]['val']['groups'])
                    })

            save_json(Path(output) / 'transfer_search.json', records)
            print(f'DEVELOPMENT fold {fold + 1}/{len(fold_sensors)} | pseudo-target {heldout} | '
                  f'{budget} UGMs = {n_cycles} cycles | DS/PDS = {len(target_pairs)} '
                  f'pairs with the mean of {len(training)} aligned source sensors', flush=True)
    return records
