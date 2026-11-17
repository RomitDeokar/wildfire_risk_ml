"""Run python -m wildfire.train after pip install -e ."""
import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, f1_score, brier_score_loss, log_loss,
                             roc_curve, confusion_matrix, ConfusionMatrixDisplay,
                             precision_score, recall_score, accuracy_score, balanced_accuracy_score)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from .bma import BayesianModelAverage
from .data import ROOT, FEATURES, RAW_FEATURES, feature_frame, load_data, grouped_split, sha256

INTERPRETABILITY = {'BMA': 'Weighted logistic subsets; block inclusion probabilities',
                   'Logistic Regression': 'Standardized coefficients',
                   'Decision Tree': 'Decision rules (depth-limited)',
                   'Random Forest': 'Ensemble; global feature importance',
                   'Prevalence Baseline': 'Training positive-class frequency'}


def metrics(y, p, threshold):
    return {'roc_auc': float(roc_auc_score(y, p)),
            'f1': float(f1_score(y, p >= threshold, zero_division=0)),
            'brier': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, p)),
            'f1_at_0_5': float(f1_score(y, p >= .5, zero_division=0)),
            'precision': float(precision_score(y, p >= threshold, zero_division=0)),
            'recall': float(recall_score(y, p >= threshold, zero_division=0)),
            'accuracy': float(accuracy_score(y, p >= threshold)),
            'balanced_accuracy': float(balanced_accuracy_score(y, p >= threshold))}


def calibration_bins(y, p, n_bins=5):
    # Fixed equal-width bins; persist counts because small samples make curves unstable.
    bins = np.minimum((np.asarray(p) * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            rows.append({'bin': b, 'count': int(mask.sum()),
                         'mean_probability': float(np.mean(p[mask])),
                         'observed_fraction': float(np.mean(y[mask]))})
    return pd.DataFrame(rows)


def build_plots(out, results, predictions, weights, ytest, thresholds):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    ordered = weights.sort_values('weight')
    ax.barh(ordered.candidate, ordered.weight, color='#c85b32')
    ax.set(xlabel='Approximate posterior model probability', title='Bayesian model weights (training data only)')
    fig.tight_layout(); fig.savefig(out / 'model_weights.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cal_tables = []
    for name, p in predictions.items():
        fpr, tpr, _ = roc_curve(ytest, p)
        axes[0].plot(fpr, tpr, label=f'{name} (AUC {roc_auc_score(ytest, p):.3f})')
        observed, predicted = calibration_curve(ytest, p, n_bins=5, strategy='uniform')
        axes[1].plot(predicted, observed, 'o-', label=name)
        table = calibration_bins(ytest, p)
        table.insert(0, 'algorithm', name); cal_tables.append(table)
    for ax in axes:
        ax.plot([0, 1], [0, 1], '--', color='gray'); ax.legend(fontsize=8)
    axes[0].set(title='Shared held-out spatial test set', xlabel='False positive rate', ylabel='True positive rate')
    axes[1].set(title='Reliability diagram (5 fixed bins)', xlabel='Mean predicted probability',
                ylabel='Observed positive-area fraction', xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout(); fig.savefig(out / 'roc_calibration.png', dpi=160); plt.close(fig)
    pd.concat(cal_tables).to_csv(out / 'calibration_bins.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ['roc_auc', 'f1', 'brier']):
        ax.bar(results.algorithm, results['test_' + metric], color='#377c75')
        ax.set(title=f'Test {metric}', ylim=(0, 1)); ax.tick_params(axis='x', rotation=65)
    fig.tight_layout(); fig.savefig(out / 'comparison.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(ytest, predictions['BMA'] >= thresholds['BMA']),
                           display_labels=['Recorded area = 0', 'Recorded area > 0']).plot(ax=ax, colorbar=False)
    ax.set_title('BMA: validation-selected threshold')
    fig.tight_layout(); fig.savefig(out / 'bma_confusion.png', dpi=160); plt.close(fig)


def cluster_bootstrap(y, predictions, groups, thresholds, n_boot=500):
    """Resample test cells, not rows; same resamples for every algorithm.

    Intervals describe sampling variability conditional on the fitted experiment,
    not uncertainty from retraining or an unbiased new external validation.
    """
    rng = np.random.default_rng(2026)
    unique = np.unique(groups)
    samples = {name: {metric: [] for metric in ['roc_auc', 'f1', 'brier']}
               for name in predictions}
    for _ in range(n_boot):
        ids = np.concatenate([np.flatnonzero(groups == g) for g in rng.choice(unique, len(unique), replace=True)])
        if len(np.unique(y[ids])) < 2:
            continue
        for name, p in predictions.items():
            values = metrics(y[ids], p[ids], thresholds[name])
            for metric in samples[name]:
                samples[name][metric].append(values[metric])
    return pd.DataFrame([{'algorithm': name, 'metric': metric,
                          'lower_95': float(np.quantile(values, .025)),
                          'upper_95': float(np.quantile(values, .975)),
                          'valid_resamples': len(values)}
                         for name, scores in samples.items() for metric, values in scores.items()])


def run_experiment(output_dir=None):
    out = Path(output_dir or ROOT / 'artifacts'); out.mkdir(parents=True, exist_ok=True)
    frame = load_data(); X = feature_frame(frame); y = (frame.area > 0).astype(int)
    splits, groups = grouped_split(frame)
    tr, va, te = (splits[k] for k in ['train', 'validation', 'test'])
    assignments = pd.DataFrame({'row_id': np.arange(len(frame)), 'group': groups, 'split': ''})
    for name, ids in splits.items(): assignments.loc[ids, 'split'] = name
    assignments.to_csv(out / 'split_assignments.csv', index=False)
    cv = list(GroupKFold(n_splits=4).split(X.iloc[tr], y.iloc[tr], groups.iloc[tr]))
    for a, b in cv:
        if y.iloc[tr[a]].nunique() != 2 or y.iloc[tr[b]].nunique() != 2:
            raise ValueError('Inner grouped CV lacks a class.')
    specs = {
        'BMA': (BayesianModelAverage(), {'model__prior_scale': [.15, .35, .7, 1.0, 2.5],
                                        'model__inclusion_prior': [.2, .5, .8]}),
        'Logistic Regression': (LogisticRegression(max_iter=2000, random_state=42), {'model__C': [.001, .01, .1, 1, 10, 100]}),
        'Decision Tree': (DecisionTreeClassifier(random_state=42),
                          {'model__max_depth': [1, 2, 4, 6], 'model__min_samples_leaf': [5, 15, 30],
                           'model__ccp_alpha': [0.0, .01]}),
        'Random Forest': (RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1),
                          {'model__max_depth': [3, None], 'model__min_samples_leaf': [5, 15, 30],
                           'model__max_features': [.5, 1.0]}),
        'Prevalence Baseline': (DummyClassifier(strategy='prior'), {})}
    models, thresholds, predictions, rows, tuning = {}, {}, {}, [], {}
    for name, (model, grid) in specs.items():
        print(f'Fitting {name}...', flush=True)
        pipeline = Pipeline([('scale', StandardScaler()), ('model', model)])
        search = GridSearchCV(pipeline, grid, scoring='neg_brier_score', cv=cv,
                              n_jobs=1, refit=True, error_score='raise')
        start = perf_counter(); search.fit(X.iloc[tr], y.iloc[tr]); elapsed = perf_counter() - start
        fitted = search.best_estimator_; models[name] = fitted
        pv = fitted.predict_proba(X.iloc[va])[:, 1]
        candidates = np.linspace(.05, .95, 91)
        f1s = np.array([f1_score(y.iloc[va], pv >= t, zero_division=0) for t in candidates])
        # Tie: closest threshold to 0.5. Validation labels never enter training/model evidence.
        best = np.flatnonzero(np.isclose(f1s, f1s.max()))
        threshold = float(candidates[best[np.argmin(abs(candidates[best] - .5))]])
        thresholds[name] = threshold
        ptest = fitted.predict_proba(X.iloc[te])[:, 1]; predictions[name] = ptest
        # Warm process-local batch timing, averaged over 30 calls. Hardware-dependent.
        start = perf_counter()
        for _ in range(30): fitted.predict_proba(X.iloc[te])
        infer_ms = (perf_counter() - start) / 30 / len(te) * 1000
        row = {'algorithm': name, 'cv_brier': float(-search.best_score_),
               'threshold': threshold, 'training_seconds_including_search': elapsed,
               'selected_fit_seconds': float(search.refit_time_),
               'inference_ms_per_row_batch': infer_ms, 'interpretability': INTERPRETABILITY[name]}
        for prefix, yy, pp in [('validation', y.iloc[va], pv), ('test', y.iloc[te], ptest)]:
            row.update({prefix + '_' + k: v for k, v in metrics(yy, pp, threshold).items()})
        rows.append(row); tuning[name] = search.best_params_
        pd.DataFrame(search.cv_results_).to_csv(out / (name.lower().replace(' ', '_') + '_cv.csv'), index=False)
    results = pd.DataFrame(rows); results.to_csv(out / 'model_comparison.csv', index=False)
    weights = models['BMA'].named_steps['model'].weight_table()
    weights.to_csv(out / 'bma_weights.csv', index=False)
    inclusion = pd.DataFrame([{'block': block, 'inclusion_probability': float(sum(
        w for m, w in zip(models['BMA'].named_steps['model'].models_, models['BMA'].named_steps['model'].weights_)
        if block in m['name'].split(' + ')))} for block in ['weather', 'moisture', 'fire_indices', 'season']])
    inclusion.to_csv(out / 'block_inclusion.csv', index=False)
    pred = pd.DataFrame({'row_id': te, 'y_true': y.iloc[te].to_numpy(), 'group': groups.iloc[te].to_numpy()})
    for name, p in predictions.items(): pred[name] = p
    pred.to_csv(out / 'test_predictions.csv', index=False)
    build_plots(out, results, predictions, weights, y.iloc[te].to_numpy(), thresholds)
    cluster_bootstrap(y.iloc[te].to_numpy(), predictions, groups.iloc[te].to_numpy(), thresholds).to_csv(
        out / 'test_metric_intervals.csv', index=False)
    summary = {name: {'records': len(ids), 'spatial_groups': int(groups.iloc[ids].nunique()),
                      'positive_fraction': float(y.iloc[ids].mean())} for name, ids in splits.items()}
    metadata = {'target': 'P(recorded area > 0 | recorded fire, predictors); proxy, not ignition probability',
                'split_seed': 42, 'split_summary': summary, 'rows': len(frame),
                'data_sha256': sha256(ROOT / 'data/forestfires.csv'), 'best_parameters': tuning,
                'python': platform.python_version(), 'versions': {p: version(p) for p in
                ['numpy', 'pandas', 'scipy', 'scikit-learn', 'streamlit', 'joblib']},
                'threshold_protocol': 'Validation F1 grid search; validation F1 is selection-biased',
                'validation_brier_protocol': 'Untouched by hyperparameter search; no calibration fitting',
                'model_selection': 'Expanded prespecified grids; four-fold training-only grouped CV minimizes Brier. No test-based selection.',
                'test_reuse_note': 'This is a revision of a repository with previously published test scores; the fixed test set is not a fresh external validation.',
                'interval_protocol': '500 spatial-group bootstrap resamples, seed 2026; percentile 95% intervals, no model refitting.',
                'uncertainty': 'Laplace evidence and coefficient posteriors are approximations; weights conditional on candidate family.'}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    bundle = {'models': models, 'thresholds': thresholds, 'metadata': metadata,
              'training_ranges': frame.iloc[tr][RAW_FEATURES[:-1]].agg(['min', 'max']).to_dict(),
              'training_months': sorted(frame.iloc[tr].month.unique().tolist()), 'features': FEATURES}
    joblib.dump(bundle, out / 'models.joblib')
    print(results[['algorithm', 'validation_brier', 'test_roc_auc', 'test_f1', 'test_brier']].to_string(index=False))
    return results


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(); run_experiment(args.output_dir)


if __name__ == '__main__': main()
