"""
plots.py  –  CS334 Data Mining Project
=======================================
Enhanced plotting module with better visualizations for all techniques.
"""

from pathlib import Path
from collections import Counter
import math

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

RESULTS_DIR = Path(__file__).resolve().parent / 'results'
CLASSES = ['Hypo', 'Normal', 'Elevated', 'High']
CLASS_COLORS = {
    'Hypo': '#3B82F6',
    'Normal': '#22C55E',
    'Elevated': '#F59E0B',
    'High': '#EF4444',
}


def save_plots_02(y_train, y_test, results, best_features, gains, depth_results):
    """Decision Tree plots."""
    if not MATPLOTLIB_OK:
        return

    prefix = 'tech02_dt'

    # Plot 1: Class distribution
    _plot_class_distribution(y_train, y_test, prefix, "Decision Tree")

    # Plot 2: Per-class accuracy
    _plot_per_class_accuracy(results, prefix, "Decision Tree")

    # Plot 3: Confusion matrix
    _plot_confusion_matrix(results, prefix, "Decision Tree")

    # Plot 4: Feature importance (information gain)
    _plot_feature_importance(best_features, gains, prefix)

    # Plot 5: Depth vs accuracy
    if depth_results:
        _plot_depth_vs_accuracy(depth_results, prefix)


def save_plots_03(y_train, y_test, results, all_patterns, scores_list):
    """Association Pattern plots."""
    if not MATPLOTLIB_OK:
        return

    prefix = 'tech03_ap'

    # Plot 1: Class distribution
    _plot_class_distribution(y_train, y_test, prefix, "Association Patterns")

    # Plot 2: Per-class accuracy
    _plot_per_class_accuracy(results, prefix, "Association Patterns")

    # Plot 3: Confusion matrix
    _plot_confusion_matrix(results, prefix, "Association Patterns")

    # Plot 4: Pattern analysis by class
    _plot_pattern_analysis(all_patterns, prefix)

    # Plot 5: Model score comparison
    if scores_list:
        _plot_model_scores(scores_list, [r['true'] for r in results[:20]], prefix)


def save_plots_04(y_train, y_test, results, mi_values, model_uni, model_emp,
                  df_val, glucose_test, valid_waves, feature_cutpoints):
    """Bayesian Network plots."""
    if not MATPLOTLIB_OK:
        return

    prefix = 'tech04_bn'

    # Plot 1: Class distribution
    _plot_class_distribution(y_train, y_test, prefix, "Bayesian Network")

    # Plot 2: Per-class accuracy
    _plot_per_class_accuracy(results, prefix, "Bayesian Network")

    # Plot 3: Confusion matrix
    _plot_confusion_matrix(results, prefix, "Bayesian Network")

    # Plot 4: MI independence analysis
    _plot_mi_analysis(mi_values, valid_waves[:15] if valid_waves else list(mi_values.keys())[:15], prefix)

    # Plot 5: Prior comparison
    _plot_prior_comparison(model_uni, model_emp, df_val, glucose_test, valid_waves, feature_cutpoints, prefix)


def _plot_class_distribution(y_train, y_test, prefix, technique_name):
    """Class distribution bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{technique_name}: Glucose Class Distribution', fontsize=14, fontweight='bold')

    for ax, labels, split in zip(axes, [y_train, y_test], ['Training (n={})', 'Validation (n={})']):
        counts = Counter(labels)
        vals = [counts.get(c, 0) for c in CLASSES]
        colors = [CLASS_COLORS[c] for c in CLASSES]
        bars = ax.bar(CLASSES, vals, color=colors, edgecolor='white', linewidth=1.5)

        total = sum(vals)
        for bar, v in zip(bars, vals):
            pct = v / total * 100 if total > 0 else 0
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{v}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title(split.format(total), fontsize=11)
        ax.set_ylabel('Number of Samples')
        ax.set_ylim(0, max(vals) * 1.2)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{prefix}_1_class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_per_class_accuracy(results, prefix, technique_name):
    """Per-class accuracy bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    cls_acc = {}
    cls_total = {}
    for cls in CLASSES:
        cls_r = [r for r in results if r['true'] == cls]
        if cls_r:
            cls_acc[cls] = sum(r['correct'] for r in cls_r) / len(cls_r) * 100
            cls_total[cls] = len(cls_r)

    x = range(len(CLASSES))
    vals = [cls_acc.get(c, 0) for c in CLASSES]
    colors = [CLASS_COLORS[c] for c in CLASSES]
    bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=1.5)

    # Baseline (majority class performance)
    high_acc = cls_acc.get('High', 0)
    ax.axhline(high_acc, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(len(CLASSES) - 0.5, high_acc + 1, f'High baseline: {high_acc:.0f}%',
            ha='right', fontsize=9, color='gray')

    for bar, cls, v in zip(bars, CLASSES, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2,
                f'{v:.0f}%\n(n={cls_total.get(cls, 0)})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title(f'{technique_name}: Per-Class Accuracy', fontsize=13, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{prefix}_2_per_class_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_confusion_matrix(results, prefix, technique_name):
    """Confusion matrix heatmap."""
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    cls_idx = {c: i for i, c in enumerate(CLASSES)}

    for r in results:
        if r['true'] in cls_idx and r['pred'] in cls_idx:
            matrix[cls_idx[r['true']]][cls_idx[r['pred']]] += 1

    fig, ax = plt.subplots(figsize=(8, 7))

    cmap = LinearSegmentedColormap.from_list('cm', ['#f0f9ff', '#0ea5e9', '#0369a1'])
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')

    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, fontsize=10)
    ax.set_yticklabels(CLASSES, fontsize=10)
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    ax.set_title(f'{technique_name}: Confusion Matrix', fontsize=13, fontweight='bold')

    max_val = matrix.max() or 1
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            v = matrix[i, j]
            color = 'white' if v > max_val * 0.5 else '#1e293b'
            diag = ' ✓' if i == j else ''
            ax.text(j, i, f'{v}{diag}', ha='center', va='center', fontsize=11, fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label='Count')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{prefix}_3_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_feature_importance(best_features, gains, prefix, n=15):
    """Feature importance bar chart for Decision Tree."""
    top_feats = best_features[:n]
    top_gains = [gains.get(f, 0) for f in top_feats]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, n))[::-1]
    bars = ax.barh(range(n), top_gains, color=colors, edgecolor='white')

    ax.set_yticks(range(n))
    ax.set_yticklabels([f'{f} nm' for f in top_feats], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Weighted Information Gain', fontsize=11)
    ax.set_title('Decision Tree: Top 15 Features by Information Gain', fontsize=13, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

    for bar, v in zip(bars, top_gains):
        ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2, f'{v:.4f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{prefix}_4_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_depth_vs_accuracy(depth_results, prefix):
    """Tree depth vs accuracy plot."""
    depths, train_accs, val_accs = zip(*depth_results)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(depths, train_accs, 'o-', color='#3B82F6', linewidth=2, markersize=8, label='Training Accuracy')
    ax.plot(depths, val_accs, 's-', color='#EF4444', linewidth=2, markersize=8, label='Validation Accuracy')

    best_depth = depths[np.argmax(val_accs)]
    best_acc = max(val_accs)
    ax.axvline(best_depth, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(best_depth + 0.2, best_acc - 5, f'Best depth: {best_depth}\n({best_acc:.1f}%)', fontsize=9, color='green')

    ax.set_xlabel('Tree Depth (max_depth)', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Decision Tree: Depth vs Accuracy Trade-off', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks(depths)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{prefix}_5_depth_vs_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_pattern_analysis(all_patterns, prefix):
    """Pattern analysis by class for Association Patterns."""
    patterns_by_class = {cls: [] for cls in CLASSES}
    for cond, supp, norm_supp, mi in all_patterns:
        cls = cond.get('GlucoseClass')
        if cls in patterns_by_class:
            patterns_by_class[cls].append((norm_supp, mi))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Association Patterns: Quality by Class', fontsize=13, fontweight='bold')

    # Left: MI distribution
    ax = axes[0]
    for cls in CLASSES:
        mis = [mi for _, mi in patterns_by_class[cls]]
        if mis:
            ax.scatter([cls] * len(mis), mis, color=CLASS_COLORS[cls], alpha=0.6, s=40)
            ax.plot([cls, cls], [min(mis), max(mis)], color=CLASS_COLORS[cls], linewidth=1.5, alpha=0.4)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_ylabel('Mutual Information (MI)', fontsize=11)
    ax.set_title('Pattern Strength Distribution', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)

    # Right: Pattern count
    ax2 = axes[1]
    counts = [len(patterns_by_class[cls]) for cls in CLASSES]
    bars = ax2.bar(CLASSES, counts, color=[CLASS_COLORS[c] for c in CLASSES], edgecolor='white')

    for bar, cnt in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(cnt),
                 ha='center', fontsize=10, fontweight='bold')

    ax2.set_ylabel('Number of Patterns', fontsize=11)
    ax2.set_title('Pattern Count by Class', fontsize=11)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{prefix}_4_pattern_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_model_scores(scores_list, y_true, prefix, n_show=15):
    """Model scores per sample for Association Patterns."""
    scores_list = scores_list[:n_show]
    y_true = y_true[:n_show]
    x = np.arange(len(scores_list))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, cls in enumerate(CLASSES):
        vals = [s.get(cls, 0) for s in scores_list]
        ax.bar(x + i * width, vals, width, color=CLASS_COLORS[cls], alpha=0.85, label=cls, edgecolor='white')

    # Mark true class
    for j, true_cls in enumerate(y_true):
        ymax = max(s.get(c, 0) for s in [scores_list[j]] for c in CLASSES)
        ax.text(j + CLASSES.index(true_cls) * width + width / 2, ymax + 0.02, '★',
                ha='center', fontsize=12, color='black')

    ax.set_xlabel('Validation Sample Index', fontsize=11)
    ax.set_ylabel('Model Score', fontsize=11)
    ax.set_title('Association Patterns: Per-Sample Class Scores', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(range(1, len(scores_list) + 1), fontsize=8)
    ax.legend(title='Class', fontsize=9, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{prefix}_5_model_scores.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_mi_analysis(mi_values, wavelengths, prefix):
    """MI analysis for Bayesian Network."""
    mi_list = [mi_values.get(w, 0) for w in wavelengths]

    fig, ax = plt.subplots(figsize=(12, 5))

    colors = ['#22C55E' if m > 0.03 else '#94A3B8' for m in mi_list]
    bars = ax.bar(range(len(wavelengths)), mi_list, color=colors, edgecolor='white')

    ax.axhline(0.03, color='#EF4444', linestyle='--', linewidth=1.5, label='Dependency threshold (MI=0.03)')

    for bar, v in zip(bars, mi_list):
        if v > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Wavelength (nm)', fontsize=11)
    ax.set_ylabel('Mutual Information with Glucose Class', fontsize=11)
    ax.set_title('Bayesian Network: Independence Analysis', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(wavelengths)))
    ax.set_xticklabels([f'{w} nm' for w in wavelengths], fontsize=9, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{prefix}_4_mi_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_prior_comparison(model_uni, model_emp, df_val, glucose_test, valid_waves, feature_cutpoints, prefix):
    """Prior comparison for Bayesian Network."""
    from data_loader import apply_cutpoints

    def evaluate_with_prior(cpts, parents, wavelengths):
        correct = {c: 0 for c in CLASSES}
        total = {c: 0 for c in CLASSES}

        for i in range(min(len(df_val), 50)):  # Use subset for speed
            obs = {}
            for wave in wavelengths:
                if wave in df_val.columns:
                    val = float(df_val[wave].iloc[i])
                    obs[wave] = f'B{apply_cutpoints(val, feature_cutpoints.get(wave, []))}'

            # Simple inference
            log_probs = {cls: math.log(cpts['GlucoseClass'][cls]) for cls in CLASSES}
            for wave in wavelengths:
                obs_val = obs.get(wave, 'B0')
                if wave in cpts:
                    for cls in CLASSES:
                        if isinstance(cpts[wave], dict) and cls in cpts[wave]:
                            prob = cpts[wave][cls].get(obs_val, 0.01)
                            log_probs[cls] += math.log(max(prob, 1e-10))

            max_log = max(log_probs.values())
            probs = {cls: math.exp(lp - max_log) for cls, lp in log_probs.items()}
            total_sum = sum(probs.values())
            if total_sum > 0:
                pred = max(probs, key=probs.get)
            else:
                pred = 'High'

            true_cls = glucose_test[i]
            total[true_cls] += 1
            if pred == true_cls:
                correct[true_cls] += 1

        return {c: (correct[c] / total[c] * 100 if total[c] > 0 else 0) for c in CLASSES}

    # Extract from model tuples
    cpts_uni, parents_uni, waves_uni = model_uni
    cpts_emp, parents_emp, waves_emp = model_emp

    acc_uni = evaluate_with_prior(cpts_uni, parents_uni, waves_uni)
    acc_emp = evaluate_with_prior(cpts_emp, parents_emp, waves_emp)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(CLASSES))
    width = 0.35

    bars1 = ax.bar(x - width / 2, [acc_emp.get(c, 0) for c in CLASSES], width,
                   label='Empirical Prior (biased)', color='#94A3B8', edgecolor='white')
    bars2 = ax.bar(x + width / 2, [acc_uni.get(c, 0) for c in CLASSES], width,
                   label='Uniform Prior (fixed)', color='#3B82F6', edgecolor='white')

    for bar, acc in zip(bars1, [acc_emp.get(c, 0) for c in CLASSES]):
        if acc > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{acc:.0f}%',
                    ha='center', fontsize=9, fontweight='bold')

    for bar, acc in zip(bars2, [acc_uni.get(c, 0) for c in CLASSES]):
        if acc > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{acc:.0f}%',
                    ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title('Bayesian Network: Prior Distribution Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{prefix}_5_prior_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()