"""
technique_02_decision_tree.py  –  CS334 Data Mining Project
============================================================
Flow: Data → Discretization → Decision Tree → Prediction

"""

import math
import csv
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from data_loader import (
    load_dataset,
    discretize_glucose_clinical,
    discretize_feature_equal_freq,
    apply_cutpoints,
)
from plot import save_plots_02

RESULTS_DIR = Path(__file__).resolve().parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

TOP_K = 20
N_BINS = 4
MAX_DEPTH = 6
MIN_SAMPLES_SPLIT = 5
CLASSES = ['Hypo', 'Normal', 'Elevated', 'High']


def compute_class_weights(labels: List[str]) -> Dict[str, float]:
    """
    CORRECTED: Class weights = total_samples / (n_classes × count_c)
    Rare classes get HIGHER weights, common classes get LOWER weights.
    """
    n = len(labels)
    counts = Counter(labels)
    n_cls = len(counts)

    weights = {}
    for cls in CLASSES:
        cnt = counts.get(cls, 1)
        # Inverse frequency weight: rare classes get higher weight
        weights[cls] = n / (n_cls * cnt)

    # Normalize so weights sum to n_classes
    total = sum(weights.values())
    weights = {c: w / total * n_cls for c, w in weights.items()}

    return weights


def weighted_entropy(labels: List[str], class_weights: Dict[str, float]) -> float:
    """Weighted entropy: H(Y) = -Σ w_c · p(c) · log₂(p(c))"""
    n = len(labels)
    if n == 0:
        return 0.0

    counts = Counter(labels)
    h = 0.0
    for cls, cnt in counts.items():
        p = cnt / n
        w = class_weights.get(cls, 1.0)
        if p > 0:
            h -= w * p * math.log2(p)
    return h


def find_best_split(data: List[Dict], feature: str, target: str,
                    class_weights: Dict[str, float]) -> Tuple[str, float, Dict[str, List[Dict]]]:
    """Find best split using weighted entropy."""
    # Group by feature values
    groups: Dict[str, List[Dict]] = {}
    for row in data:
        val = row.get(feature, 'B0')
        groups.setdefault(val, []).append(row)

    if len(groups) <= 1:
        return None, 0.0, {}

    labels = [row[target] for row in data]
    h_parent = weighted_entropy(labels, class_weights)
    n_total = len(data)

    best_gain = 0.0
    best_groups = {}
    best_feature = None

    # Try each value as a potential split point
    for split_val, group in groups.items():
        if len(group) == 0 or len(group) == n_total:
            continue

        other_groups = {k: v for k, v in groups.items() if k != split_val}
        left_group = group
        right_group = []
        for g in other_groups.values():
            right_group.extend(g)

        if len(left_group) < MIN_SAMPLES_SPLIT or len(right_group) < MIN_SAMPLES_SPLIT:
            continue

        p_left = len(left_group) / n_total
        p_right = len(right_group) / n_total

        left_labels = [r[target] for r in left_group]
        right_labels = [r[target] for r in right_group]

        h_left = weighted_entropy(left_labels, class_weights)
        h_right = weighted_entropy(right_labels, class_weights)

        gain = h_parent - (p_left * h_left + p_right * h_right)

        if gain > best_gain:
            best_gain = gain
            best_feature = feature
            best_groups = {'left': left_group, 'right': right_group}

    return best_feature, best_gain, best_groups


class DecisionNode:
    def __init__(self, feature: str, split_value: str = None):
        self.feature = feature
        self.split_value = split_value
        self.children: Dict[str, Any] = {}


class LeafNode:
    def __init__(self, label: str, confidence: float = 1.0, distribution: Dict[str, float] = None):
        self.label = label
        self.confidence = confidence
        self.distribution = distribution or {}


def build_tree(data: List[Dict], features: List[str], target: str,
               class_weights: Dict[str, float], depth: int = 0,
               max_depth: int = MAX_DEPTH) -> Any:
    """Build decision tree with weighted entropy."""
    labels = [row[target] for row in data]

    # Check if all same class
    if len(set(labels)) == 1:
        return LeafNode(labels[0], confidence=1.0, distribution={labels[0]: 1.0})

    # Check max depth
    if depth >= max_depth or not features:
        counts = Counter(labels)
        n = len(labels)
        # Weighted majority
        weighted_counts = {c: cnt * class_weights.get(c, 1.0) for c, cnt in counts.items()}
        majority = max(weighted_counts, key=weighted_counts.get)
        confidence = counts[majority] / n
        dist = {c: cnt / n for c, cnt in counts.items()}
        return LeafNode(majority, confidence, dist)

    # Find best split
    best_feature = None
    best_gain = 0.0
    best_groups = {}

    for feature in features:
        _, gain, groups = find_best_split(data, feature, target, class_weights)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature
            best_groups = groups

    if best_feature is None or best_gain <= 0.001:
        counts = Counter(labels)
        majority = max(counts, key=counts.get)
        confidence = counts[majority] / len(labels)
        dist = {c: cnt / len(labels) for c, cnt in counts.items()}
        return LeafNode(majority, confidence, dist)

    # Create node and recurse
    node = DecisionNode(best_feature)
    remaining_features = [f for f in features if f != best_feature]

    for branch, subset in best_groups.items():
        if subset:
            node.children[branch] = build_tree(subset, remaining_features, target,
                                               class_weights, depth + 1, max_depth)
        else:
            # Empty branch - use parent distribution
            counts = Counter(labels)
            majority = max(counts, key=counts.get)
            node.children[branch] = LeafNode(majority, 0.5)

    return node


def predict(node: Any, sample: Dict[str, str]) -> Tuple[str, float, Dict[str, float]]:
    """Predict class for a sample."""
    if isinstance(node, LeafNode):
        return node.label, node.confidence, node.distribution

    val = sample.get(node.feature, 'B0')

    # Determine branch
    if val in node.children:
        return predict(node.children[val], sample)
    elif node.children:
        # Use first available child
        return predict(list(node.children.values())[0], sample)
    else:
        return 'High', 0.5, {'High': 1.0}


def select_top_features(X_cal: np.ndarray, y_train: List[str], wave_cols: List[str],
                        class_weights: Dict[str, float], top_k: int = TOP_K) -> Tuple[List[str], Dict[str, float]]:
    """Select top features by weighted information gain."""
    gains = {}

    for i, col in enumerate(wave_cols):
        values = X_cal[:, i].tolist()
        labels = y_train

        # Simple threshold split at median
        median = np.median(values)
        left_labels = [lbl for v, lbl in zip(values, labels) if v <= median]
        right_labels = [lbl for v, lbl in zip(values, labels) if v > median]

        if len(left_labels) == 0 or len(right_labels) == 0:
            gains[col] = 0
            continue

        h_parent = weighted_entropy(labels, class_weights)
        p_left = len(left_labels) / len(labels)
        p_right = len(right_labels) / len(labels)
        h_left = weighted_entropy(left_labels, class_weights)
        h_right = weighted_entropy(right_labels, class_weights)

        gain = h_parent - (p_left * h_left + p_right * h_right)
        gains[col] = gain

    sorted_feats = sorted(gains, key=gains.get, reverse=True)
    return sorted_feats[:top_k], gains


def main():
    print("=" * 60)
    print("  TECHNIQUE 02 — Weighted Decision Tree (FIXED)")
    print("  Flow: Data → Discretization → Decision Tree → Prediction")
    print("=" * 60)

    # Load data
    data = load_dataset()
    df_cal = data['df_cal']
    df_val = data['df_val']
    X_cal = data['X_cal']
    wave_cols = data['wave_cols']

    # Discretize target
    y_train = discretize_glucose_clinical(data['y_cal'])
    y_test = discretize_glucose_clinical(data['y_val'])

    # Compute class weights (CORRECTED)
    class_weights = compute_class_weights(y_train)
    print("\n  Class Weights (higher = more important):")
    for cls, w in class_weights.items():
        print(f"    {cls:10s}: {w:.3f}")

    # Feature selection
    best_features, gains = select_top_features(X_cal, y_train, wave_cols, class_weights, TOP_K)
    print(f"\n  Selected {len(best_features)} features")

    # Compute cutpoints and discretize
    feature_cutpoints = {}
    for col in best_features:
        vals = df_cal[col].values.astype(float).tolist()
        _, cps = discretize_feature_equal_freq(vals, n_bins=N_BINS)
        feature_cutpoints[col] = cps

    # Build training data
    X_train_disc = []
    for _, row in df_cal.iterrows():
        rec = {}
        for col in best_features:
            val = float(row[col])
            rec[col] = f'B{apply_cutpoints(val, feature_cutpoints[col])}'
        X_train_disc.append(rec)

    X_test_disc = []
    for _, row in df_val.iterrows():
        rec = {}
        for col in best_features:
            if col in df_val.columns:
                val = float(row[col])
                rec[col] = f'B{apply_cutpoints(val, feature_cutpoints[col])}'
        X_test_disc.append(rec)

    # Add target
    TARGET = 'GlucoseClass'
    for i, rec in enumerate(X_train_disc):
        rec[TARGET] = y_train[i]

    # Build tree
    tree = build_tree(X_train_disc, best_features, TARGET, class_weights, max_depth=MAX_DEPTH)

    # Evaluate
    results = []
    for sample, true_label in zip(X_test_disc, y_test):
        pred, conf, dist = predict(tree, sample)
        results.append({'true': true_label, 'pred': pred, 'correct': pred == true_label})

    correct = sum(r['correct'] for r in results)
    accuracy = correct / len(results)

    print(f"\n  Overall Accuracy: {accuracy * 100:.1f}%")
    print(f"\n  Per-Class Performance:")
    for cls in CLASSES:
        cls_results = [r for r in results if r['true'] == cls]
        if cls_results:
            cls_correct = sum(r['correct'] for r in cls_results)
            cls_acc = cls_correct / len(cls_results) * 100
            print(f"    {cls:10s}: {cls_correct}/{len(cls_results)} = {cls_acc:.1f}%")

    # Depth analysis
    depth_results = []
    for d in range(1, 9):
        test_tree = build_tree(X_train_disc, best_features, TARGET, class_weights, max_depth=d)
        train_correct = sum(1 for rec in X_train_disc if predict(test_tree, rec)[0] == rec[TARGET])
        val_correct = sum(1 for rec, true in zip(X_test_disc, y_test) if predict(test_tree, rec)[0] == true)
        depth_results.append((d, train_correct / len(X_train_disc) * 100, val_correct / len(X_test_disc) * 100))

    # Save
    out_path = RESULTS_DIR / 'technique_02_results.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample', 'true', 'pred', 'correct'])
        for i, r in enumerate(results):
            writer.writerow([i + 1, r['true'], r['pred'], 'YES' if r['correct'] else 'NO'])
        writer.writerow([f'Overall Accuracy,{accuracy * 100:.1f}%'])

    # Generate plots
    save_plots_02(y_train, y_test, results, best_features, gains, depth_results)

    print("\n" + "=" * 60)
    print("  TECHNIQUE 02 COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()