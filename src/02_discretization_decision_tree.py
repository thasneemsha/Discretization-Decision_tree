"""
02_discretization_decision_tree.py  –  CS334 Data Mining Project
=================================================================
Technique 2 : Data → Discretization → Decision Tree → Prediction

What this script does
---------------------
1. Loads the NIR calibration and validation spectra.
2. Reduces the 4200-wavelength spectra to 10 features using PCA
   (Principal Component Analysis) so the decision tree has manageable input.
3. Discretizes the continuous glucose values into 3 named classes:
       Low (0–10 mM)  |  Normal (10–30 mM)  |  High (30–50 mM)
4. Trains a Decision Tree classifier on the discretized labels.
5. Evaluates it on the validation set (accuracy, confusion matrix).
6. Visualises the learned tree so you can read every rule it uses.

Why discretize?
---------------
Decision trees natively split on "feature < threshold", which works with
continuous targets but can produce very deep trees.  By binning glucose
into Low / Normal / High first we get a clean 3-class problem that maps
directly onto what a clinician cares about: is the patient hypoglycaemic,
normal, or hyperglycaemic?
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_dataset, discretize_glucose

print("=" * 60)
print("TECHNIQUE 2 : DISCRETIZATION → DECISION TREE")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 – Load data
# ═══════════════════════════════════════════════════════════════════════════════

data = load_dataset()
X_cal      = data['X_cal']       # (273, 4200) absorbance matrix
y_cal      = data['y_cal']       # (273,)  continuous glucose mM
X_val      = data['X_val']       # (120, 4200)
y_val      = data['y_val']       # (120,)  continuous glucose mM


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 – Dimensionality reduction with PCA
# ═══════════════════════════════════════════════════════════════════════════════
#
# We have 4200 features (wavelengths) but only 273 training samples.
# A decision tree trained on all 4200 features would:
#   a) be impossibly slow to train, and
#   b) easily overfit the training data.
#
# PCA finds new axes (principal components) that capture the most variance
# in the data.  The first 10 PCs already explain >95 % of the spectral
# variation, so we compress 4200 → 10 without losing much information.

print("\n[Step 2] Reducing 4200 wavelengths → 10 PCA components …")

N_COMPONENTS = 10

pca = PCA(n_components=N_COMPONENTS, random_state=42)

# IMPORTANT: fit the PCA only on calibration data, then apply to validation.
# If we included validation data in the fit we would be "peeking" at test data.
X_cal_pca = pca.fit_transform(X_cal)    # shape (273, 10)
X_val_pca = pca.transform(X_val)        # shape (120, 10)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

print(f"  Variance explained by each PC:")
for i, (var, cum) in enumerate(zip(explained, cumulative)):
    print(f"    PC{i+1}: {var*100:5.2f}%   cumulative: {cum*100:5.2f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 – Discretize glucose into 3 classes
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[Step 3] Discretizing glucose concentrations …")
print("  Bins: 0–10 mM → 'Low'  |  10–30 mM → 'Normal'  |  30–50 mM → 'High'")

y_cal_class = discretize_glucose(y_cal)     # array of strings: 'Low','Normal','High'
y_val_class = discretize_glucose(y_val)

# Show class distribution in calibration set
from collections import Counter
cal_counts = Counter(y_cal_class)
val_counts = Counter(y_val_class)
print(f"\n  Calibration class counts : {dict(cal_counts)}")
print(f"  Validation  class counts : {dict(val_counts)}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 – How a Decision Tree works (the math)
# ═══════════════════════════════════════════════════════════════════════════════
#
# At each internal node the tree asks "Is feature_k < threshold?"
# and routes samples left (Yes) or right (No).
#
# The tree picks the BEST split by maximising Information Gain:
#
#   Gain(S, A) = H(S) − Σ  |S_v| / |S| * H(S_v)
#                         v
#
# where H(S) = Gini Impurity (or Entropy) of set S:
#
#   Gini(S) = 1 − Σ  p_k²          (what sklearn uses by default)
#                  k
#
#   Entropy(S) = − Σ  p_k * log2(p_k)
#                   k
#
# p_k = fraction of samples in class k within set S.
# A pure node (all same class) has Gini = 0.  Mixed node has Gini up to 0.67
# for a 3-class problem.
#
# The tree grows until max_depth is reached or nodes are pure.

print("\n[Step 4] Training Decision Tree classifier …")

CLASS_ORDER = ['Low', 'Normal', 'High']    # consistent ordering for plots

clf = DecisionTreeClassifier(
    max_depth=5,          # limit tree depth to prevent overfitting
    criterion='gini',     # use Gini impurity to choose splits
    min_samples_leaf=5,   # each leaf must have at least 5 training samples
    random_state=42
)

clf.fit(X_cal_pca, y_cal_class)

print(f"  Tree depth    : {clf.get_depth()}")
print(f"  Number of leaves : {clf.get_n_leaves()}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 – Print the decision rules in text form
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[Step 5] Decision rules learned by the tree:")
print("-" * 50)
tree_text = export_text(clf,
    feature_names=[f'PC{i+1}' for i in range(N_COMPONENTS)],
    max_depth=3)          # only show top 3 levels for readability
print(tree_text)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 – Evaluate on validation set
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[Step 6] Evaluating on validation set …")

y_pred_val = clf.predict(X_val_pca)

acc = accuracy_score(y_val_class, y_pred_val)
print(f"  Validation Accuracy : {acc * 100:.1f}%")

print("\n  Classification Report:")
print(classification_report(y_val_class, y_pred_val,
                             target_names=CLASS_ORDER, zero_division=0))

cm = confusion_matrix(y_val_class, y_pred_val, labels=CLASS_ORDER)
print("  Confusion Matrix (rows=actual, cols=predicted):")
print(f"  {'':>10}  {'Low':>8}  {'Normal':>8}  {'High':>8}")
for label, row in zip(CLASS_ORDER, cm):
    print(f"  {label:>10}  {row[0]:>8}  {row[1]:>8}  {row[2]:>8}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 – Visualise
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[Step 7] Saving plots …")

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle("Technique 2 – Discretization + Decision Tree",
             fontsize=13, fontweight='bold')

# ── Left: the tree diagram ────────────────────────────────────────────────────
ax = axes[0]
plot_tree(
    clf,
    feature_names=[f'PC{i+1}' for i in range(N_COMPONENTS)],
    class_names=CLASS_ORDER,
    filled=True,            # colour nodes by majority class
    rounded=True,
    max_depth=3,            # only draw top 3 levels (keeps it readable)
    fontsize=7,
    ax=ax
)
ax.set_title('Learned Decision Tree (top 3 levels)', fontsize=11)

# ── Right: confusion matrix heat-map ─────────────────────────────────────────
ax = axes[1]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_ORDER)
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Confusion Matrix (Validation)\nAccuracy = {acc*100:.1f}%',
             fontsize=11)

plt.tight_layout()
plt.savefig('results/02_decision_tree_results.png', dpi=150)
plt.close()
print("  Saved → 02_decision_tree_results.png")

# ── PCA variance bar chart ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(1, N_COMPONENTS + 1), explained * 100, color='steelblue',
       edgecolor='white')
ax.plot(range(1, N_COMPONENTS + 1), cumulative * 100, 'ro-',
        markersize=6, label='Cumulative')
ax.axhline(95, color='gray', linestyle='--', linewidth=1,
           label='95 % threshold')
ax.set_xlabel('Principal Component', fontsize=11)
ax.set_ylabel('Explained Variance (%)', fontsize=11)
ax.set_title('PCA: Variance Explained per Component', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/02_pca_variance.png', dpi=150)
plt.close()
print("  Saved → 02_pca_variance.png")

print("\n✓ Technique 2 complete.")