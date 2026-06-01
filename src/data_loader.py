"""
data_loader.py  –  CS334 Data Mining Project
=============================================
Shared utility: loads and cleans the NIR glucose dataset.
All technique scripts import from here.

Dataset facts (from README):
  • Calibration : 273 samples
  • Validation  : 120 samples
  • 9 info columns  → Glucose, Lactate, Acetaminophen, Caffeine,
                       Ethanol, Temperature, Cuvette, Day, Run
  • 4200 wavelength columns  → 400 nm to 2499.5 nm (step 0.5 nm)
  • Files are UTF-16 LE encoded, tab-separated
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Base project folder — works regardless of which script imports this
BASE_DIR = Path(__file__).resolve().parent

# ── file paths ──────────────────────────────────────────────────────────────
CAL_PATH = BASE_DIR / 'data' / 'CalibrationData.txt'
VAL_PATH = BASE_DIR / 'data' / 'ValidationData.txt'

# ── column name constants ────────────────────────────────────────────────────
INFO_COLS = [
    'Glucose (mM)',
    'Lactate (mM)',
    'Acetaminophen (mM)',
    'Caffeine (mM)',
    'Ethanol (mM)',
    'Temperature (C)',
    'Cuvette',
    'Day',
    'Run',
]

# Glucose level bins (used across all techniques)
#   Low   : 0 – 10 mM   (hypoglycaemia range)
#   Normal: 10 – 30 mM
#   High  : 30 – 50 mM
GLUCOSE_BINS   = [0, 10, 30, 50]
GLUCOSE_LABELS = ['Low', 'Normal', 'High']

# Clinical cut-points (used by techniques 02, 03, 04)
#   Hypo    :      < 4 mM
#   Normal  :  4 – 7 mM
#   Elevated:  7 – 15 mM
#   High    :    >= 15 mM
CLINICAL_BINS   = [4, 7, 15]
CLINICAL_LABELS = ['Hypo', 'Normal', 'Elevated', 'High']


def load_raw(path, max_real_cols=4209):
    """
    Read the raw spectrometer file and drop junk columns.

    Parameters
    ----------
    path          : Path or str – path to the .txt file
    max_real_cols : int – keep only the first N columns (default 4209)

    Returns
    -------
    df : pd.DataFrame with clean column names
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Data file not found: {path}\n"
            f"        Please place CalibrationData.txt and ValidationData.txt\n"
            f"        inside:  {path.parent}\n"
        )

    df = pd.read_csv(path, encoding='utf-16', sep='\t', header=0)
    df = df.iloc[:, :max_real_cols]
    df.columns = [c.strip() for c in df.columns]
    return df


def _detect_cuvette_col(df):
    """Handle both 'Cuvette' and 'Kuvette' spellings in the source files."""
    for name in ('Cuvette', 'Kuvette', 'cuvette', 'kuvette'):
        if name in df.columns:
            return name
    return None


def load_dataset():
    """
    Load, clean, and split the NIR dataset.

    Returns
    -------
    dict with keys:
      X_cal, y_cal, info_cal   – calibration arrays / DataFrame
      X_val, y_val, info_val   – validation arrays / DataFrame
      wavelengths               – (4200,) array of nm values
      df_cal, df_val            – full raw DataFrames (for column access)
    """
    print("Loading calibration data …")
    df_cal = load_raw(CAL_PATH)

    print("Loading validation data …")
    df_val = load_raw(VAL_PATH)

    # ── Rename first 9 columns to canonical INFO_COLS names ─────────────────
    # (handles Cuvette / Kuvette variant automatically)
    def _rename(df):
        raw9 = list(df.columns[:9])
        rename_map = dict(zip(raw9, INFO_COLS))
        df = df.rename(columns=rename_map)
        return df

    df_cal = _rename(df_cal)
    df_val = _rename(df_val)

    # ── Separate info vs. spectra ────────────────────────────────────────────
    info_cal = df_cal[INFO_COLS].copy()
    info_val = df_val[INFO_COLS].copy()

    wave_cols   = df_cal.columns[9:].tolist()
    wavelengths = np.array([float(w) for w in wave_cols])

    X_cal = df_cal[wave_cols].values.astype(float)   # (273, 4200)
    X_val = df_val[wave_cols].values.astype(float)   # (120, 4200)

    y_cal = info_cal['Glucose (mM)'].values.astype(float)
    y_val = info_val['Glucose (mM)'].values.astype(float)

    # Drop rows with NaN glucose (none expected, but defensive)
    cal_ok = ~np.isnan(y_cal)
    val_ok = ~np.isnan(y_val)
    X_cal, y_cal = X_cal[cal_ok], y_cal[cal_ok]
    X_val, y_val = X_val[val_ok], y_val[val_ok]
    info_cal = info_cal[cal_ok].reset_index(drop=True)
    info_val = info_val[val_ok].reset_index(drop=True)

    print(f"  Calibration : {X_cal.shape[0]} samples, "
          f"{X_cal.shape[1]} wavelength points")
    print(f"  Validation  : {X_val.shape[0]} samples, "
          f"{X_val.shape[1]} wavelength points")
    print(f"  Wavelength  : {wavelengths[0]:.1f} – {wavelengths[-1]:.1f} nm\n")

    return dict(
        X_cal=X_cal, y_cal=y_cal, info_cal=info_cal,
        X_val=X_val, y_val=y_val, info_val=info_val,
        wavelengths=wavelengths,
        df_cal=df_cal, df_val=df_val,
        wave_cols=wave_cols,
    )


# ── Discretization helpers ───────────────────────────────────────────────────

def discretize_glucose_clinical(glucose_values):
    """
    Clinical cut-points → 4 class labels.
      < 4  → 'Hypo'
      4-7  → 'Normal'
      7-15 → 'Elevated'
      ≥15  → 'High'
    """
    labels = []
    for g in glucose_values:
        if g < 4:
            labels.append('Hypo')
        elif g < 7:
            labels.append('Normal')
        elif g < 15:
            labels.append('Elevated')
        else:
            labels.append('High')
    return labels


def discretize_feature_equal_freq(values, n_bins=3):
    """
    Equal-frequency binning for a continuous feature.

    Returns
    -------
    bin_labels : list of 'B0', 'B1', … strings
    cut_points : list of cut-point values
    """
    import numpy as np
    arr = np.array(values, dtype=float)
    percentiles = [100 * i / n_bins for i in range(1, n_bins)]
    cut_points = [float(np.percentile(arr, p)) for p in percentiles]

    def assign(v):
        return sum(1 for cp in cut_points if v > cp)

    return [f'B{assign(v)}' for v in arr], cut_points


def apply_cutpoints(value, cut_points):
    """Apply pre-computed cut-points to a single value → bin index."""
    return sum(1 for cp in cut_points if value > cp)


def top_correlated_wavelengths(X, y, wavelengths, n=20):
    """
    Return the n wavelengths most correlated with glucose (by Pearson |r|).

    Returns
    -------
    top_idx : indices into the 4200-column axis
    top_nm  : corresponding nm values
    top_r   : correlation coefficients
    """
    correlations = np.array([
        np.corrcoef(X[:, i], y)[0, 1]
        for i in range(X.shape[1])
    ])
    top_idx = np.argsort(np.abs(correlations))[-n:][::-1]
    return top_idx, wavelengths[top_idx], correlations[top_idx]