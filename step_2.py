"""
HiveDrive Step-2: Event windows → Features → RandomForest → Evaluate + PPT plots

What it produces (PNG for PPT):
1) tripX_04_window_examples.png        (pothole window vs non-pothole window)
2) rf_feature_importances.png          (which features matter)
3) rf_confusion_matrix.png             (pothole vs non-pothole classification)
4) tripX_05_baseline_vs_rf_events.png  (event-level precision/recall comparison)

Notes:
- Your dataset has fs≈5Hz (as printed). So we DON'T do 1–20Hz bandpass here.
- This script still demonstrates the full logic: detect → window → features → classify → evaluate.

Run:
  python step2_rf.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support


# ----------------------------
# CONFIG (edit these)
# ----------------------------
DATA_PATH = Path("archive_1.zip")  # zip OR extracted folder
OUT_DIR = Path("outputs_hivedrive_step2")

TRIP_TO_REPORT = "trip1"

# Window around event peak/label
PRE_S = 1.5
POST_S = 1.5

# Baseline candidate detection (same idea as your Step-1)
HP_WINDOW_S = 2.0          # detrend window (seconds)
BASELINE_S = 20.0          # rolling baseline window (seconds)
K_SIGMA = 2.7
REFRACTORY_S = 1.0
PEAK_SEARCH_S = 0.8

# Matching tolerance to compare predicted events to labels
MATCH_TOL_S = 1.0

# Training data balancing
NEG_PER_POS = 2            # how many negatives per positive from same trip
EXCLUSION_S = 2.0          # don't sample negatives too close to a pothole label


# ----------------------------
# Helpers
# ----------------------------
def extract_if_zip(path: Path) -> Path:
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".zip":
        dst = Path("_extracted_dataset")
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(dst)
        return dst
    return path


def list_trips(root: Path) -> list[str]:
    pothole_dir = root / "Pothole"
    trips = []
    for fp in pothole_dir.glob("trip*_sensors.csv"):
        trips.append(fp.stem.replace("_sensors", ""))
    return sorted(set(trips))


def load_trip(root: Path, trip: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    sensors_fp = root / "Pothole" / f"{trip}_sensors.csv"
    potholes_fp = root / "Pothole" / f"{trip}_potholes.csv"
    df_s = pd.read_csv(sensors_fp)
    df_p = pd.read_csv(potholes_fp) if potholes_fp.exists() else None
    return df_s, df_p


def load_roadcondition(root: Path) -> list[pd.DataFrame]:
    """Optional extra negatives from 'good road' segments."""
    out = []
    rc = root / "RoadCondition"
    for fp in sorted(rc.glob("good*_sensors.csv")):
        out.append(pd.read_csv(fp))
    return out


def compute_fs(ts: np.ndarray) -> float:
    dt = np.median(np.diff(ts))
    if dt <= 0:
        raise ValueError("Non-increasing timestamps.")
    return 1.0 / dt


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


def detrend_axes(df: pd.DataFrame, fs: float, hp_window_s: float):
    """Remove slow trend per axis (approx gravity/orientation drift)."""
    w = max(3, int(fs * hp_window_s))
    ax = df["accelerometerX"].to_numpy()
    ay = df["accelerometerY"].to_numpy()
    az = df["accelerometerZ"].to_numpy()
    gx = df["gyroX"].to_numpy()
    gy = df["gyroY"].to_numpy()
    gz = df["gyroZ"].to_numpy()

    ax0 = ax - rolling_mean(ax, w)
    ay0 = ay - rolling_mean(ay, w)
    az0 = az - rolling_mean(az, w)

    gx0 = gx - rolling_mean(gx, w)
    gy0 = gy - rolling_mean(gy, w)
    gz0 = gz - rolling_mean(gz, w)

    return ax0, ay0, az0, gx0, gy0, gz0


def derived_signals(df: pd.DataFrame, fs: float, hp_window_s: float) -> dict[str, np.ndarray]:
    ax0, ay0, az0, gx0, gy0, gz0 = detrend_axes(df, fs, hp_window_s)
    acc_norm = np.sqrt(ax0 * ax0 + ay0 * ay0 + az0 * az0)
    gyro_norm = np.sqrt(gx0 * gx0 + gy0 * gy0 + gz0 * gz0)
    jerk = np.concatenate([[0.0], np.diff(acc_norm) * fs])
    return {
        "acc_norm": acc_norm,
        "gyro_norm": gyro_norm,
        "jerk": jerk,
        "ax0": ax0,
        "ay0": ay0,
        "az0": az0,
        "gx0": gx0,
        "gy0": gy0,
        "gz0": gz0,
    }


def adaptive_threshold(sig: np.ndarray, fs: float, baseline_s: float, k_sigma: float):
    w = max(10, int(fs * baseline_s))
    kernel = np.ones(w) / w
    mu = np.convolve(sig, kernel, mode="same")
    mu2 = np.convolve(sig * sig, kernel, mode="same")
    sigma = np.sqrt(np.maximum(mu2 - mu * mu, 1e-9))
    th = mu + k_sigma * sigma
    return mu, sigma, th


def detect_peaks(sig: np.ndarray, th: np.ndarray, fs: float, refractory_s: float, peak_search_s: float) -> np.ndarray:
    refractory = int(fs * refractory_s)
    search = max(1, int(fs * peak_search_s))
    above = sig > th

    peaks = []
    i = 0
    n = len(sig)
    while i < n:
        if above[i]:
            j = min(i + search, n - 1)
            p = i + int(np.argmax(sig[i : j + 1]))
            peaks.append(p)
            i = p + refractory
        else:
            i += 1
    return np.array(peaks, dtype=int)


def extract_window_idx(t: np.ndarray, center_t: float, pre_s: float, post_s: float) -> np.ndarray:
    return np.where((t >= center_t - pre_s) & (t <= center_t + post_s))[0]


def features_from_window(sig_dict: dict[str, np.ndarray], idx: np.ndarray, fs: float) -> dict[str, float]:
    acc = sig_dict["acc_norm"][idx]
    gyro = sig_dict["gyro_norm"][idx]
    jerk = sig_dict["jerk"][idx]

    # time-domain features (simple + explainable)
    f = {}
    f["acc_peak"] = float(np.max(acc))
    f["acc_rms"] = float(np.sqrt(np.mean(acc ** 2)))
    f["acc_ptp"] = float(np.ptp(acc))
    f["acc_var"] = float(np.var(acc))
    f["jerk_peak"] = float(np.max(np.abs(jerk)))
    f["gyro_peak"] = float(np.max(gyro))
    f["gyro_rms"] = float(np.sqrt(np.mean(gyro ** 2)))
    f["gyro_var"] = float(np.var(gyro))

    # slope-like feature (coarse fs, still useful)
    f["acc_mean_abs_diff"] = float(np.mean(np.abs(np.diff(acc))) * fs) if len(acc) > 1 else 0.0
    f["gyro_mean_abs_diff"] = float(np.mean(np.abs(np.diff(gyro))) * fs) if len(gyro) > 1 else 0.0
    return f


def build_train_data(root: Path, trips: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build supervised dataset:
    - Positive windows centered at labeled potholes.
    - Negative windows sampled away from labels + additional negatives from 'good road'.
    Returns: X (features), y (0/1), group (trip name)
    """
    rows = []
    ys = []
    groups = []

    for trip in trips:
        df_s, df_p = load_trip(root, trip)
        ts = df_s["timestamp"].to_numpy().astype(float)
        t = ts - ts[0]
        fs = compute_fs(ts)

        sigs = derived_signals(df_s, fs, HP_WINDOW_S)

        label_t = (df_p["timestamp"].to_numpy().astype(float) - ts[0]) if df_p is not None else np.array([])

        # positives
        for lt in label_t:
            idx = extract_window_idx(t, lt, PRE_S, POST_S)
            if len(idx) < 5:
                continue
            rows.append(features_from_window(sigs, idx, fs))
            ys.append(1)
            groups.append(trip)

        # negatives from same trip (random times far from labels)
        if len(label_t) > 0:
            # build mask of forbidden times near potholes
            forbidden = np.zeros_like(t, dtype=bool)
            for lt in label_t:
                forbidden |= (np.abs(t - lt) <= EXCLUSION_S)
            allowed_idx = np.where(~forbidden)[0]
        else:
            allowed_idx = np.arange(len(t))

        rng = np.random.default_rng(123)
        n_neg = max(1, NEG_PER_POS * max(1, len(label_t)))
        for _ in range(n_neg):
            if len(allowed_idx) == 0:
                break
            c = int(rng.choice(allowed_idx))
            ct = t[c]
            idx = extract_window_idx(t, ct, PRE_S, POST_S)
            if len(idx) < 5:
                continue
            rows.append(features_from_window(sigs, idx, fs))
            ys.append(0)
            groups.append(trip)

    # extra negatives from good road segments (optional)
    for i, df_g in enumerate(load_roadcondition(root)):
        ts = df_g["timestamp"].to_numpy().astype(float)
        t = ts - ts[0]
        fs = compute_fs(ts)
        sigs = derived_signals(df_g, fs, HP_WINDOW_S)

        rng = np.random.default_rng(999 + i)
        for _ in range(10):  # small amount; just to add variety
            c = int(rng.integers(0, len(t)))
            idx = extract_window_idx(t, t[c], PRE_S, POST_S)
            if len(idx) < 5:
                continue
            rows.append(features_from_window(sigs, idx, fs))
            ys.append(0)
            groups.append(f"good{i+1}")

    X = pd.DataFrame(rows).fillna(0.0)
    y = pd.Series(ys, name="is_pothole")
    g = pd.Series(groups, name="group")
    return X, y, g


def match_events(label_t: np.ndarray, pred_t: np.ndarray, tol_s: float) -> int:
    """1-to-1 match labels to predictions within tolerance."""
    used = set()
    matched = 0
    for lt in label_t:
        if len(pred_t) == 0:
            break
        j = int(np.argmin(np.abs(pred_t - lt)))
        if abs(pred_t[j] - lt) <= tol_s and j not in used:
            used.add(j)
            matched += 1
    return matched


# ----------------------------
# Main
# ----------------------------
def main():
    root = extract_if_zip(DATA_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    trips = list_trips(root)
    print("Trips:", trips)

    # 1) Build training dataset from all trips
    X, y, g = build_train_data(root, trips)
    print("\nTraining samples:", len(X), "features:", X.shape[1], "pothole positives:", int(y.sum()))

    # 2) Train a RandomForest (simple MVP)
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=7,
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    rf.fit(X, y)

    # 3) Evaluate classification (window-level) using training set (for PPT baseline)
    y_hat = rf.predict(X)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_hat, average="binary", zero_division=0)
    print(f"\nWindow-level (train) Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")

    # Confusion matrix plot
    cm = confusion_matrix(y, y_hat, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["non-pothole", "pothole"])
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    plt.title("Random Forest: Window-level Confusion Matrix (train set)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rf_confusion_matrix.png", dpi=200)
    plt.close()

    # Feature importance plot
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top = importances.head(12)
    plt.figure()
    plt.bar(top.index, top.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("importance")
    plt.title("Top Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rf_feature_importances.png", dpi=200)
    plt.close()

    # 4) Demonstrate "baseline peak detector" vs "peak detector + RF filter" at event-level
    df_s, df_p = load_trip(root, TRIP_TO_REPORT)
    ts = df_s["timestamp"].to_numpy().astype(float)
    t = ts - ts[0]
    fs = compute_fs(ts)

    sigs = derived_signals(df_s, fs, HP_WINDOW_S)
    sig = sigs["acc_norm"]
    _, _, th = adaptive_threshold(sig, fs, BASELINE_S, K_SIGMA)
    peaks = detect_peaks(sig, th, fs, REFRACTORY_S, PEAK_SEARCH_S)

    peak_t = t[peaks]
    label_t = (df_p["timestamp"].to_numpy().astype(float) - ts[0]) if df_p is not None else np.array([])

    # Baseline: treat every peak as pothole
    matched_base = match_events(label_t, peak_t, MATCH_TOL_S)
    prec_base = matched_base / len(peak_t) if len(peak_t) else 0.0
    rec_base = matched_base / len(label_t) if len(label_t) else 0.0

    # RF-filtered: classify each detected peak window, keep only pothole predictions
    pred_pothole_times = []
    for p in peaks:
        ct = t[p]
        idx = extract_window_idx(t, ct, PRE_S, POST_S)
        if len(idx) < 5:
            continue
        feat = pd.DataFrame([features_from_window(sigs, idx, fs)]).fillna(0.0)
        # align columns
        for c in X.columns:
            if c not in feat.columns:
                feat[c] = 0.0
        feat = feat[X.columns]
        pred = rf.predict(feat)[0]
        if pred == 1:
            pred_pothole_times.append(ct)

    pred_pothole_times = np.array(pred_pothole_times, dtype=float)
    matched_rf = match_events(label_t, pred_pothole_times, MATCH_TOL_S)
    prec_rf = matched_rf / len(pred_pothole_times) if len(pred_pothole_times) else 0.0
    rec_rf = matched_rf / len(label_t) if len(label_t) else 0.0

    print(f"\nEvent-level on {TRIP_TO_REPORT}:")
    print(f"Baseline peaks={len(peak_t)} matched={matched_base}/{len(label_t)}  Precision={prec_base:.3f} Recall={rec_base:.3f}")
    print(f"RF-filtered peaks={len(pred_pothole_times)} matched={matched_rf}/{len(label_t)}  Precision={prec_rf:.3f} Recall={rec_rf:.3f}")

    # Plot: baseline vs RF-filtered events timeline
    plt.figure()
    plt.plot(t, sig, label="acc_norm (detrended)")
    plt.plot(t, th, label="adaptive threshold")
    if len(peak_t):
        plt.scatter(peak_t, sig[peaks], marker="o", label=f"baseline peaks ({len(peak_t)})")
    if len(pred_pothole_times):
        plt.scatter(pred_pothole_times, np.interp(pred_pothole_times, t, sig), marker="x",
                    label=f"RF pothole peaks ({len(pred_pothole_times)})")
    if len(label_t):
        for lt in label_t:
            plt.axvline(lt, linewidth=0.7, alpha=0.25)
    plt.xlabel("time (s)")
    plt.ylabel("acc_norm")
    plt.title(
        f"{TRIP_TO_REPORT}: Baseline detection vs RF-filtered pothole events\n"
        f"Baseline P={prec_base:.2f} R={rec_base:.2f} | RF P={prec_rf:.2f} R={rec_rf:.2f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{TRIP_TO_REPORT}_05_baseline_vs_rf_events.png", dpi=200)
    plt.close()

    # Plot: example pothole window vs non-pothole window
    # pick first label as pothole example; and first allowed random as non-pothole
    if len(label_t):
        idx_p = extract_window_idx(t, label_t[0], PRE_S, POST_S)
    else:
        idx_p = np.arange(0, min(len(t), int(fs*(PRE_S+POST_S))))

    # find a non-pothole center far from labels
    forbidden = np.zeros_like(t, dtype=bool)
    for lt in label_t:
        forbidden |= (np.abs(t - lt) <= EXCLUSION_S)
    allowed = np.where(~forbidden)[0]
    idx_n = extract_window_idx(t, t[allowed[len(allowed)//2]] if len(allowed) else t[len(t)//2], PRE_S, POST_S)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t[idx_p], sigs["acc_norm"][idx_p], label="acc_norm")
    plt.plot(t[idx_p], sigs["gyro_norm"][idx_p], label="gyro_norm")
    plt.title("Example POTHOLE window (features computed here)")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t[idx_n], sigs["acc_norm"][idx_n], label="acc_norm")
    plt.plot(t[idx_n], sigs["gyro_norm"][idx_n], label="gyro_norm")
    plt.title("Example NON-POTHOLE window")
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{TRIP_TO_REPORT}_04_window_examples.png", dpi=200)
    plt.close()

    print(f"\nSaved PPT images to: {OUT_DIR.resolve()}")
    for fp in sorted(OUT_DIR.glob("*.png")):
        print(" -", fp.name)


if __name__ == "__main__":
    main()