"""
HiveDrive – IMU dataset quick plotting + basic pothole analysis (no prototype required)

What this script does:
1) Loads the provided dataset (from ZIP or extracted folder)
2) Plots raw accelerometer/gyro signals
3) Uses a simple, explainable baseline algorithm:
   - remove slow trend (rolling mean per axis)
   - compute "dynamic acceleration norm"
   - adaptive threshold (rolling mean/std) + refractory
4) Compares detected events vs labeled pothole timestamps (if available)
5) Saves PPT-ready PNG plots + prints summary counts/metrics
"""

from __future__ import annotations

import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# CONFIG (edit these)
# ----------------------------
DATA_PATH = Path("archive_1.zip")  # <- set to your zip OR extracted folder
OUT_DIR = Path("outputs_hivedrive")
TRIP_TO_PLOT = "trip1"  # trip1..trip5

# Detection knobs (keep simple for PPT)
HP_WINDOW_S = 2.0         # seconds: rolling mean window for removing slow drift/gravity-ish component
BASELINE_S = 20.0         # seconds: rolling mean/std for adaptive threshold
K_SIGMA = 2.7             # threshold = mu + K_SIGMA*std (tune per trip)
REFRACTORY_S = 1.0        # seconds: after a detection, ignore further triggers for this long
PEAK_SEARCH_S = 0.8       # seconds: search local max after threshold crossing
MATCH_TOL_S = 1.0         # seconds: label matching tolerance


# ----------------------------
# Helpers
# ----------------------------
def extract_if_zip(path: Path) -> Path:
    """If path is a .zip, extract to ./_extracted_dataset and return that folder; else return path."""
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".zip":
        dst = Path("_extracted_dataset")
        if dst.exists():
            # don't blow away user data; just reuse
            pass
        else:
            dst.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(dst)
        return dst
    return path


def list_trips(root: Path) -> list[str]:
    """Return trip names available in root/Pothole/ like trip1.."""
    pothole_dir = root / "Pothole"
    trips = []
    for fp in pothole_dir.glob("trip*_sensors.csv"):
        trips.append(fp.stem.replace("_sensors", ""))
    return sorted(set(trips))


def load_trip(root: Path, trip: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load sensors + pothole labels (if present)."""
    sensors_fp = root / "Pothole" / f"{trip}_sensors.csv"
    potholes_fp = root / "Pothole" / f"{trip}_potholes.csv"
    df_s = pd.read_csv(sensors_fp)

    df_p = None
    if potholes_fp.exists():
        df_p = pd.read_csv(potholes_fp)

    return df_s, df_p


def compute_fs(ts: np.ndarray) -> float:
    """Estimate sampling frequency from timestamp seconds (median dt)."""
    dt = np.median(np.diff(ts))
    if dt <= 0:
        raise ValueError("Non-increasing timestamps in sensor file.")
    return 1.0 / dt


def dynamic_acc_norm(df_s: pd.DataFrame, fs: float, hp_window_s: float) -> np.ndarray:
    """
    Remove slow trend on each accel axis using rolling mean, then compute norm.
    This approximates "linear acceleration magnitude" without doing full AHRS.
    """
    w = max(5, int(fs * hp_window_s))
    ax = df_s["accelerometerX"].to_numpy()
    ay = df_s["accelerometerY"].to_numpy()
    az = df_s["accelerometerZ"].to_numpy()

    ax0 = ax - pd.Series(ax).rolling(w, center=True, min_periods=1).mean().to_numpy()
    ay0 = ay - pd.Series(ay).rolling(w, center=True, min_periods=1).mean().to_numpy()
    az0 = az - pd.Series(az).rolling(w, center=True, min_periods=1).mean().to_numpy()

    return np.sqrt(ax0 * ax0 + ay0 * ay0 + az0 * az0)


def adaptive_threshold(sig: np.ndarray, fs: float, baseline_s: float, k_sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rolling mean/std and threshold = mu + k*std using convolution."""
    w = max(10, int(fs * baseline_s))
    kernel = np.ones(w) / w
    mu = np.convolve(sig, kernel, mode="same")
    mu2 = np.convolve(sig * sig, kernel, mode="same")
    sigma = np.sqrt(np.maximum(mu2 - mu * mu, 1e-9))
    th = mu + k_sigma * sigma
    return mu, sigma, th


def detect_peaks(sig: np.ndarray, th: np.ndarray, fs: float, refractory_s: float, peak_search_s: float) -> np.ndarray:
    """Detect peaks where sig crosses above th; choose local max; apply refractory."""
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


def match_labels_to_peaks(label_t: np.ndarray, peak_t: np.ndarray, tol_s: float) -> int:
    """Count how many labels can be matched to a detected peak within tolerance (1-to-1)."""
    used = set()
    matched = 0
    for lt in label_t:
        if len(peak_t) == 0:
            break
        j = int(np.argmin(np.abs(peak_t - lt)))
        if abs(peak_t[j] - lt) <= tol_s and j not in used:
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
    if not trips:
        raise RuntimeError(f"No trips found under {root}/Pothole. Check DATA_PATH.")

    print("Trips found:", trips)

    # ----- Summary across all trips (ground truth pothole counts) -----
    print("\nGround-truth pothole counts (from *_potholes.csv):")
    total_gt = 0
    for trip in trips:
        _, df_p = load_trip(root, trip)
        gt = len(df_p) if df_p is not None else 0
        total_gt += gt
        print(f"  {trip}: {gt}")
    print("  TOTAL:", total_gt)

    # ----- Detailed plot + analysis for one trip -----
    trip = TRIP_TO_PLOT
    df_s, df_p = load_trip(root, trip)

    ts = df_s["timestamp"].to_numpy().astype(float)
    t = ts - ts[0]  # relative seconds
    fs = compute_fs(ts)

    print(f"\n=== {trip} ===")
    print(f"samples={len(df_s)}, duration_s={t[-1]:.1f}, fs≈{fs:.2f} Hz")

    # Basic derived signal for detection
    sig = dynamic_acc_norm(df_s, fs=fs, hp_window_s=HP_WINDOW_S)
    mu, sigma, th = adaptive_threshold(sig, fs=fs, baseline_s=BASELINE_S, k_sigma=K_SIGMA)
    peaks = detect_peaks(sig, th, fs=fs, refractory_s=REFRACTORY_S, peak_search_s=PEAK_SEARCH_S)

    peak_t = t[peaks]
    print(f"Detected events (simple baseline): {len(peaks)}")

    # Labels (if present)
    label_t = None
    gt = 0
    if df_p is not None and "timestamp" in df_p.columns:
        label_ts = df_p["timestamp"].to_numpy().astype(float)
        label_t = label_ts - ts[0]
        gt = len(label_t)
        matched = match_labels_to_peaks(label_t, peak_t, tol_s=MATCH_TOL_S)
        precision = matched / len(peaks) if len(peaks) else 0.0
        recall = matched / gt if gt else 0.0
        print(f"Ground-truth potholes: {gt}")
        print(f"Matched within ±{MATCH_TOL_S:.1f}s: {matched}")
        print(f"Precision≈{precision:.3f}, Recall≈{recall:.3f}")

    # ---- Plot 1: Raw accel axes ----
    plt.figure()
    plt.plot(t, df_s["accelerometerX"], label="accX", linewidth=0.8)
    plt.plot(t, df_s["accelerometerY"], label="accY", linewidth=0.8)
    plt.plot(t, df_s["accelerometerZ"], label="accZ", linewidth=0.8)
    if label_t is not None:
        for lt in label_t:
            plt.axvline(lt, linewidth=0.7, alpha=0.4)
    plt.xlabel("time (s)")
    plt.ylabel("acc (dataset units, often ~g)")
    plt.title(f"{trip}: Raw Accelerometer Axes (vertical lines = labeled potholes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{trip}_01_raw_acc_axes.png", dpi=200)
    plt.close()

    # ---- Plot 2: Dynamic accel norm + threshold + detections ----
    plt.figure()
    plt.plot(t, sig, label="dynamic_acc_norm", linewidth=0.8)
    plt.plot(t, th, label=f"adaptive threshold (mu + {K_SIGMA}σ)", linewidth=0.8)
    if len(peaks):
        plt.scatter(peak_t, sig[peaks], marker="o", label="detected peaks")
    if label_t is not None:
        plt.scatter(label_t, np.interp(label_t, t, sig), marker="x", label="labeled potholes")
    plt.xlabel("time (s)")
    plt.ylabel("dynamic accel norm")
    plt.title(f"{trip}: Simple Detection (rolling detrend → adaptive threshold)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{trip}_02_detection_overview.png", dpi=200)
    plt.close()

    # ---- Plot 3: Zoomed view around first few detections ----
    if len(peaks):
        zoom_center = peak_t[0]
        x0, x1 = max(0, zoom_center - 10), min(t[-1], zoom_center + 10)
        mask = (t >= x0) & (t <= x1)

        plt.figure()
        plt.plot(t[mask], sig[mask], label="dynamic_acc_norm", linewidth=0.8)
        plt.plot(t[mask], th[mask], label="adaptive threshold", linewidth=0.8)
        # show peaks in zoom window
        z_peaks = peaks[(peak_t >= x0) & (peak_t <= x1)]
        if len(z_peaks):
            plt.scatter(t[z_peaks], sig[z_peaks], marker="o", label="detected peaks")
        if label_t is not None:
            z_labels = label_t[(label_t >= x0) & (label_t <= x1)]
            if len(z_labels):
                plt.scatter(z_labels, np.interp(z_labels, t, sig), marker="x", label="labeled potholes")
        plt.xlabel("time (s)")
        plt.ylabel("dynamic accel norm")
        plt.title(f"{trip}: Zoom (±10s around first detected event)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{trip}_03_zoom.png", dpi=200)
        plt.close()

    print(f"\nSaved plots to: {OUT_DIR.resolve()}")
    print("Use these PNGs directly in your PPT:")
    for fp in sorted(OUT_DIR.glob(f"{trip}_*.png")):
        print(" -", fp.name)


if __name__ == "__main__":
    main()