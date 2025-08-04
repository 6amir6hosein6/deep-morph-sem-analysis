import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import os

def smooth_line(x, y):
    if len(x) < 4:
        return x, y  # Cannot interpolate if fewer than 4 points
    spline = make_interp_spline(x, y, k=3)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

def plot_distribution(csv_path):
    if not os.path.isfile(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return
    
    # Read CSV
    df = pd.read_csv(csv_path, header=None)
    diameters = df.values.flatten()

    # Histogram
    bins = np.linspace(min(diameters), max(diameters), 40)
    counts, _ = np.histogram(diameters, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Smooth
    x_smooth, y_smooth = smooth_line(bin_centers, counts)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_smooth, y_smooth, color='blue', label='Smoothed Distribution')
    plt.scatter(bin_centers, counts, color='red', s=20, alpha=0.6)
    plt.xlabel("Diameter (nm)")
    plt.ylabel("Count")
    plt.title(f"Distribution from {os.path.basename(csv_path)}")
    plt.grid(True)
    plt.tight_layout()
    output_file = os.path.splitext(csv_path)[0] + "_distribution.png"
    plt.savefig(output_file, dpi=300)
    plt.show()
    print(f"[INFO] Plot saved as: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot size distribution from CSV.")
    parser.add_argument("-c", "--csv", type=str, required=True, help="Path to CSV file containing diameter values")
    args = parser.parse_args()

    plot_distribution(args.csv)

