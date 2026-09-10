import os
import glob
import numpy as np
import csv

INPUT_DIR = "outputs/pohang"
OUTPUT_CSV = "outputs/pohang/disparity_statistics.csv"

files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.npy")))

if not files:
    raise FileNotFoundError(f"没有在 {INPUT_DIR} 找到 .npy 文件")

rows = []

for path in files:
    disp = np.load(path).astype(np.float32)

    total = disp.size

    nan_mask = np.isnan(disp)
    inf_mask = np.isinf(disp)
    finite_mask = np.isfinite(disp)

    finite_disp = disp[finite_mask]

    if finite_disp.size == 0:
        print(f"{path}: 没有有效视差")
        continue

    # 基本统计
    d_min = float(np.min(finite_disp))
    d_max = float(np.max(finite_disp))
    d_mean = float(np.mean(finite_disp))
    d_median = float(np.median(finite_disp))

    p5 = float(np.percentile(finite_disp, 5))
    p25 = float(np.percentile(finite_disp, 25))
    p75 = float(np.percentile(finite_disp, 75))
    p95 = float(np.percentile(finite_disp, 95))
    p99 = float(np.percentile(finite_disp, 99))

    # 异常/特殊视差比例
    near_zero_ratio = float(np.sum(finite_disp <= 0.1) / finite_disp.size)
    negative_ratio = float(np.sum(finite_disp < 0) / finite_disp.size)
    over_192_ratio = float(np.sum(finite_disp > 192) / finite_disp.size)

    nan_ratio = float(np.sum(nan_mask) / total)
    inf_ratio = float(np.sum(inf_mask) / total)

    filename = os.path.basename(path)

    rows.append({
        "file": filename,
        "min": d_min,
        "max": d_max,
        "mean": d_mean,
        "median": d_median,
        "p5": p5,
        "p25": p25,
        "p75": p75,
        "p95": p95,
        "p99": p99,
        "near_zero_ratio": near_zero_ratio,
        "negative_ratio": negative_ratio,
        "over_192_ratio": over_192_ratio,
        "nan_ratio": nan_ratio,
        "inf_ratio": inf_ratio,
    })

    print(
        f"{filename:15s} "
        f"min={d_min:8.3f} "
        f"max={d_max:8.3f} "
        f"mean={d_mean:8.3f} "
        f"median={d_median:8.3f} "
        f"p95={p95:8.3f} "
        f"zero={near_zero_ratio*100:6.2f}% "
        f">192={over_192_ratio*100:6.2f}%"
    )

# 保存 CSV
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

fieldnames = [
    "file",
    "min",
    "max",
    "mean",
    "median",
    "p5",
    "p25",
    "p75",
    "p95",
    "p99",
    "near_zero_ratio",
    "negative_ratio",
    "over_192_ratio",
    "nan_ratio",
    "inf_ratio",
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\n统计完成")
print(f"共分析 {len(rows)} 个视差文件")
print(f"CSV 保存到: {OUTPUT_CSV}")