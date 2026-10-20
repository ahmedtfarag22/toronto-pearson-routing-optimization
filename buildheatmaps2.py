import json
import gzip
import os
import numpy as np
from collections import defaultdict
from scipy.ndimage import gaussian_filter

# -----------------------------
# folders containing traces
# -----------------------------

folders = [
"/Users/ahmed/012",
"/Users/ahmed/02",
"/Users/ahmed/022",
"/Users/ahmed/03",
"/Users/ahmed/04",
"/Users/ahmed/05",
"/Users/ahmed/052",
"/Users/ahmed/06",
"/Users/ahmed/062",
"/Users/ahmed/07",
"/Users/ahmed/072",
"/Users/ahmed/08",
"/Users/ahmed/092",
"/Users/ahmed/0a",
"/Users/ahmed/0b",
"/Users/ahmed/0c"
]


files = [
"trace_full_48ae05.json",
"trace_full_c1e805.json",
"trace_full_48ae06.json",
"trace_full_c1e806.json",
"trace_full_c1e807.json",
"trace_full_c05d08.json",
"trace_full_a7d308.json",
"trace_full_c05f08.json",
"trace_full_c05f09.json",
"trace_full_c07b09.json",
"trace_full_c1e909.json",
"trace_full_c01809.json",
"trace_full_c06709.json",
"trace_full_aa490a.json",
"trace_full_c05f0a.json",
"trace_full_c07b0a.json",
"trace_full_c1e80a.json",
"trace_full_c1e80b.json",
"trace_full_a02d0c.json",
"trace_full_c1e80c.json",
"trace_full_49530c.json",
"trace_full_a55a0c.json"
]


LAT_MIN = 43.60
LAT_MAX = 43.75
LON_MIN = -79.70
LON_MAX = -79.50

GRID_SIZE = 64
WINDOW = 30

# augmented copies per real position (spatial + temporal jitter)
AUGMENT_PER_POINT = 50

windows = defaultdict(list)

# -----------------------------
# collect aircraft
# -----------------------------

for file in files:

    path = None

    for folder in folders:
        candidate = os.path.join(folder, file)
        if os.path.exists(candidate):
            path = candidate
            break

    if path is None:
        print("Skipping:", file)
        continue

    try:
        with gzip.open(path, "rt") as f:
            data = json.load(f)
    except:
        with open(path, "r") as f:
            data = json.load(f)

    trace = data.get("trace", [])

    for point in trace:

        timestamp = point[0]
        lat = point[1]
        lon = point[2]

        if lat is None or lon is None:
            continue

        if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
            continue

        window_id = int(timestamp // WINDOW)

        windows[window_id].append((lat, lon))

        # augment with jittered copies of this position
        for _ in range(AUGMENT_PER_POINT):

            lat_offset = np.random.normal(0, 0.001)
            lon_offset = np.random.normal(0, 0.001)

            aug_lat = lat + lat_offset
            aug_lon = lon + lon_offset

            if LAT_MIN <= aug_lat <= LAT_MAX and LON_MIN <= aug_lon <= LON_MAX:

                time_shift = np.random.randint(-3, 4)

                windows[window_id + time_shift].append((aug_lat, aug_lon))

print("Total time windows:", len(windows))

heatmaps = []

for window_id in sorted(windows.keys()):

    grid = np.zeros((GRID_SIZE, GRID_SIZE))

    for lat, lon in windows[window_id]:

        x = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * GRID_SIZE)
        y = int((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * GRID_SIZE)

        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            grid[y, x] += 1

    if np.sum(grid) > 0:

        # convert points to blobs
        grid = gaussian_filter(grid, sigma=1.2)

        heatmaps.append(grid)

heatmaps = np.array(heatmaps)

print("Heatmaps created:", heatmaps.shape)

# normalize dataset
heatmaps = heatmaps / np.max(heatmaps)

np.save("yyz_heatmaps_augmented.npy", heatmaps)

print("Saved dataset to yyz_heatmaps_augmented.npy")