import json
import gzip
import os
import numpy as np

# Folders where trace files may exist
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

# Flights you want to process
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
"trace_full_a55a0c.json",
"trace_full_a35701.json",
"trace_full_c07f02.json",
"trace_full_c03c02.json",
"trace_full_a4ba02.json",
"trace_full_c06403.json",
"trace_full_c1e803.json",
"trace_full_c03003.json",
"trace_full_0d0b04.json",
"trace_full_c05f04.json",
"trace_full_c05004.json",
"trace_full_495304.json",
"trace_full_a2c604.json",
"trace_full_c08106.json",
"trace_full_c02207.json",
"trace_full_a67407.json",
"trace_full_aaf007.json",
"trace_full_3c6708.json",
"trace_full_495309.json",
"trace_full_c05e09.json"
]

# Pearson Airport bounding box
LAT_MIN = 43.60
LAT_MAX = 43.75
LON_MIN = -79.70
LON_MAX = -79.50

GRID_SIZE = 64
WINDOW = 30  # seconds

heatmaps = []

for file in files:

    path = None

    # search through folders
    for folder in folders:
        candidate = os.path.join(folder, file)
        if os.path.exists(candidate):
            path = candidate
            break

    if path is None:
        print("Skipping missing file:", file)
        continue

    try:
        with gzip.open(path, "rt") as f:
            data = json.load(f)
    except:
        with open(path, "r") as f:
            data = json.load(f)

    trace = data.get("trace", [])

    if len(trace) == 0:
        continue

    trace.sort(key=lambda x: x[0])

    start_time = trace[0][0]
    end_time = trace[-1][0]

    current_time = start_time

    while current_time <= end_time:

        grid = np.zeros((GRID_SIZE, GRID_SIZE))

        for point in trace:

            timestamp = point[0]
            lat = point[1]
            lon = point[2]
            alt = point[3]

            if lat is None or lon is None:
                continue

            # keep aircraft near ground
            if alt == "ground" or (isinstance(alt, (int, float)) and alt < 1000):

                if current_time <= timestamp < current_time + WINDOW:

                    if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:

                        x = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * GRID_SIZE)
                        y = int((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * GRID_SIZE)

                        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                            grid[y, x] += 1

        if np.sum(grid) > 0:
            heatmaps.append(grid)

        current_time += WINDOW

print("Total heatmaps generated:", len(heatmaps))

heatmaps = np.array(heatmaps)

np.save("yyz_heatmaps.npy", heatmaps)

print("Dataset saved: yyz_heatmaps.npy")
print("Dataset shape:", heatmaps.shape)