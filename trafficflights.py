import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

GRID = 64
SAMPLES = 2000

X = []
Y = []

for _ in range(SAMPLES):

    current = np.zeros((GRID, GRID))
    next_map = np.zeros((GRID, GRID))

    # ----------------------
    # spaced aircraft
    # ----------------------

    planes = np.random.randint(4,7)

    plane_positions = []

    for i in range(planes):

        x = np.random.randint(10,54)
        y = np.random.randint(10,54)

        current[y,x] = 1
        plane_positions.append((x,y))

    # ----------------------
    # congestion zone
    # ----------------------

    cx = np.random.randint(20,44)
    cy = np.random.randint(20,44)

    congestion_size = np.random.randint(8,15)

    for i in range(congestion_size):

        x = int(np.random.normal(cx,2))
        y = int(np.random.normal(cy,2))

        if 0 <= x < GRID and 0 <= y < GRID:
            next_map[y,x] += 1

    # move planes toward congestion
    for x,y in plane_positions:

        dx = int((cx - x)*0.2)
        dy = int((cy - y)*0.2)

        nx = x + dx + np.random.randint(-1,2)
        ny = y + dy + np.random.randint(-1,2)

        if 0 <= nx < GRID and 0 <= ny < GRID:
            next_map[ny,nx] += 1

    # blur into heatmaps
    current = gaussian_filter(current,1)
    next_map = gaussian_filter(next_map,1)

    X.append(current)
    Y.append(next_map)

X = np.array(X)
Y = np.array(Y)

X = X[..., np.newaxis]
Y = Y[..., np.newaxis]

np.save("clean_current.npy",X)
np.save("clean_next.npy",Y)

print("Dataset created")
print("X:",X.shape)
print("Y:",Y.shape)