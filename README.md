# YYZ Ground Congestion Prediction

Predicting near-term aircraft ground traffic at Toronto Pearson International Airport (YYZ) from ADS-B flight tracking data, using convolutional neural networks.

Built for the **I4 Case Competition: The YYZ Optimization Challenge** (March 2026), which asked teams to find bottlenecks in Pearson's operations using real-world data and propose engineering solutions.

## Approach

1. **Find aircraft at YYZ:** scan ADS-B trace files and keep aircraft whose positions pass through a bounding box around Pearson.
2. **Build traffic heatmaps:** bin aircraft positions on or near the ground into a 64×64 grid over the airport, in 30-second time windows, and blur each grid into a density heatmap.
3. **Augment:** the real data covers 22 flights, so each real position is expanded into jittered augmented copies (small Gaussian offsets in space and time) to produce enough training samples (1,704 heatmaps).
4. **Predict:** train a CNN that takes the current heatmap and predicts the heatmap for the next time window, so congestion can be anticipated before it builds up.

A separate fully synthetic dataset (aircraft converging on a congestion zone) is used to test the model design in a controlled setting.

## Files

| File | Purpose |
|---|---|
| `casecomp.py` | Lists aircraft in a trace folder that passed through the YYZ bounding box |
| `buildheatmaps.py` | Time-windowed heatmaps from real traces (earlier version) |
| `buildheatmaps2.py` | Time-windowed heatmaps with Gaussian blur, data augmentation and normalization → `yyz_heatmaps_augmented.npy` |
| `traincnn.py` | CNN (2× conv + pooling, dense decoder) trained to predict the next heatmap from the current one |
| `trafficflights.py` | Generates a fully synthetic dataset of current → next traffic heatmaps → `clean_current.npy`, `clean_next.npy` |
| `cnn2.py` | Fully convolutional CNN trained on the synthetic dataset |

## Data

ADS-B traces are JSON files (`trace_full_<icao>.json`, optionally gzipped) where each point is `[timestamp, lat, lon, altitude, ...]`. Raw traces and generated `.npy` datasets are not included in this repository because of their size. To regenerate them, place the trace folders locally, update the folder paths at the top of the scripts, and run:

```bash
pip install -r requirements.txt
python buildheatmaps2.py   # real + augmented heatmaps
python traincnn.py         # train and visualize a prediction

python trafficflights.py   # synthetic dataset
python cnn2.py
```

## Status

Code for the competition prototype. The congestion-aware routing component that used these predictions was developed separately and is not included here.

## Author

Ahmed Farag, University of Toronto Engineering Science (Machine Intelligence)
