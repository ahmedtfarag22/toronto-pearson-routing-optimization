import json
import os
import gzip

LAT_MIN = 43.65
LAT_MAX = 43.70
LON_MIN = -79.65
LON_MAX = -79.60

folder = "/Users/ahmed/09"

yyz_aircraft = []

for file in os.listdir(folder):
    if file.endswith(".json") or file.endswith(".gz"):
        path = os.path.join(folder, file)

        try:
            # Try reading as gzip
            with gzip.open(path, "rt") as f:
                data = json.load(f)
        except:
            # If not gzip, read normally
            with open(path, "r") as f:
                data = json.load(f)

        trace = data.get("trace", [])

        for point in trace:
            lat = point[1]
            lon = point[2]

            if (
                lat is not None and
                lon is not None and
                LAT_MIN <= lat <= LAT_MAX and
                LON_MIN <= lon <= LON_MAX
            ):
                yyz_aircraft.append(file)
                break

print("Aircraft near YYZ:")
print(yyz_aircraft)