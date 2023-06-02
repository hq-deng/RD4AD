import json
import os

import requests

KSDD_DATASET = "/home/ubuntu/modelplayground/anomaly-detection/datasets/KSDD2"
KSDD_IMAGES = os.path.join(KSDD_DATASET, "all")
MATROID_DETECTOR_TRAIN_SET = os.listdir(os.path.join(KSDD_DATASET, "matroid_detector_data_file/defects"))
OUT_FILE = os.path.join(KSDD_DATASET, "matroid_detector_predictions.json")

HEADERS = {"Authorization": "Bearer ab001ac29305579591ba91ff3a88c050"}
DETECTOR_URL = "https://app.matroid.com/api/v1/detectors/630ae495a41a7f0007e5b68a/classify_image"

train_set = set(MATROID_DETECTOR_TRAIN_SET)
results = {}

for img in os.listdir(KSDD_IMAGES):
    if img in train_set:
        print("Skipping img", img)
        continue

    filepath = os.path.join(KSDD_IMAGES, img)
    result = requests.post(DETECTOR_URL, headers=HEADERS, files={'file': open(filepath, 'rb')})
    print(result.text)
    result = result.json()["results"][0]
    print(result)
    if result.get('message') == 'no objects found in image':
        predictions = []
    else:
        predictions = result["predictions"]
    results[result['file']['name']] = predictions

with open(OUT_FILE, 'w') as f:
    json.dump(results, f)

