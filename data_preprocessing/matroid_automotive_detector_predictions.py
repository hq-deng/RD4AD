import json
import os

import requests

AUTOMOTIVE_DATASET = "/home/ubuntu/modelplayground/anomaly-detection/datasets/automotive/"
AUTOMOTIVE_IMAGES = os.path.join(AUTOMOTIVE_DATASET, "processed/top_easy_split/imgs")
MATROID_DETECTOR_TRAIN_SET = os.listdir(os.path.join(AUTOMOTIVE_DATASET, "matroid_detector_data_file/defects"))
OUT_FILE = os.path.join(AUTOMOTIVE_DATASET, "matroid_detector_predictions.json")

HEADERS = {"Authorization": "Bearer ab001ac29305579591ba91ff3a88c050"}
DETECTOR_URL = "https://app.matroid.com/api/v1/detectors/63117cb12f6d8b0007ea6ed4/classify_image"

train_set = set(MATROID_DETECTOR_TRAIN_SET)
results = {}

for img in os.listdir(AUTOMOTIVE_IMAGES):
    if img in train_set:
        print("Skipping img", img)
        continue

    filepath = os.path.join(AUTOMOTIVE_IMAGES, img)
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

