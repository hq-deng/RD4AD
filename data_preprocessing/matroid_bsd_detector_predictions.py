import json
import os

import requests

BSD_DATASET = "/home/ubuntu/modelplayground/anomaly-detection/datasets/BSData"
BSD_IMAGES = os.path.join(BSD_DATASET, "data")
# MATROID_DETECTOR_TRAIN_SET = os.path.join(BSD_DATASET, "matroid_detector_train_set.json")
MATROID_DETECTOR_TRAIN_SET = os.listdir(os.path.join(BSD_DATASET, "matroid_detector_v2_data_file/pitting"))
# OUT_FILE = os.path.join(BSD_DATASET, "matroid_detector_predictions.json")
OUT_FILE = os.path.join(BSD_DATASET, "matroid_detector_v2_predictions.json")

HEADERS = {"Authorization": "Bearer ab001ac29305579591ba91ff3a88c050"}
# DETECTOR_URL = "https://app.matroid.com/api/v1/detectors/61b0c3230f890f0006dd9030/classify_image"
DETECTOR_URL = "https://app.matroid.com/api/v1/detectors/62ffde85a9eb0d0007abcea2/classify_image"

# with open(MATROID_DETECTOR_TRAIN_SET, 'r') as f:
#     train_set = json.load(f)
#     train_set = {image["fileName"] for image in train_set["images"]}
train_set = set(MATROID_DETECTOR_TRAIN_SET)
results = {}

for img in os.listdir(BSD_IMAGES):
    if img in train_set:
        print("Skipping img", img)
        continue

    filepath = os.path.join(BSD_IMAGES, img)
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

