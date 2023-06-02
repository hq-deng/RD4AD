# Common utilities for anomaly detection
This directory contains a few common utilities for anomaly detection:
- `datasets/` contains `torch.utils.Datasets` for the anomaly detection datasets
  that Nikhil worked with. `bsd_dataset.py` is intended for the Ball-screw drive
  dataset, `custom_dataset.py` is intended for the customer datasets, namely the
  connectors dataset and the chrome automotive parts dataset, and `ksdd_dataset.py`
  is intended for the KSDD2 dataset.
- `evaluation/` contains evaluation code (and tests) for a few important metrics,
mainly:
  1. bounding box metrics, i.e., mAP, that are intended to be calculated either from
  bounding box predictions produced by an object detection model or heatmaps produced
  by an anomaly detection model.
  2. segmentation metrics, i.e., AUPRO.
- `profiler.py` contains a class that can be used to profile training of a model and measure
  e.g., CPU and GPU RAM usage.
- `utils.py` contains a couple other useful utilities.
- `visualize.py` contains visualization code that I used to generate the figures that I
  used in my presentations.
