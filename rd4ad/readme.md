# Matroid fork of RD4AD code with modifications.
See old_readme.md for the original readme.md created by the authors
with citation information, etc.

## Overview
This repo contains a modified implementation of Anomaly Detection Using Reverse Distillation from One-Class Embedding.

To run the code, first ensure that you have cloned this code not in isolation, but as a part of the Matroid modelplayground.
This will ensure that you have access to a few common anomaly detection utils found in that repository.

Copy any datasets that you need to into the datasets/ folder in modelplayground/anomaly-detection/datasets. You may need
to adjust the paths to the dataset in main.py.

## Environment

To create a new conda environment with the necessary packages, run the following:
```
conda env create --name rd4ad --file=environment/environment.yml
conda activate rd4ad
```

## Running the code
Here is an example of training an anomaly detection model on the bsd (Ball-screw drive) dataset, one of
the three datasets featured in my final presentation:
```
export PYTHONPATH=".:.." && python main.py --lr 0.0004 -bs 1 -inp 1024 --dataset bsd --epochs 70
```
Here is an example of testing the same model:
```
export PYTHONPATH=".:.." && python main.py --dataset bsd --inp 1024 --action-type norm-test --save-segmentation-images --checkpoint checkpoints/PATH_TO_CHECKPOINT
```
This demonstrates the most important command line arguments/flags that can be supplied. The complete list
of command line arguments is in the `config.py` file. Here is a list of the most important ones:
   - `--dataset`: Name of the dataset (not the path to the dataset; the path is set in the code and can be changed there)
   - `--checkpoint`: Path to checkpoint
   - `--class-names`: A list of classnames upon which to train/test. This is only useful for datasets that have multiple classes; e.g. MVTec has cable, pill, etc.
   - `--inp`: Image size. How exactly this is handled is up to the torch.utils.Dataset defined for the particular dataset. For example, some datasets might perform additional cropping or square padding.
   - `--action-type`: Whether to train/test. These are the two most important actions. There are other actions as well that I used to run various less important experiments featured in my presentation (e.g., comparing to Matroid, etc.).
   - `--bs`: Batch size
   - `--lr`: Learning rate
   - `--epochs`: epochs
   - `--save-segmentation-images`: If we are evaluating, whether to save off the predictions as images.
   - `--patches-per-row`: If patches-per-row is > 1, this trains or tests RD4AD in a sliding-window approach over a grid, using the designated number of patches in every row. How this is handled depends on the implementation of the torch.utils.Dataset for the dataset.
   - `--patches-per-column`: Same as `--patches-per-row`, but designates the number of patches in each column instead.

The other arguments are used for specific experiments that I ran but are not vital to replicate. These are documented in config.py.
## Triton inference code
See `ad_inference/` for the Triton inference code.

To test said inference code, follow this procedure.
  1. Install nvidia-docker2 if it is not already installed.

  2. Start up the server docker instance using the following commands:
  ```
  docker run --runtime=nvidia --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:22.04-py3

  git clone https://github.com/nraghuraman-matroid/RD4AD-fork
  cd RD4AD-fork

  bash triton_startup_script.sh
  ```
  This bash script above downloads some example weights from Google drive and uses a json
  containing example hyperparameters (`example_training_stats.json`). In a production setting, these weights and the json
  containing hyperparameters would be produced by the training script.

  3. Start up the client docker instance and run a test using the following commands:
  ```
  docker run --runtime=nvidia -ti --net host nvcr.io/nvidia/tritonserver:22.04-py3-sdk /bin/bash
  git clone https://github.com/nraghuraman-matroid/RD4AD-fork
  cd RD4AD-fork/ad_inference
  python3 client.py
  ```
  If the whole process succeeded, you should see `PASS: ad_inference`

## Example scripts
The folder sample_scripts/ has a few examples of sample scripts that I ran for simpler experiments from the first half of my
internship.

The folder other_scripts/ has all other scripts that I've used for my experiments. These aren't meant to be read directly as they are mostly for various
experiments that I've run, not all of which are necessary to replicate. However, these can be grepped to see example usages of certain
command line examples.

## Output of the model
Currently, `main.py` outputs a `.pth` file containing the trained model weights. This does not exactly match the input required
for the Triton server code which, as it's currently written, additionally expects a `.json` containing a few important statistics
from training. However, this can be easily changed.

It's also important to note that the anomaly scores produced by the model are between 0 and 3, with the difference between normal and
anomalous anomaly scores rather minimal. Ideally, when displaying anomaly scores to the user, we would like them to match the interpretation
that 0=extremely normal, 0.5=iffy, 1=extremely anomalous. This is something that I have experimented with in my code but have not fully
resolved; there's a branch of this repo titled "rescale" that will explore this issue in greater detail.

## Other 
As mentioned above, the two most important ways to interact with this code are training and testing. There are four other ways to
interact with the code. These are less important unless you specifically want to replicate certain experiments I included in my
presentation.
  - `--action-type` is `--norm-test-fps`
  - `--action-type` is `--norm-compare-to-matroid`
  - `--action-type` is `--norm-patches-test`
  - `--action-type` is `--norm-patches-compare-to-matroid`

The first tests the fps of the RD4AD model.
The second compares the RD4AD on a dataset to a Matroid detector was trained on the same dataset (computes mAP, detection F1 score, segmentation F1
score, etc.). For this code path, it is required that
`--pred-threshold` is set, as the mAP metrics computed in this code path require the usage of a specific anomaly threshold. It is also
required that `--cmp-bboxes` is set to a path to a json containing Matroid predictions on all images in the dataset upon which
the detector was not trained.
The third is the same as `--norm-test` but is useful for when the model was trained on patches rather than full images. It evaluates the model
on patches and "unpatches" them back to full images in order to compute scores.
The fourth is the same as `--norm-compare-to-matroid` except it also unpatches patched images back to full images.

## Common anomaly detection utils
See matroid/modelplayground/anomaly-detection for common anomaly detection utils as well as the torch.utils.Datasets that I used
for this code.