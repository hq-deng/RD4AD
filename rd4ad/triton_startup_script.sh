rm -rf ad_inference/1/ && rm model_checkpoint.pth && rm -rf models/
pip install -r environment/docker_requirements.txt

mkdir ad_inference/1
cp ad_inference/model.py ad_inference/1/ 
cp ad_inference/config.pbtxt ad_inference/1/ 
cp ad_inference/example_training_stats.json ad_inference/1/training_stats.json

gdown 1E6M0HNn7Mc9wMDf-VH3IWWjIPqFxAHEl

mkdir models
cp -R ad_inference  models/
cp model_checkpoint.pth models/ad_inference/1/
cp training_stats.json models/ad_inference/1/

tritonserver --model-repository `pwd`/models

