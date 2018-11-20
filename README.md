# Unsupervised Cross Domain Image Matching with Outlier Detection

TensorFlow implementation of the paper for unsupervised cross domain image matching (e.g. google street views match to old city photos), and detecting outliers in the target domain at the same time.

## Prerequisites
- Linux or maxOS
- Python3
- NVIDIA GPU + CUDA CuDNN

## Get Started
### Installation
### Train a model
python train.py --train\_DomainS\_path ./datasets/xx.txt --train\_DomainT\_path ./datasets/xx.txt --checkpointPath ./checkpoints --tr\_DS\_data ./datasets/Source --tr\_DT\_data ./datasets/Target

### Test
python test.py --databaseSetPath ./datasets/xx.txt --databaseImagePath ./datasets/Database --querySetPath ./datasets/xx.txt --queryImagePath ./datasets/Query --checkpointModel ./checkpoints/model_epoch
