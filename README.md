# Unsupervised Cross Domain Image Matching with Outlier Detection

TensorFlow implementation of the [master thesis](https://repository.tudelft.nl/islandora/object/uuid%3Afcd6c0f8-6618-4fdb-b8ad-e183b3a81b73?collection=education) for unsupervised cross domain image matching (e.g. google street views match to old city photos), and detecting outliers in the target domain at the same time.

## Prerequisites
- Linux or maxOS
- Python3
- NVIDIA GPU + CUDA CuDNN

## Get Started
### Installation
### Train a model
```bash
python train.py --train_DomainS_path ./datasets/xx.txt --train_DomainT_path ./datasets/xx.txt --checkpointPath ./checkpoints --tr_DS_data ./datasets/Source --tr_DT_data ./datasets/Target
```

### Test
```bash
python test.py --databaseSetPath ./datasets/xx.txt --databaseImagePath ./datasets/Database --querySetPath ./datasets/xx.txt --queryImagePath ./datasets/Query --checkpointModel ./checkpoints/model_epoch
```
