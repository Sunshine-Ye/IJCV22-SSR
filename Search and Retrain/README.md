# Searching and Retraining light-weight and real-time segmentation network via SSR based NAS framework

PyTorch code for training ERFNet model on Cityscapes. The code was based initially on the code from [bodokaiser/piwise](https://github.com/bodokaiser/piwise), adapted with several custom added modifications and tweaks. Some of them are:
- Load cityscapes dataset
- ERFNet model definition
- Calculate IoU on each epoch during training
- Save snapshots and best model during training
- Save additional output files useful for checking results (see below "Output files...")
- Resume training from checkpoint (use "--resume" flag in the command)

## Searching Files
main files: main_search.py main_search2.py main_search3.py

## Retraining Files
main files: main.py main1.py


