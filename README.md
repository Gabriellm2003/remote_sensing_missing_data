# Facing the Void: Overcoming Missing Data in Multi-View Remote Sensing Imagery

## Overview
This repository contains code developed for the paper "Facing the Void: Overcoming Missing Data in Multi-View Remote Sensing Imagery". 

It contains all the source code of the framework proposed for the paper. It also contains the dataset division proposed for this work, so it can be used as a baseline for future multi-view missing data experiments.

An overview of the framework can be checked in the image below:

![alt text](images/./framework.png)

### Retrieval Network
The architecture of the network integrated into this framework can be checked below:
![alt text](images/./architecture.png)

## Instructions to run the code

Before running any step it is necessary to install all the requirements listed in 'requirements.txt'.

### Image Classification
All the necessary code to train the classification models are located in the folder 'classification'. 

To train classification models, run the following command:

```diff
python classification/train.py --dataset_path <PATH_TO_DATASET_FOLDER> --output_path <PATH_TO_FOLDER_THAT_RESULTS_WILL_BE_SAVED> --batch <BATCH_SIZE> --epochs <TOTAL_EPOCHS> --network_type <CHOOSE_BETWEEN:[resnet,vgg,densenet,alexnet,squeezenet,inception,seresnet,sknet]> --early_stop <EPOCHS_WITHOUT_IMPROVING_TO_STOP_TRAINING> --fine_tunning_imagenet <[True/False]> --image_type <[aerial/ground]>
```

### Retrieval
All the necessary code to train the retrieval models are located in the folder 'metric_learning'.

To train retrieval models, run the following command:

```diff
python metric_learning/train.py --aerial_path <PATH_TO_AERIAL_IMAGES_DATASET> --ground_path <PATH_TO_GROUND_IMAGES_DATASET> --output <PATH_TO_FOLDER_THAT_RESULTS_WILL_BE_SAVED> --epochs <TOTAL_EPOCHS>
```

To get the best models using the traning log, run the following command:

```diff
python metric_learning/select_best_model_from_log.py --log_path <PATH_TO_TRAIN_LOG_FILE>
```

To 
