# TAO Experiments

This repository contains all code and instructions needed to recreate the results for each experiment shown in the TAO white paper **link**.


# Experiment Result Summary 
## Experiment 1 Data Augmentation
Data Augmentation with PCB Defects
This experiment shows how data augmentation can be used to improve an object detection model on a small dataset.

Results
### 100 Image Subset Trainings
| Dataset | Images | mAP | +Online Aug mAP |
| ------ | ------ | ------ | ------ |
| 100 x1 | 100 | 36.06% | 50.01% |
| Offline Aug 100 x10 | 1,000 | 66.31% | 75.55% |
| Offline Aug 100 x20 | 2,000 | 73.19% | 78.70% |

### 500 Image Subset Trainings
| Dataset | Images | mAP | +Online Aug mAP |
| ------ | ------| ------ | ------ | 
| 500 x1 | 500 | 82.32% | 93.01% | 
| Offline Aug 500 x10 | 5,000 | 91.34% | 95.11% |
| Offline Aug 500 x20 | 10,000 | 93.80% | 95.13% |


## Experiment 2 PeopleNet Domain Transfer
PeopleNet transfer learning to infrared images 
This experiment shows how pretrained models from NVIDIA such as peoplenet can be trained across image domains to reduce the number of images needed to train and improve the accuracy of the model.

| Dataset Size | 1256 | 2514 | 3647 | 4779 | 6288 |
| ----- | ----- | ---- | ---- | ---- | ---- |
| mAP with PeopleNet | 65.58% | 78.14% | 81.25% | 82.21% | 82.67% |
| mAP without PeopleNet | 44.81% | 63.41% | 70.89% | 73.44% | 77.27%| 

## Experiment 3 PeopleNet Add Custom Class
PeoplNet add a custom class
This experiment shows how you can take PeopleNet and add a custom class while still retaining its ability to detect people in addition to the custom class. This technique can be used on any of our pretrained models. In this experiment we add a helmet class to PeopleNet.

|Epoch | 10 | 20 | 30 | 40 | 50 | 
| ---- | --- | -- | -- | -- | -- |
|Helmet AP%| 51% | 68% | 77% | 77% | 80%|

# How To Run Experiments

This repository has been setup to recreate all results shown above using jupyter notebooks for each experiment. 

## Prerequisites

Python 3.8

TAO Launcher https://docs.nvidia.com/tlt/tlt-user-guide/text/tlt_quick_start_guide.html

Follow all steps in the TAO launcher quickstart to ensure docker and NGC are set up

Ensure the following two commands run properly

tlt --help
ngc --help

## Preparation
First Clone the repository
```
git clone ssh://git@gitlab-master.nvidia.com:12051/sochoa/tao_experiments.git
```

Download required datasets 

Experiment 1 will automatically download the dataset

Experiment 2 requires the FLIR infrared dataset. This can be downloaded from this link 
https://www.flir.com/oem/adas/adas-dataset-form/ 

The dataset comes in a split archive with 16 parts. All 16 parts must be downloaded and placed in datasets/infrared

Experiment 3 requires a helmet dataset from kaggle. This can be downoaded from this link 
https://www.kaggle.com/andrewmvd/helmet-detection
The download should be named archive.zip and must be places in datasets/helmet 

Do not rename any downloaded datasets



Start Jupyter Notebook
```
cd tao_experiments
jupyter notebook --allow-root --port=8889 --ip=0.0.0.0 &
```

## Running Experiments
The experiments can now be run by opening the experiment notebook and running all the cells. 

[Experiment 1 Notebook](workspace/pcb_data_aug/Process&Train_PCB.ipynb): /tlt_exp/pcb_data_aug/Process&Train_PCB.ipynb  
[Experiment 2 Notebook](workspace/peoplenet_helmet/Process&Train_Helmet.ipynb): /tlt_exp/peoplenet_helmet/Process&Train_Helmet.ipynb  
[Experiment 3 Notebook](workspace/peoplenet_ir/Process&Train_IR.ipynb): /tlt_exp/peoplenet_ir/Process&Train_IR.ipynb  
