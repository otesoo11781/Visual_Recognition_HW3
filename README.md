# Visual_Recognition_HW3
This is howework 3 for selected topics in visual recongnition using deep learning. The goal is instance segmentation for a given tiny PASCAL VOC dataset.

I use Cascade Mask RCNN as the mask prediciton model and ResNeSt50 as backbone network.

ResNest is a state-of-the-art network in image classification, e.g. ImageNet. 

In addition, It also performs well on downstream tasks, for instance, instance segmentation. That's why I choose it as my backbone.

For the given tiny PASCAL VOC dataset, it can achieve **0.38246 mAP**.

The details please refer to original paper [Cascade Mask RCNN](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8917599) and [ResNeSt](https://arxiv.org/pdf/2004.08955.pdf).

**Important: the implementation is based on [mmdetection](https://github.com/open-mmlab/mmdetection), which is an open source object detection toolbox.**

## Hardware
The following specs were used to train and test the model:
- Ubuntu 16.04 LTS
- 2x RTX 2080 with CUDA=10.1

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Transfer Training](#transfer-training)
4. [Inference](#inference)
5. [Make Submission](#make-submission)

## Installation
The details about the installation for mmdetection can refer to [get_started.md](mmdetection/docs/get_started.md).

Run the following command to install my modifed mmdetection:

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y

# install the mmcv
pip install mmcv-full

# install mmdetection
git clone https://github.com/otesoo11781/Visual_Recognition_HW3.git
cd Visual_Recognition_HW3/mmdetection
pip install -r requirements/build.txt
pip install -v -e . # or "python setup.py develop"

# install some required packages
conda install matplotlib
conda install scipy
```

Besides, You can install mmdetection by the orignal repo [mmdetection](https://github.com/open-mmlab/mmdetection), and then download **mmdetection/configs/myconfigs** folder from my repo to the same location in original repo. 

If there is any problem about installation, please refer to the original repo of [mmdetection](https://github.com/open-mmlab/mmdetection).

## Dataset Preparation
Download the dataset from the [Google drive](https://drive.google.com/drive/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK) which is provided by TAs.

Then, unzip them and put it under the **mmdetetction/data/** directory.

Hence, the data directory is structured as:
```
./mmdetection/
  +- data/
  |  +- train_images/
     |  +- 2007_000042.jpg ...
  |  +- test_images/
     |  +- 2007_000629.jpg ...
  |  +- pascal_train.json
  |  +- test.json
```

## Transfer Training
**Important: This step is optional. If you don't want to retrain the ImageNet pretrained model, please download [my trained weights](https://drive.google.com/file/d/1UOZ7AEisLbKZkZhwHsd8rVqxsKE9J13x/view?usp=sharing).**

- **latest.pth**: my weights trained on tiny PASCAL VOC dataset (provided by TAs) with 24 epochs. 

Then, move **latest.pth** to the **./mmdetection/work_dirs/cascade_mask_rcnn_resnest/** directory.

Hence, the weights directory is structured as:
```
./mmdetection/
  +- work_dirs/
  |  +- cascade_mask_rcnn_resnest/
     |  +- latest.pth
```

### Retrain the ImageNet pretrained model on the given dataset (optional)
P.S. If you don't want to spend a half day training a model, you can skip this step and just use the **latest.pth** I provided to inference. 

Now, let's transferly train the Cascade Mask RCNN + ResNeSt on tiny PASCAL VOC dataset:

1. please ensure ./mmdetection/configs/myconfigs/cascade_mask_rcnn_resnest.py exists.

2. please check your current directory is ./mmdetection.

3. run the following training command (the last argument "2" means the number of gpus):

```
bash ./tools/dist_train.sh configs/myconfigs/cascade_mask_rcnn_resnest.py 2
```

It takes about 13 hours to train the model on 2 RTX 2080 GPUs.

Finally, we can find the final weights **latest.pth** in **./mmdetection/work_dirs/cascade_mask_rcnn_resnest/** directory.


## Inference
With the testing dataset and trained model, you can run the following commands to obtain the predicted results:

1. please check your current directory is ./mmdetection.

2. run the testing bash script (the third argument "2" means the number of gpus):

```
./tools/dist_test.sh configs/myconfigs/cascade_mask_rcnn_resnest.py ./work_dirs/cascade_mask_rcnn_resnest/latest.pth 2 --format-only --options "jsonfile_prefix=./0856610"
```

After that, you will get final segmentation result (**./mmdetection/0856610.segm.json**).


## Make Submission
1. rename the 0856610.segm.json as 0856610.json

2. submit **0856610.json** to [here](https://drive.google.com/drive/folders/1VhuHvCyz2CH4yzDreyVTwhZiOFbQB09B).

**Note**: The repo has provided **mAP_0.38246_0856610.json** which is my submission of predicted segmentation result with **0.38246 mAP**.


