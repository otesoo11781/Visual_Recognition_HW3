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

# install the latest mmcv
pip install mmcv-full==latest+torch1.6.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html

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
Download the dataset from the [Google drive](https://drive.google.com/drive/u/1/folders/1Ob5oT9Lcmz7g5mVOcYH3QugA7tV3WsSl) which is provided by TAs

Then, unzip them and put it under the **./darknet/DigitDetection/dataset/** directory

Hence, the data directory is structured as:
```
./darknet/
  +- DigitDetection/
  |  +- dataset/
     |  +- train/
          |  +- 1.jpg ...
          |  +- digitStruct.mat
     |  +- test/
          |  +- 1.jpg ...
     |  construct_datasets.py    
```

Next, process the annotation file (digitStruct.mat) into yolov4 format by running:
```
cd DigitDetection/dataset/
python construct_datasets.py
cd ../../
```
After that, you will get a txt file, which contains the corresponding bounding boxes, for each training image in train/ directory.

**Important: you can download the processed dataset from [dataset.zip](https://drive.google.com/file/d/1dlNmVJmfG9Df9z21hZwKe9hR_h-dPhuG/view?usp=sharing)**

## Transfer Training
**Important: Download the required pretrained weights and transfer trained weights by [weights.zip](https://drive.google.com/file/d/16GZVXv3TJ7jCptoKbXecIS2hxq25dr3H/view?usp=sharing)**

- **yolov4.conv.137**: pretrained on MS COCO dataset
- **yolov4-HN_best.weights**: best weights trained on SVHN dataset in 20,000 iterations.
- **yolov4-HN_final.weights**: fianl weights trained on SVHN dataset in 20,000 iterations. 

Move all the weights to the **./darknet/DigitDetection/weights/** directory.
Hence, the weights directory is structured as:
```
./darknet/
  +- DigitDetection/
  |  +- weights/
     |  +- yolov4.conv.137
     |  +- yolov4-HN_best.weights
     |  +- yolov4-HN_final.weights
```

### Retrain the yolov4 model which is pretrained on MS COCO dataset (optional)
If you don't want to spend 4 days training a model, you can skip this step and just use the **yolov4-HN_best.weights** I provided to inference. 

Now, let's transfer train the yolov4.conv.137 on SVHN dataset:

1. please ensure there is MS COCO pretrained Yolov4 model (i.e. yolov4.conv.137).

2. modify ./DigitDetection/cfg/yolov4-HN.cfg file to training mode:
```
# Training
batch=64
subdivisions=64
```

3. run the following command:
```
./darknet detector train ./DigitDetection/cfg/HN.data ./DigitDetection/cfg/yolov4-HN.cfg ./DigitDetection/weights/yolov4.conv.137 -map -gpus 0,1
```
It takes about 3~4 days to train the model on 2 RTX 2080 GPUs.

Finally, we can find the best weights **yolov4-HN_best.weights** in **./darknet/DigitDetection/weights/** directory.


## Inference
With the testing dataset and trained model, you can run the following commands to obtain the prediction results:

1. modify ./DigitDetection/cfg/yolov4-HN.cfg file to testing mode:
```
# Testing
batch=1
subdivisions=1
```
2. run yolov4 detector:
```
./darknet detector test ./DigitDetection/cfg/HN.data ./DigitDetection/cfg/yolov4-HN.cfg ./DigitDetection/weights/yolov4-HN_best.weights -thresh 0.005 -dont_show -out ./DigitDetection/result.json < ./DigitDetection/cfg/test.txt
```

After that, you will get detection results (**./DigitDetection/result.json**).

**Note**: You can test my model on [Colab notebook](https://colab.research.google.com/drive/1cdcXTFOS86gu9-ziz4vtU19kIUxt_AtG?usp=sharing). It will show a inference speed of **24.538 fps**.

**Note**: The repo has provided **result.json** which is inferred on Colab.

## Make Submission
1. Transform result.json into submission format by:
```
python ./DigitDetection/parse_result.py --input ./DigitDetection/result.json --output ./DigitDetection/0856610.json
```
2. Submit transformed **0856610.json** to [here](https://drive.google.com/drive/folders/1QNW9YvzFM7Nmg0PqUqbjgqpFyoo1wBEu).

**Note**: the repo has provided **0856610.json** which is corresponding to my submission with **mAP 0.49137**. 


