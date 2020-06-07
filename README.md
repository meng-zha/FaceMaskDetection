# FaceMaskDetection
2019-2020 PR project

## Data
Download AIZOO data from [AIZOO](https://cloud.tsinghua.edu.cn/d/af356cf803894d65b447/?p=%2FAIZOO&mode=list). Then unzip it.

Edit the val/test_00000306.xml because the class name in it is "fask_nask". Correct it to "face_mask".

## MODULE
* create.py: create the train/val/test split.
* voc2coco.py: convert the voc style data to coco style data.
* dataset.py: dataset of AIZOO if you want to use your model.
* mmdetection/: the codes forked from [open-mmlab](https://github.com/open-mmlab/mmdetection), train and test of retinanet and faster rcnn.
* PyTorch-YOLOv3/: the codes forked from [eriklindernoren](https://github.com/eriklindernoren/PyTorch-YOLOv3), train and test of yolov3.

## Faster RCNN and RetinaNet
##### Clone and install requirements
```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y
git clone https://github.com/meng-zha/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install -v -e .
```

##### Data preparation
1. transfer the voc-xml data to coco-style:
```
python voc2coco.py $AIZOO_PATH/train ./imagesets/train_split.txt $AIZOO_PATH/annotations/instances_train.json 0
python voc2coco.py $AIZOO_PATH/train ./imagesets/val_split.txt $AIZOO_PATH/annotations/instances_val.json 4918
python voc2coco.py $AIZOO_PATH/val ./imagesets/test_split.txt $AIZOO_PATH/annotations/instances_test.json 6120
```
2. replace the data root path of yours in [configs/_base_/datasets/coco_detection.py](https://github.com/meng-zha/mmdetection/blob/master/configs/_base_/datasets/coco_detection.py), [configs/ssd/ssd512_coco.py](https://github.com/meng-zha/mmdetection/blob/master/configs/ssd/ssd512_coco.py).
#### Retinanet
##### Train
To train on the AIZOO dataset run:
```
$ ./tools/train.sh configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py
```

##### Test
To test:
```
$ ./tools/test.sh configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py $CHECKPOINTS_DIR $NUM_OF_MODEL
```
##### Inference
To get the result of the test images, please run:
```
$ python demo/image_demo.py $IMG_PATH config
s/scratch/faster_rcnn_r50_fpn_gn-all_scratch_6x_coco.py $CHECKPOINTS_PATH
```

#### Faster RCNN
Your can replace _"configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py"_ by _"configs/scratch/faster_rcnn_r50_fpn_gn-all_scratch_6x_coco.py"_ to use the command above to train and test the Faster RCNN model.

------------------------------------------
## YOLOv3
##### Clone and install requirements
    $ git clone https://github.com/meng-zha/PyTorch-YOLOv3
    $ cd PyTorch-YOLOv3/
    $ sudo pip3 install -r requirements.txt

##### Data preparation
Edit the root_path in [custom.data](https://github.com/meng-zha/PyTorch-YOLOv3/blob/master/config/custom.data)

##### Train
To train on the AIZOO dataset run:
```
$ python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data
```

##### Test
To test:
```
$ python test.py --weights_path $CHECKPOINTS_PATH --model_def config/yolov3-custom.cfg --class_path data/custom/classes.names --data_config config/custom.data  --mode=test
```
To evaluate:
```
$ python test.py --weights_path $CHECKPOINTS_PATH --model_def config/yolov3-custom.cfg --class_path data/custom/classes.names --data_config config/custom.data  
```
##### Inference
To get the result of the test images, please move the images to [/data/samples/](https://github.com/meng-zha/PyTorch-YOLOv3/tree/master/data/samples).
Then run:
```
$ python detect.py --weights_path $CHECKPOINTS_PATH --model_def config/yolov3-custom.cfg --class_path data/custom/classes.names
```

## Result
Our result can be download from [TsinghuCloud](https://cloud.tsinghua.edu.cn/d/f013602590fc4030aecd/).

The best model of faster rcnn is in [faster_rcnn_batch_1_best](https://cloud.tsinghua.edu.cn/d/e6f901601a5f493fad8e/).

The best model of retinanet is in [retinanet_baseline](https://cloud.tsinghua.edu.cn/d/cadc5ae7179e4f17b201/).

The best model of yolov3 is in [yolov3_baseline](https://cloud.tsinghua.edu.cn/d/b7825aede9c948d386a2/).

## Contact
Please don't contact me if you find some bugs. If not, my mailbox will be overflowing.