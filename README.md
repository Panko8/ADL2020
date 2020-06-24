# ADL2020

ADL2020 Final Project

A transfer learning model to monitor social distance, with the help of pretrained YOLOv4 model

- The pytorch implementation of YOLOv4 is modified from <https://github.com/Tianxiaomo/pytorch-YOLOv4>
- The pretrained YOLOv4 model and cfg is the same as <https://github.com/AlexeyAB/darknet>
- Our original dataset: Kitti dataset Object Tracking Evaluation 2012 <http://www.cvlibs.net/datasets/kitti/eval_tracking.php>

```txt
├── README.md
├── __init__.py
├── calculate_MSE.py            calculate distance range specific MSE
├── calculate_precision_IOU.py  calculate AP and IoU
├── demo.py                     demo and run inference on input frames/video
├── demo_YOLO.py                a debug script to run YOLO
├── demo_getbox.py              produce bbox, conf used for calculating AP and IoU
├── extract_kitti_label.py      parsing Kitti labels
├── fastview.py                 a debug script for generating video
├── global_variable.py          define *Important* global variables
├── net_visualize.py            visualize any model using tensorboard
├── our_dataset.py              construct, parsed and save dataset
├── our_model.py                downstream CNN model
├── plot_customMSE.py           visualize effect of customMSE
├── plot_learning_curve.py      plot learning curves based on history files
├── prepare_tune_yolo.py        a script generate train.txt for YOLO fine-tuning
├── result_holder.py            result_holder class
├── train.py                    training, see comments for more information
├── visualize_bbox.py           visualize image with bbox
├── data
├── tool
│   ├── config.py
│   ├── darknet2pytorch.py
│   ├── region_loss.py
│   ├── utils.py
│   └── yolo_layer.py
```

## Warning: Before running any script, please adjust global_variable.py to fit your data location to avoid any annoying error. For example, all_db.pt should be placed in DATASET_HUMAN_PATH directory.

## Description

- Please download YOLO model weight from <https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights>
- Please also download YOLO model config file from <https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg>
- It is recommended to download parsed dataset, rather than construct them all by yourself. (If you download this, you can safely ignore kitti related variables in global_variable.py) The parsed dataset can be downloaded from <https://drive.google.com/file/d/1lzz4P0IMMMN7gy1KhJvaIebRXAcIXkct/view?usp=sharing>
- After downloading those files, please make sure you modify those variables listed in global_variable.py

## Training

- Adjust some variable at the top of train.py, such as NAME and BATCH_SIZE.
- The default setting runs normal MSE, but customMSE is defined within train.py, so it is not hard to change the criteria by yourself
- Start training by simply type ...

```bash
python train.py
```

## Inference

- Adjust all the parameters at the top of demo.py, and then simply type ...

```bash
python demo.py
```

## Ploting learning curves

- Make sure that all the history files are placed in history_xxx/ folder at current working directory, and then type ...

```bash
python plot_learning_curve.py
```
