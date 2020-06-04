# ADL2020
ADL2020 Final Project

A transfer learning model to monitor social distance, with the help of pretrained YOLOv4 model
- The pretrained YOLOv4 model and cfg is modified from https://github.com/Tianxiaomo/pytorch-YOLOv4
- Using Kitti dataset Object Tracking Evaluation 2012 http://www.cvlibs.net/datasets/kitti/eval_tracking.php

```
├── README.md
├── __init__.py      
├── demo_YOLO.py              demo to run pytorch
├── extract_kitti_label.py    parsing Kitti labels
├── global_variable.py        define *Important* global variables
├── net_visualize.py          visualize model using tensorboard
├── our_dataset.py            construct and save dataset
├── our_model.py              downstream CNN model
├── result_holder.py          result_holder class
├── train.py                  training
├── visualize_bbox.py         visualize image with bbox
├── data            
├── tool
│   ├── config.py
│   ├── darknet2pytorch.py
│   ├── region_loss.py
│   ├── utils.py
│   └── yolo_layer.py
```

## Description
- Please download YOLO model weight, YOLO model cfg by yourself
- It is recommended to download parsed dataset, rather than construct them all by yourself
- After downloading those files, please make sure you modify those variables listed in global_variable.py
