# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from tool.darknet2pytorch import Darknet
import argparse

"""hyper parameters"""
use_cuda = True
num_classes = 80
if num_classes == 20:
    namesfile = 'data/voc.names'
elif num_classes == 80:
    namesfile = 'data/coco.names'
else:
    namesfile = 'data/x.names'


def detect(cfgfile, weightfile, imgfile, outfile):
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((m.width, m.height))
    #print("model width/height", m.width, m.height)
    for i in range(1):
        start = time.time()
        boxes, feature_maps = do_detect_with_maps(m, sized, 0.5, num_classes, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))
    ##debug
    #from torchvision import transforms
    #trans=transforms.ToTensor()
    #print("img.shape", trans(img).shape)
    #print("feature map shape", *(fmap for fmap in feature_maps))
    #assert 1==2, 'eof'
    ##
    class_names = load_class_names(namesfile)
    #person_class_id = class_names[0]
    
    plot_boxes(img, boxes, outfile, class_names)
    #print()
    #print('Number of feature maps', len(feature_maps))
    for key, value in feature_maps.items():
        print(key, value.shape)
    #print(boxes)
    


def detect_imges(cfgfile, weightfile, imgfile_list=['data/dog.jpg', 'data/giraffe.jpg']):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    imges = []
    imges_list = []
    for imgfile in imgfile_list:
        img = Image.open(imgfile).convert('RGB')
        imges_list.append(img)
        sized = img.resize((m.width, m.height))
        imges.append(np.expand_dims(np.array(sized), axis=0))

    images = np.concatenate(imges, 0)
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, images, 0.5, num_classes, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)
    for i,(img,box) in enumerate(zip(imges_list,boxes)):
        plot_boxes(img, box, 'predictions{}.jpg'.format(i), class_names)


def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, m.num_classes, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


def detect_cv2_camera(cfgfile, weightfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("./test.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    print("Starting the YOLO loop...")

    while True:
        ret, img = cap.read()
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = do_detect(m, sized, 0.5, num_classes, 0.4, use_cuda)
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))

        class_names = load_class_names(namesfile)
        result_img = plot_boxes_cv2(img, boxes, savename=None, class_names=class_names)

        cv2.imshow('Yolo demo', result_img)
        cv2.waitKey(1)

    cap.release()


def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, m.num_classes, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)

import os

def detect_folder(cfgfile, weightfile, folderpath, outfile, batch_size=5): #no render, self convert bbox
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    images = []
    images_list = []
    counter=0
    for imgfile in os.listdir(folderpath):
        counter += 1
        img = Image.open(imgfile).convert('RGB')
        sized = img.resize((m.width, m.height))
        images_list.append(img)
        images.append(np.expand_dims(np.array(sized), axis=0))
        if counter==batch_size: #start inference
            counter=0
            images_stack = np.concatenate(images, 0)
            images=[]
            for i in range(2):
                start = time.time()
                boxes, feature_maps = do_detect_with_maps(m, images_stack, 0.5, num_classes, 0.4, use_cuda) #get boxes(scaled), feature maps
                finish = time.time()
                if i == 1:
                    print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))
                    
            '''TODO:
            0. unscale boxes, see "plot_boxes" in "utils.py" for more information
            1. render boxes to mask image (2 slice of size [original image])
            2. upscaling and concat feature maps
            3. concat (1),(2),[original_images] as the input of out downstream model
            '''
                    


    class_names = load_class_names(namesfile)  
    ###plot_boxes(img, boxes, outfile, class_names)
    print()
    print('Number of feature maps', len(feature_maps))
    for key, value in feature_maps.items():
        print(key, value.shape)
    #print(boxes)

def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default=r'./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='../yolov4.weights',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfile', type=str,
                        default=r'D:/AI/Final project/Kitti/ote2012/data_tracking_image_2/training/image_02/0000/000000.png',
                        help='path of your image file.', dest='imgfile')
    parser.add_argument('-outfile', type=str,
                        default='out.png',
                        help='path of output image file', dest='outfile')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    if True:
        detect(args.cfgfile, args.weightfile, args.imgfile, args.outfile)
        # detect_imges(args.cfgfile, args.weightfile)
        # detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
        # detect_skimage(args.cfgfile, args.weightfile, args.imgfile)
    else:
        detect_cv2_camera(args.cfgfile, args.weightfile)
