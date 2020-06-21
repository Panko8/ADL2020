
import sys
sys.path += ['D:\\AI\\Final project\\YOLOv4_pretrained\\Attempt']
from extract_kitti_label import parse_file
from global_variable import *
from tool.utils import *
from PIL import Image

# map to Person, Car, Truck in data/coco.names
#YOLO_LABEL={"Person_sitting":0, "Pedestrian":0, "Cyclist":0, "Car":2, "Truck":7, "Van":7}

YOLO_LABEL={"Person_sitting":0, "Pedestrian":0, "Cyclist":0, "Car":2, "Truck":1, "Van":1}
LOCAL_IMAGE_PATH_PREFIX="../../Kitti/ote2012/data_tracking_image_2"
#Local
IMAGE_PATH_PREFIX=LOCAL_IMAGE_PATH_PREFIX
#Colab
#IMAGE_PATH_PREFIX="/content/drive/My Drive/DL/Kitti_Dataset_ote2012"


def to_yolo_bbox_scale(W, H , bbox):
    X1,Y1,X2,Y2 = bbox
    x1 = (X2+X1)/2/W
    y1 = (Y2+Y1)/2/H
    x2 = (X2-X1)/W
    y2 = (Y2-Y1)/H
    bbox = tuple(map(lambda x: round(x,3), (x1,y1,x2,y2)))
    return bbox
    
def main(scale=True):
    global total_out, lines
    files = [13,15,16,17,19]
    total_out=""
    for file in files:
        lines = parse_file('{:04d}.txt'.format(file))
        for frame in lines:
            if scale:
                local_img_path = LOCAL_IMAGE_PATH_PREFIX + "/" + r"training/image_02/{:04d}/{:06d}.png".format(file, frame)
                W,H = Image.open(local_img_path).size
            img_path = IMAGE_PATH_PREFIX + "/" + r"training/image_02/{:04d}/{:06d}.png".format(file, frame)
            frame_data=""
            for obj in lines[frame]:
                if obj['class'] in YOLO_LABEL:
                    typ = obj['class']
                    #x1,y1,x2,y2 = obj['bbox']
                    if scale:
                        x1,y1,x2,y2 = to_yolo_bbox_scale(W,H,obj['bbox'])
                    else:
                        x1,y1,x2,y2 = obj['bbox']
                    frame_data += " {},{},{},{},{}".format(x1,y1,x2,y2,YOLO_LABEL[typ])
            if frame_data!="":
                total_out += img_path + frame_data + '\n'
    with open("../YOLOv4/train.txt", "w") as f:
        f.write(total_out)         
                    
if __name__ == "__main__":
    main(scale=False)
