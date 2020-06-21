import torch
import cv2
import numpy as np
from global_variable import *
from PIL import Image
from tool.utils import *
from tool.darknet2pytorch import Darknet
import os
from our_dataset import Image_dataset
from our_model import Image_model_by_distance
from visualize_bbox import visualize_image_with_bbox_and_distance as visualize

INPUT_IMAGES_DIR = r"./data_tracking_image_2/training/image_02/0015" #16 18 19! 26
SAMPLE_VIDEO_PATH = r"D:/AI/Final project/YOLOv4_pretrained/Attempt/videos/samples/taipei_walk_trim.mp4"
MODE = 1  #1=images, 2=video

MAX_SCREEN = 1500

OUR_MODEL_WEIGHT_PATH = r"./model_weight_video_w3s2/model_e56.pkl"
COLOR_TABLE = {0:None, 1:(0,255,0), 2:(0,255,255), 3:(0,0,255)} #(B,G,R) format for cv2, get color from ranking
BBOX_COLOR = (255,128,128)
WINDOW=3
STRIDE=2
YOLO_THRESHOLD = 0.3
OUTPUT_VIDEO_DIR = r"./video19"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda = True if device=='cuda' else False

INTERVAL = int(WINDOW//2)*STRIDE
resized_w = resized_h = 608
class_names = load_class_names(r"./data/coco.names")
txt_save_path="evaluation"


assert WINDOW%2==1, "Only support odd window"

def color_ranking(distance): # get color from distance (3:red, 2:yellow, 1:green, 0:None)
    rank = 3 if distance<=3 else 2 if distance<=5 else 1 if distance<=6 else 0
    color = COLOR_TABLE[rank]
    return color

def sample_video_loader(video_path, stride=10, start=200, end=-1):
    vidcap = cv2.VideoCapture(video_path)
    progress = True
    count = 0
    while progress:
        progress,frame = vidcap.read()
        if count%stride == 0 and count>=start and (end==-1 or count<=end):
            yield frame
        count += 1
    

class image_serializer():
    def __init__(self, window=WINDOW, stride=STRIDE):
        self.images={}
        self.cursor=[]
        self.window=window
        self.stride=stride
        self.interval=int(window//2)*stride
        self.max_length = self.interval*2+1
        self.center_index = self.max_length//2
    def push(self, frame, img):
        if len(self.cursor)>=self.max_length:
            self.images.pop(self.cursor[0]) #remove old image
            self.cursor = self.cursor[1:] + [frame]
            self.images[frame]=img
        else:
            self.images[frame]=img
            self.cursor.append(frame)
    def __getitem__(self, frame):
        return self.images[frame]
    def get_center(self):
        if len(self.cursor) != self.max_length: # no output
            return None
        center = self.cursor[self.center_index] # center frame number
        video_range = range(center-self.interval, center+self.interval+1, self.stride)
        imgs = []
        for curr_frame in video_range:
            tmp = self.images[curr_frame]
            img = tmp[0]
            if curr_frame==center:
                cv2img,W,H = tmp[1:]
            #img = torch.tensor(img).float().to(device)
            #img = (img-img.mean())/255
            imgs.append(img)
        img = torch.cat( imgs, -1 ) # (W,H,C)
        return center, (cv2img,W,H), img

class fmap_serializer():
    def __init__(self, window=WINDOW, stride=STRIDE):
        self.fmaps={}
        self.cursor=[]
        self.window=window
        self.stride=stride
        self.interval=int(window//2)*stride
        self.max_length = self.interval*2+1
        self.center_index = self.max_length//2
    def push(self, frame, fmap):
        if len(self.cursor)>=self.max_length:
            self.fmaps.pop(self.cursor[0]) #remove old image
            self.cursor = self.cursor[1:] + [frame]
            self.fmaps[frame]=fmap
        else:
            self.fmaps[frame]=fmap
            self.cursor.append(frame)
    def __getitem__(self, frame):
        return self.fmaps[frame]


class data_serializer():
    def __init__(self, max_length=INTERVAL*2+1):
        self.data={}
        self.cursor=[]
        self.max_length = max_length
    def push(self, frame, x):
        if len(self.cursor)>=self.max_length and frame not in self.data: #new frame + remove old frame!
            self.data.pop(self.cursor[0])
            self.cursor = self.cursor[1:] + [frame]
            self.data[frame] = []
        elif frame not in self.data: #new frame!
            self.cursor.append(frame)
            self.data[frame] = []
        self.data[frame].append(x)
    def __getitem__(self, frame):
        return self.data[frame] if frame in self.data else []

def demo():
    YOLO_model = Darknet(YOLO_CFG_FILE_PATH).to(device)
    YOLO_model.load_weights(YOLO_WEIGHT_FILE_PATH)
    YOLO_model.eval()
    model = Image_model_by_distance(in_channel=3*WINDOW+2 ).to(device)
    model.load_state_dict(torch.load(OUR_MODEL_WEIGHT_PATH))
    model.eval()
    class null:
        pass
    null_obj = null()
    null_obj.resized_w = resized_w
    null_obj.resized_h = resized_h
    input_gen = (INPUT_IMAGES_DIR+'/'+file for file in os.listdir(INPUT_IMAGES_DIR)) if MODE==1 else sample_video_loader(SAMPLE_VIDEO_PATH)
    image_holder = image_serializer()
    fmap_holder = fmap_serializer()
    data_holder = data_serializer()

    #fourcc = cv2.VideoWriter.fourcc('H','2','6','4')
    #out_video = cv2.VideoWriter(r'D://AI/Final project/YOLOv4_pretrained/Attempt/output.mp4', fourcc, 20.0, (1224,370))
    if not os.path.exists(OUTPUT_VIDEO_DIR):
        os.mkdir(OUTPUT_VIDEO_DIR)
    frame=0

    while True:
        try:
            img = next(input_gen)
        except:
            break
        if MODE==1:
            img = Image.open(img).convert("RGB")
        elif MODE==2:
            img = Image.fromarray(img[:,:,::-1])
        W, H = img.size
        if max(W,H)>1500:
            if W>H:
                img = img.resize((MAX_SCREEN, int(H/W*MAX_SCREEN)))
            else:
                img = img.resize((int(W/H*MAX_SCREEN),MAX_SCREEN))
        W, H = img.size
        resized=img.resize((resized_w, resized_h))
        with torch.no_grad():
            boxes, fmaps = do_detect_with_maps(YOLO_model, resized, YOLO_THRESHOLD, 80, 0.4, use_cuda)

        boxes, conf = unscale(W, H, boxes, class_names, return_conf=True)
        # print(boxes)
        # print(conf)
        # print(len(boxes))
        with open(txt_save_path+'/yolo_pred15.txt','a') as f:
            for i, (box, cof) in enumerate(zip(boxes,conf)):                
                a,b,c,d = box
                x1,y1,x2,y2 = transform(a,b,c,d)
                f.write(str(frame) + ' '+ str(cof)+ ' ' + 'person' + ' '+ '0' +' '+ '0' +' '+ '0'+ ' '+str(x1) +' '+ str(y1)+' '+str(x2)+' '+str(y2)+'0'+' '+'0'+' '+'0'+' ' +'0'+' '+'0'+' '+'0'+' '+'0'+' '+'0'+' '+'\n')

        #continue here!!
        continue
        
        cv2img = np.array(img)[:,:, ::-1] # convert to opencv image format, [W,H,C], where C in order [BGR]
        img = cv2.resize(cv2img, (resized_w,resized_h) )
        img = torch.tensor(img).float().to(device)
        img = (img-img.mean())/255
        #print('img.shape',img.shape)
        image_holder.push(frame, (img,cv2img,W,H))
        fmap_holder.push(frame, fmaps)
        for i in range(len(boxes)):
            bbox1 = tuple(map(int, boxes[i]))
            p1 = (int((bbox1[0]+bbox1[2])/2), int((bbox1[1]+bbox1[3])/2))
            for j in range(i, len(boxes)):
                bbox2 = tuple(map(int, boxes[j]))
                p2 = (int((bbox2[0]+bbox2[2])/2), int((bbox2[1]+bbox2[3])/2))
                data_holder.push(frame, (bbox1, bbox2, p1, p2))

        curr_img = image_holder.get_center()
        distances = {} # key:(bbox1,bbox2)
        all_boxes = set()
        if curr_img != None: # can draw something
            center, (cv2img,W,H), img = curr_img
            x_fmap = fmap_holder[center].to(device).unsqueeze(0)
            #print(cv2img.shape)
            for bbox1, bbox2, p1, p2 in data_holder[center]: #TODO: batch version inference
                all_boxes.add(bbox1)
                all_boxes.add(bbox2)
                mask1, mask2 = Image_dataset.draw_mask_and_resize(null_obj,W,H,bbox1), Image_dataset.draw_mask_and_resize(null_obj,W,H,bbox2)
                mask1, mask2 = mask1.float().to(device), mask2.float().to(device)
                x_inp = Image_dataset.concat_imgs(null_obj, [mask1,mask2], img).unsqueeze(0) #(1,C,W,H)
                with torch.no_grad():
                    distance = model(x_inp, x_fmap)
                distances[(p1,p2)] = distance           
            drawn = visualize(cv2img, all_boxes, distances, color_ranking, BBOX_COLOR, delay=1)
            #out_video.write(drawn)
            cv2.imwrite(OUTPUT_VIDEO_DIR+'/%06d.png'%frame, drawn)
        frame += 1

    #out_video.release()

def run_testing():
    pass

if __name__ == "__main__":
    demo()
