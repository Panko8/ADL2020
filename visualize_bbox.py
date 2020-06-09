import cv2
import numpy as np
import random
import sys

'''
cv2.destroyAllWindows()
img = cv2.imread(r"D:/AI/Final project/Kitti/ote2012/data_tracking_image_2/training/image_02/0000/000000.png")
label_line = 3
with open(r"D:/AI/Final project/Kitti/ote2012/data_tracking_label_2/training/label_02/0000.txt", "r") as f:
    for i, line in enumerate(f):
        if i==label_line-1: # on the given line
            line = line.split()
            assert line[2]!='DontCare', "Don't care"
            # since the program can never draw precision below pixel, need convert to int
            bbox = tuple(map(lambda x: int(float(x)),line[6:10])) #len 4, left, top, right, bottom
            size_3d = line[10:13] #len 3
            coordinate = line[13:16] #len 3
            break

color = (0,0,255) # (B,G,R) wierd...
drawn = img.copy()

cv2.rectangle(drawn, bbox[0:2], bbox[2:4], color, 2) 
cv2.imshow("Image", img)
#cv2.waitKey(0)
cv2.imshow("Drawn", drawn)
'''

def visualize_image_with_label_line():
    pass

def visualize_image_with_bbox(img, x1,y1,x2,y2):
    img = cv2.imread(img)
    bbox=(x1,y1,x2,y2)
    color = (0,0,255) # (B,G,R) wierd...
    drawn = img.copy()
    cv2.rectangle(drawn, bbox[0:2], bbox[2:4], color, 2) 
    cv2.imshow("Drawn", drawn)
    cv2.waitKey(0)

def visualize_image_with_bbox_and_distance(img, boxes, lines, color_ranking, bbox_color, delay=1, bbox_thick=2, line_thick=2):
    #bbox=(x1,y1,x2,y2)
    color = (0,0,255) # (B,G,R) wierd...
    drawn = img.astype("uint8")
    #cv2.imshow('ori',img)
    for bbox in boxes:
        cv2.rectangle(drawn, bbox[0:2], bbox[2:4], bbox_color, bbox_thick)
    for p1,p2 in lines:
        distance = lines[(p1,p2)]
        color = color_ranking(distance)
        if color == None:
            continue
        cv2.line(drawn, p1, p2, color, line_thick)
    if delay!=0:
        cv2.imshow("Drawn", drawn)
        cv2.waitKey(delay)
    return drawn

if __name__=="__main__":
    if len(sys.argv)==6:
        img = sys.argv[1]
        x1,y1,x2,y2 = list(map(lambda x: int(float(x)), sys.argv[2:]))
        visualize_image_with_bbox(img, x1,y1,x2,y2)
    else:
        raise TypeError("Usage: python visualize.py img_file x1 y1 x2 y2")
