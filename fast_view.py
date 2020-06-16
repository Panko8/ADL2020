import cv2
import os
import numpy as np
from PIL import Image

INPUT_IMAGES_DIR = r"D:/AI/Final project/Kitti/ote2012/data_tracking_image_2/testing/image_02/0018"
IMG_SIZE = (1224,370)

input_gen = (INPUT_IMAGES_DIR+'/'+file for file in os.listdir(INPUT_IMAGES_DIR))
fourcc = cv2.VideoWriter.fourcc('H','2','6','4')
out_video = cv2.VideoWriter('output.mp4', fourcc, 20.0, IMG_SIZE)
while True:
    try:
        img = next(input_gen)
    except:
        break
    img = Image.open(img).convert("RGB")
    assert img.size == (1224,370), "Wrong size {}".format(img.size)
    cv2img = np.array(img)[:,:, ::-1]
    cv2.imshow("fastest",cv2img)
    cv2.waitKey(1)
    out_video.write(cv2img.astype('uint8'))
out_video.release()
cv2.destroyAllWindows()
