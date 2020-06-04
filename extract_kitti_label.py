
from global_variable import *
import os

str_to_int=lambda x: int(float(x))

def parse_file(file='0000.txt'):
    '''
    Parse a label file for a given video {file}
    contents--
        keys: frame_number
        values: all_useful_information_of_every object_in_that_frame
    '''
    fname = KITTI_LABEL_DIR + '/' + file
    lines={} #keys:frame, values:list of dict
    max_frame = int(os.listdir(KITTI_VIDEO_DIR + '/' + file.split(".")[0])[-1].split(".")[0])
    for frame in range(max_frame+1):
        lines[frame] = []
    with open(fname, 'r') as f:
        for i, line in enumerate(f):
            line=line.split()
            #class_name=line[2]
            frame=int(line[0])
            if line[2]!='DontCare':
                line_content={}
                line_content['file']=file
                line_content['line']=i+1
                line_content['frame']=frame
                line_content['obj_id']=int(line[1])
                line_content['class']=line[2]
                # x1, y1, x2, y2
                line_content['bbox']=tuple(map(str_to_int, line[6:10]))
                #height, width, depth
                line_content['size']=tuple(map(float, line[10:13]))
                line_content['gt_xyz']=tuple(map(float, line[13:16]))
                lines[frame].append(line_content)
    return lines

def calc_distance(coordinate1:tuple, coordinate2:tuple):
    '''
    calculate Euclidean distance given 2 coorindates in real 3D space.
    '''
    assert len(coordinate1)==len(coordinate2), 'bad input for calc_distance'
    out=0
    for i in range(len(coordinate1)):
        out += (coordinate1[i]-coordinate2[i])**2
    return out**0.5


def calc_IOU(bbox1:tuple, bbox2:tuple):
    '''
    Calculate IOU of two bboxs
    Accept format for bbox: tuple with order (x1,y1,x2,y2)
    
    Modified from the following code
        Original author: https://blog.csdn.net/leviopku/java/article/details/81629492
    '''
    assert len(bbox1)==len(bbox2)==4, 'Bad input bbox'
    area1 = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
    area2 = (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
    
    # area_sum
    area_sum=area1+area2
 
    # find the each edge of intersect rectangle
    left_line = max(bbox1[1], bbox2[1])
    right_line = min(bbox1[3], bbox2[3])
    top_line = max(bbox1[0], bbox2[0])
    bottom_line = min(bbox1[2], bbox2[2])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0 #no intersection
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        union = area_sum-intersect
        return intersect/union


def print_paired_distance_one_frame(lines:dict, frame:int, printing=True):
    '''
    Print all distances in Kitti dataset, given parsed_lines and one frame number.
    
    This function is for visualization only.
    '''
    if frame not in lines:
        #print("frame {} not in lines, skipped...".format(frame))
        raise TypeError("frame {} not in lines, skipped...".format(frame))
        return
    else:
        ##out=[]
        line=lines[frame]
        n=number_of_objects=len(line)
        for i in range(n):
            for j in range(i+1,n):
                obj1, obj2 = line[i], line[j]
                distance = calc_distance(obj1['gt_xyz'], obj2['gt_xyz'])
                ##out.append({"objs":{obj1, obj2}})
                if printing:
                    print("{}{} vs {}{} | distance={}".format(obj1['class'], obj1['obj_id'], obj2['class'], obj2['obj_id'], distance))

def get_coordinate_by_yolo_box(lines:dict, frame:int, bbox:tuple, class_limit=None, threshold=0.5, printing=False):
    '''
    Used only for downstream model *training*
    This function assigns a object in Kitti's ground truth to the input bbox generated by YOLO, then return the 3D coordinate
    It matched the objcet with "MAXIMUM IOU"! (with a threshold)
    
    **Note that this method can be wrong if any 2 bbox are too close to each other!!
    **Note that the YOLO bbox should be unscaled beforehand!!

    *To get the distance, apply this function on 2 bbox (from yolo), then call "calc_distance" on them
    Accept format for bbox: tuple with order (x1,y1,x2,y2)

    Argument:
            class_limit: list of str
                If given, only consider box of type within class_limit! (support class type in Kitti only)
                e.g. class_limit=['Cyclist', 'Pedestrian', 'Person_sitting']
        
    Output:
        (coordinate, confidence)
        coodinate: Real 3D coodinate of shape (height, width, depth)
        confidence: iou calculated of the returned object
    '''
    if frame not in lines:
        raise TypeError("frame {} not in lines, skipped...".format(frame))
        return
    else:
        line=lines[frame]
        n=number_of_objects=len(line)
        ious={}
        bbox1=bbox
        for i in range(n):
            obj=line[i]
            
            if class_limit!=None:
                if obj['class'] not in class_limit: #withdraw all bbox of not class_limit
                    continue
            
            bbox2=obj['bbox']
            iou=calc_IOU(bbox1,bbox2)
            ious[i]=iou
        if len(ious)==0 and class_limit!=None:
            if printing:
                raise TypeError('None of the object matches the given class_limit={}'.format(class_limit))
            return None
        most_probable_box = max(ious, key=ious.get)
        iou_max = ious[most_probable_box]
        if iou_max < threshold:
            if printing:
                raise TypeError("Can't find any object in Kitti ground truth that matches the given bbox!\nThe most probable object has IOU={}, which is below threshold={}".format(iou_max, threshold))
            return None
        else:
            most_probable_box = line[most_probable_box]
            if printing:
                print("find box of class {} with IOU={}".format(most_probable_box['class'], iou_max))
            #print(most_probable_box['bbox'])
            return most_probable_box['gt_xyz'], iou_max


def demo():
    lines=parse_file('0000.txt')
    #the below bbox are exactly written in Kitti's label file. It corresponds to a "Cyclist"
    print('Return format: (3D_coodinate, IOU_confidence)\n')
    print(  get_coordinate_by_yolo_box(lines, 0, (737.619499, 161.531951, 931.112229, 374.000000), printing=True)  )
    print(  get_coordinate_by_yolo_box(lines, 0, (737.619499, 161.531951, 931.112229, 374.000000), class_limit=['Cyclist', 'Car'], printing=True)  )
    print('\nThe class_limit argument restricts the class being matched with. i.e., the following line causes an error')
    print(  get_coordinate_by_yolo_box(lines, 0, (737.619499, 161.531951, 931.112229, 374.000000), class_limit=['Pedestrian','Van'], printing=True)  )


if __name__=="__main__":
    demo()
