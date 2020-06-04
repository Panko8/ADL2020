from torch.utils.data import Dataset, DataLoader
from extract_kitti_label import *
import os
from PIL import Image
from tqdm import tqdm, trange
from tool.utils import *
import torch
import cv2
import numpy as np
from global_variable import *

#KITTI_LABEL_DIR #defined in extract_kitti_label
use_cuda=True if torch.cuda.is_available() else False

class Image_dataset(Dataset): #NOT FOR REAL TIME USAGE, ONE IMAGE AT A TIMEs
    def __init__(self, YOLO_model, files_name=[0,1,2], class_limit=None, concat_original=False, data_augmentation=True, Train=True , resized_w=608, resized_h=608):
        ##self.data_augmentation = data_augmentation #flipping or not
        self.TRAIN = Train
        self.labels={}
        self.img_size={}
        self.concat_original=concat_original
        self.resized_w = resized_w
        self.resized_h = resized_h
        self.resized_images={} #key (file, frame)
        self.fmaps={} # relation between (file, frame) -> fmaps:dict [keys:layer index in yolo, values: torch.floattensor]
        ###self.masks={} # key (file, frame, bbox)
        self.data=[] #main data storage
        '''preprocessing labels'''
        for file in files_name: #choosing file=video
            raw_lines=parse_file("{:04d}.txt".format(file))
            lines=raw_lines.copy() #keys: frame number; values:list of dict
            for frame in raw_lines:
                lines[frame]=raw_lines[frame]
                if class_limit != None: #parse object class with class_limit
                    for obj in raw_lines[frame]:
                        if obj['class'] not in class_limit:
                            lines[frame].remove(obj)
            self.labels[file]=lines
            '''preprocessing images'''
            ##self.images[file]=[]
            for image_name in tqdm(  os.listdir( KITTI_VIDEO_DIR + "/{:04d}".format(file) ), desc="Construct with video {}".format(file)  ): #for every frame
                img=Image.open(KITTI_VIDEO_DIR + "/{:04d}/".format(file) + image_name).convert('RGB')
                frame = int(image_name.split(".")[0])  
                W, H = img.size
                self.img_size[(file, frame)] = (W,H)
                resized=img.resize((resized_w, resized_h))
                #self.images[file].append(img)
                _, feature_maps = do_detect_with_maps(YOLO_model, resized, 0.5, 80, 0.4, use_cuda)

                self.fmaps[(file,frame)] = feature_maps
                
                img=np.array(img)[:,:, ::-1] # convert to opencv image format, [W,H,C], where C in order [BGR]
                resized = cv2.resize(img, (resized_w,resized_h) )
                if concat_original:
                    self.resized_images[(file,frame)] = resized
                objs = self.labels[file][frame]
                for i in range(len(objs)): #for every obj1
                    obj1 = objs[i]
                    bbox1 = obj1['bbox']
                    coord1 = obj1['gt_xyz']
                    ###mask1 = self.draw_mask_and_resize(W, H, bbox1)
                    ###self.masks[(file,frame,bbox1)] = mask1
                    for j in range(i+1, len(objs)):
                        obj2 = objs[j]
                        bbox2 = obj2['bbox']
                        coord2 = obj2['gt_xyz']
                        ###mask2 = self.draw_mask_and_resize(W, H, bbox2)
                        ###self.masks[(file,frame,bbox2)] = mask2
                        distance = calc_distance(coord1, coord2)
                        # Visualize rescaled img with masks (OK!)
                        #cv2.imshow("ori", concat[:,:,0:3].astype("uint8"))
                        #cv2.imshow("flipped", np.flip(concat[:,:,0:3], [1]).astype("uint8"))
                        #cv2.imshow("mask1", concat[:,:,3])
                        #cv2.imshow("mask2", concat[:,:,4])
                        #cv2.waitKey(0)

                        if self.TRAIN: #training/validation
                            '''
                            datum = (bboxs, (file, frame, flip), [distance])
                            '''
                            datum = ((bbox1,bbox2), (file, frame, False), distance) #(file,frame) is the key pointing to fmap!
                        else: #testing
                            datum = ((bbox1,bbox2), (file, frame, False))
                        self.data.append(datum)

                        if data_augmentation: #data augmentation -- flip [we don't resave inp/fmap here]
                            if self.TRAIN:
                                datum2 = ((bbox1,bbox2), (file, frame, True), distance)
                            else:
                                datum2 = ((bbox1,bbox2), (file, frame, True))
                            self.data.append(datum2)

    def draw_mask_and_resize(self, W, H, bbox):
        '''
        Draw masks of bbox and resize to input shape
        @Arguments:
            W: width in "original" image
            H: height in "original" image
            bbox: bbox of (left, top, right, bottom) order in pixel scale
        @Return:
            mask: a mask of shape (resized_w, resized_h)
        '''
        x1,y1,x2,y2 = bbox
        mask = cv2.resize( get_mask(W, H, bbox), (self.resized_w, self.resized_h) )
        return mask
    
    
    def concat_imgs(self, masks, img=None):
        '''
        Concat 2 masks. If img is not None, also concat img at the top.
        @Arguments:
            masks: a list of masks of shape (resized_w, resized_h)
            img: an original image of shape (resized_w, resized_h, 3)
        @Return:
            x: a FloatTensor of shape (Channel, resized_w, resized_h)
        '''
        concat=np.stack( masks, -1 ) #(W, H, 2)
        if type(img)!=type(None):
            concat=np.concatenate( (img, concat), -1 ) #(W,H,5)
        concat=concat.transpose((2,0,1)) # (C=2or5, W, H)
        return torch.tensor(concat).float()
                
                    
    def flip(self, t):
        '''
        Flip on H (dim=2) and create a new tensor of shape (C,W,H) to get flipped input/fmaps
        '''
        return t.flip([2])

    def __len__(self):
        return len(self.data)
                
    def __getitem__(self, index):
        if self.TRAIN:
            (bbox1,bbox2), (file, frame, flip), distance = self.data[index]
            fmap = self.fmaps[(file,frame)]
            W,H = self.img_size[(file,frame)]
            mask1, mask2 = self.draw_mask_and_resize(W,H,bbox1), self.draw_mask_and_resize(W,H,bbox2)
            ###mask1, mask2 = self.masks[(file,frame,bbox1)], self.masks[(file,frame,bbox2)]
            img = self.resized_images[(file,frame)] if self.concat_original else None
            inp = self.concat_imgs([mask1,mask2], img)
            if flip:
                inp = self.flip(inp)
                fmap = self.flip(fmap)
            return file, frame, flip, bbox1, bbox2, inp, fmap, distance
        else:
            (bbox1,bbox2), (file, frame, flip) = self.data[index]
            fmap = self.fmaps[(file,frame)]
            mask1, mask2 = self.masks[(file,frame,bbox1)], self.masks[(file,frame,bbox2)]
            img = self.resized_images[(file,frame)] if self.concat_original else None
            inp = self.concat_imgs([mask1,mask2], img)
            if flip:
                inp = self.flip(inp)
                fmap = self.flip(fmap)
            return file, frame, flip, bbox1, bbox2, inp, fmap

    @classmethod
    def construct_datasets_human(cls, YOLO_model, files:list, OUT_DIR=r"data/human_only", overwrite=False, VALID_CLASS=['Pedestrian', 'Person_sitting', 'Cyclist']):
        """
        Construct dataset with human mask only and save them to OUT_DIR
        @Arguments:
            YOLO_model: YOLO_model object
            files: list of int, the index of videos you want to constuct
            OUT_DIR: str [Normally == DATASET_HUMAN_PATH, but you should set it by yourself, in case]
            overwrite: bool
            VALID_CLASS: list of str, the Kitti class, only bbox, distance within this argument will be constructed. If the argument is None, all classes are valid except DontCare.
        @Return: None
        """
        concat_original=True #you can explicitly set this to False, but it is recommended to construct with original image, and set it right before training
        try:
            os.makedirs(OUT_DIR)
        except FileExistsError:
            pass
        for file in files: #list of int
            #if not concat_original:
            #    if "mask{}_db.pt".format(file) not in os.listdir(OUT_DIR) or overwrite:
            #        curr_db = cls(YOLO_model, [file], class_limit=VALID_CLASS, concat_original=concat_original, data_augmentation=True)
            #        torch.save(curr_db, OUT_DIR + "/mask{}_db.pt".format(file))
            #else:
            if "video{}_db.pt".format(file) not in os.listdir(OUT_DIR) or overwrite:
                curr_db = cls(YOLO_model, [file], class_limit=VALID_CLASS, concat_original=concat_original, data_augmentation=True)
                torch.save(curr_db, OUT_DIR + "/video{}_db.pt".format(file))



    @classmethod
    def concat_datasets(cls, datasets, TRAIN):
        """
        Concat datasets of same type.
        Please make sure their data are unique by yourself or the data can be duplicated.
        @Arguments:
            datasets: list of Image_dataset object
            TRAIN: bool
        @Return:
            out: a Image_dataset object
        """
        def merge_list_of_dicts(dicts:list):
            merge={}
            for dic in dicts:
                merge.update(dic)    # modifies z with y's keys and values & returns None
            return merge
        def is_concatable(datasets):
            def equal(iterator):
               return len(set(iterator)) <= 1
            return equal([dataset.concat_original for dataset in datasets]) and equal([dataset.resized_w for dataset in datasets]) and equal([dataset.resized_h for dataset in datasets])
        assert len(datasets)>0, "Empty input"
        if not is_concatable(datasets):
            raise TypeError("Only dataset with same shape of input can be concatenated")
        out=cls(YOLO_model=None, files_name=[])
        out.TRAIN=TRAIN
        out.concat_original = datasets[0].concat_original
        out.resized_w = datasets[0].resized_w
        out.resized_h = datasets[0].resized_h
        out.labels = merge_list_of_dicts( [dataset.labels for dataset in datasets] )
        out.img_size = merge_list_of_dicts( [dataset.img_size for dataset in datasets] )
        out.resized_images = merge_list_of_dicts( [dataset.resized_images for dataset in datasets] )
        out.fmaps = merge_list_of_dicts( [dataset.fmaps for dataset in datasets] )
        out.data = sum([dataset.data for dataset in datasets], [])
        return out


if __name__=="__main__":
    from tool.utils import *
    from tool.darknet2pytorch import Darknet
    ## Construct dataset, need cuda!
    ## PLEASE MAKE SURE YOU HAVE EDIT "DATASET_HUMAN_PATH" BEFOREHAND, OR THIS SCRIPT CAN HARM YOUR DRIVE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        raise TypeError("You should better in the cuda environment to construct dataset...\nYou can explicitly delete this line to try construting with cpu.")
    YOLO_model = Darknet(YOLO_CFG_FILE_PATH).to(device)
    YOLO_model.load_weights(YOLO_WEIGHT_FILE_PATH)
    YOLO_model.eval()
    print("Load YOLO Complete. Start construction ...")
    Image_dataset.construct_datasets_human(YOLO_model, list(range(21)), OUT_DIR=DATASET_HUMAN_PATH )
    
        



        
        



            
        
        
