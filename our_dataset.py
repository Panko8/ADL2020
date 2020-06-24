from torch.utils.data import Dataset, DataLoader
from extract_kitti_label import *
import os
from PIL import Image
from tqdm import tqdm, trange
#from tool.utils import *
import torch
import cv2
import numpy as np
from global_variable import *
from copy import deepcopy
import numpy as np
from torch.nn import functional as F

#KITTI_LABEL_DIR #defined in extract_kitti_label
use_cuda=True if torch.cuda.is_available() else False
device = "cuda" if use_cuda else "cpu"


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
        self.max_frame={} #key: file
        '''preprocessing labels'''
        for file in files_name: #choosing file=video
            raw_lines = parse_file("{:04d}.txt".format(file))
            self.max_frame[file] = max(raw_lines)
            lines=raw_lines.copy() #keys: frame number; values:list of dict
            for frame in raw_lines:
                lines[frame]=raw_lines[frame].copy()
                if class_limit != None: #parse object class with class_limit
                    for obj in raw_lines[frame]:
                        if obj['class'] not in class_limit:
                            lines[frame].remove(obj)
            self.labels[file]=lines
            '''preprocessing images'''
            ##self.images[file]=[]
            for image_name in tqdm(  os.listdir( KITTI_VIDEO_DIR + "/{:04d}".format(file) ), desc="Construct with video {}".format(file)  ): #for every frame
            #for image_name in os.listdir( KITTI_VIDEO_DIR + "/{:04d}".format(file) ):
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

    #def draw_mask_and_resize(self, W, H, bbox):
    #    '''
    #    Draw masks of bbox and resize to input shape
    #    @Arguments:
    #        W: width in "original" image
    #        H: height in "original" image
    #        bbox: bbox of (left, top, right, bottom) order in pixel scale
    #    @Return:
    #        mask: a mask of shape (resized_w, resized_h)
    #    '''
    #    x1,y1,x2,y2 = bbox
    #    mask = cv2.resize( get_mask(W, H, bbox).float(), (self.resized_w, self.resized_h) )
    #    return mask

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
        def get_mask_tensor(W,H,bbox):
            mask=torch.zeros((H,W), dtype=int).to(device)
            x1,y1,x2,y2 = bbox
            x1,y1,x2,y2 = map(str_to_int, [x1,y1,x2,y2])
            mask[y1:y2+1,x1:x2+1] = 1
            return mask
        x1,y1,x2,y2 = bbox
        mask = F.interpolate( get_mask_tensor(W, H, bbox).float().unsqueeze(0), self.resized_h)
        mask = F.interpolate( mask.permute(0,2,1), self.resized_w)
        return mask.squeeze(0).permute(1,0)
    
    
    def concat_imgs(self, masks, img=None):
        '''
        Concat 2 masks. If img is not None, also concat img at the top.
        @Arguments:
            masks: a list of masks of shape (resized_w, resized_h)
            img: an original image of shape (resized_w, resized_h, 3)
        @Return:
            x: a FloatTensor of shape (Channel, resized_w, resized_h)
        '''
        concat=torch.stack( masks, -1 ) #(W, H, 2)
        if type(img)!=type(None):
            concat=torch.cat( (img, concat), -1 ) #(W,H,5)
        concat=concat.permute((2,0,1)) # (C=2or5, W, H)
        return concat.float()
                
                    
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
            mask1, mask2 = mask1.float().to(device), mask2.float().to(device)
            if self.concat_original:
                img = torch.tensor(img).float().to(device)
                img = (img-img.mean())/255
            inp = self.concat_imgs([mask1,mask2], img)
            if flip:
                inp = self.flip(inp)
                fmap = self.flip(fmap)
            return file, frame, flip, bbox1, bbox2, inp, fmap, distance
        else:
            (bbox1,bbox2), (file, frame, flip) = self.data[index]
            fmap = self.fmaps[(file,frame)]
            W,H = self.img_size[(file,frame)]
            mask1, mask2 = self.draw_mask_and_resize(W,H,bbox1), self.draw_mask_and_resize(W,H,bbox2)
            img = self.resized_images[(file,frame)] if self.concat_original else None
            mask1, mask2 = torch.tensor(mask1).float().to(device), torch.tensor(mask2).float().to(device)
            if self.concat_original:
                img = torch.tensor(img).float().to(device)
                img = (img-img.mean())/255
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
        out.max_frame = merge_list_of_dicts( [dataset.max_frame for dataset in datasets] )
        return out

    @staticmethod
    def remove_augmentation(dataset):
        """
        Remove augmentation. i.e. remove all odd index data ...
        """
        out = deepcopy(dataset)
        new_data = dataset.data[::2]
        out.data = new_data.copy()
        return out

class Video_dataset(Image_dataset): # can only be instantiated by "concat datasets"
    @classmethod
    def concat_datasets(cls, datasets, TRAIN, window, stride):
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
            return all([dataset.concat_original==True for dataset in datasets]) and equal([dataset.resized_w for dataset in datasets]) and equal([dataset.resized_h for dataset in datasets])
        assert len(datasets)>0, "Empty input"
        if not is_concatable(datasets):
            raise TypeError("Only dataset with same shape of input can be concatenated")
        out=cls(YOLO_model=None, files_name=[])
        ## New part
        assert window%2==1, "Only support odd window"
        out.window = window
        #out._window = (window-1)*2+1 # flip skipping
        out.stride = stride
        ##
        out.TRAIN=TRAIN
        out.concat_original = datasets[0].concat_original
        out.resized_w = datasets[0].resized_w
        out.resized_h = datasets[0].resized_h
        out.labels = merge_list_of_dicts( [dataset.labels for dataset in datasets] )
        out.img_size = merge_list_of_dicts( [dataset.img_size for dataset in datasets] )
        out.resized_images = merge_list_of_dicts( [dataset.resized_images for dataset in datasets] )
        out.fmaps = merge_list_of_dicts( [dataset.fmaps for dataset in datasets] )
        #out.data = sum([dataset.data for dataset in datasets], [])
        out.max_frame = merge_list_of_dicts( [dataset.max_frame for dataset in datasets] )
        ## New part
        trim_datas=[]
        #global dataset
        for dataset in datasets:
            trim_data = dataset.data.copy()
            lowest_frame = int(window//2)*(stride)
            assert len(dataset.max_frame)==1, "Cannot construct with any single dataset including 2 or more files"
            highest_frame = dataset.max_frame.popitem()[1] - lowest_frame
            for sample in dataset.data:
                if sample[1][1]<lowest_frame:
                    trim_data.remove(sample)
                else:
                    break
            for sample in dataset.data[::-1]:
                if sample[1][1]>highest_frame:
                    trim_data.remove(sample)
                else:
                    break
            trim_datas.append(trim_data)
        out.data = sum(trim_datas, [])
        ##
        return out ##TODO view count, trim inbalanced

    @classmethod
    def all_db2videoset(cls, dataset, window, stride):
        '''
        Convert Image_dataset to Video_dataset, especially all_db format
        '''
        def trim_data(dataset):
            trim_data = list(dataset.data)
            lowest_frame = int(window//2)*(stride)
            highest_frame = {}
            for file in dataset.max_frame:
                highest_frame[file] = dataset.max_frame[file] - lowest_frame
            for sample in dataset.data:
                curr_file = sample[1][0]
                curr_frame = sample[1][1]
                if curr_frame < lowest_frame or curr_frame > highest_frame[curr_file]:
                    trim_data.remove(sample)
            return trim_data
        out=cls(YOLO_model=None, files_name=[])
        assert window%2==1, "Only support odd window"
        out.window = window
        #out._window = (window-1)*2+1 # flip skipping
        out.stride = stride
        ##
        out.TRAIN = dataset.TRAIN
        out.concat_original = dataset.concat_original
        out.resized_w = dataset.resized_w
        out.resized_h = dataset.resized_h
        out.labels = dataset.labels
        out.img_size = dataset.img_size
        out.resized_images = dataset.resized_images
        out.fmaps = dataset.fmaps
        
        dataset.data = dataset.train_data
        out.train_data = trim_data(dataset)
        dataset.data = dataset.valid_data
        out.valid_data = trim_data(dataset)
        dataset.data = dataset.test_data
        out.test_data = trim_data(dataset)
        out.data=[]
        return out
        

    def __len__(self):
        '''N=len(self.data)
        #r=self._window//2
        w=self._window
        d=self.stride
        if N<w:
            return 0
        return int((N-w)//d)+1'''
        return len(self.data)
    
    def __getitem__(self, index):
        '''index = int( self._window//2 + (index*self.stride) ) #center point's index. It make sure index=0 not causing error
        if index+self._window//2 >= len(self.data): #out of range
            raise IndexError("Bad index")

        (bbox1,bbox2), (file, frame, flip), distance = self.data[index]
        
        index_range = range(index-self._window//2, index+self._window//2+1, 2) #flip skipping
        print(index_range)
        center = index'''
        if self.TRAIN:
            '''frames=[]
            for index in index_range:
                (bbox1,bbox2), (file, frame, flip), distance = self.data[index]
                fmap = self.fmaps[(file,frame)]
                W,H = self.img_size[(file,frame)]
                mask1, mask2 = self.draw_mask_and_resize(W,H,bbox1), self.draw_mask_and_resize(W,H,bbox2)
                img = self.resized_images[(file,frame)] if self.concat_original else None
                inp = self.concat_imgs([mask1,mask2], img) #(2or5, 608, 608)
                if flip:
                    inp = self.flip(inp)
                    fmap = self.flip(fmap)
                frames.append(inp)
                
            (bbox1,bbox2), (file, frame, flip), distance = self.data[center] #only center file/frame/bboxs/distance is returned
            fmap = self.fmaps[(file,frame)] #only center fmap is returned
            inp = torch.cat(frames, dim=0) #cat on channel!! --> (6or15, 608, 608)                                    
            return file, frame, flip, bbox1, bbox2, inp, fmap, distance'''
            
            (bbox1,bbox2), (file, frame, flip), distance = self.data[index]
            fmap = self.fmaps[(file,frame)]
            W,H = self.img_size[(file,frame)]
            mask1, mask2 = self.draw_mask_and_resize(W,H,bbox1), self.draw_mask_and_resize(W,H,bbox2)
            mask1, mask2 = mask1.float().to(device), mask2.float().to(device)
            interval = int(self.window//2)*self.stride
            center = frame
            video_range = range(center-interval, center+interval+1, self.stride)
            ##print("index={}, center={}, interval={}, window={}, stride={}".format(index, center, interval, self.window, self.stride))
            imgs=[]
            for cur_frame in video_range:
                img = self.resized_images[(file,cur_frame)]
                img = torch.tensor(img).float().to(device)
                img = (img-img.mean())/255
                imgs.append(img)
            img = torch.cat( imgs, -1 )# (W,H,C)
            assert img.device.type == device, "GPU conversion error"
            inp = self.concat_imgs([mask1,mask2], img) #(2or5, 608, 608)
            if flip:
                inp = self.flip(inp)
                fmap = self.flip(fmap)
            return file, frame, flip, bbox1, bbox2, inp, fmap, distance
            
        else:
            (bbox1,bbox2), (file, frame, flip) = self.data[index]
            fmap = self.fmaps[(file,frame)]
            W,H = self.img_size[(file,frame)]
            mask1, mask2 = self.draw_mask_and_resize(W,H,bbox1), self.draw_mask_and_resize(W,H,bbox2)
            mask1, mask2 = mask1.float().to(device), mask2.float().to(device)
            interval = int(self.window//2)*self.stride
            center = frame
            video_range = range(center-interval, center+interval+1, self.stride)
            imgs=[]
            for cur_frame in video_range:
                img = self.resized_images[(file,cur_frame)]
                img = torch.tensor(img).float().to(device)
                img = (img-img.mean())/255
                imgs.append(img)
            img = torch.cat( imgs, -1 )# (W,H,C)
            assert img.device.type == device, "GPU conversion error"
            inp = self.concat_imgs([mask1,mask2], img) #(2or5, 608, 608)
            if flip:
                inp = self.flip(inp)
                fmap = self.flip(fmap)
            return file, frame, flip, bbox1, bbox2, inp, fmap

    

from tool.utils import *
from tool.darknet2pytorch import Darknet
def construct_all_raw(overwrite=False):
    ## Construct dataset, need cuda!
    ## PLEASE MAKE SURE YOU HAVE EDIT "DATASET_HUMAN_PATH" BEFOREHAND, OR THIS SCRIPT CAN HARM YOUR DRIVE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        raise TypeError("You should better in the cuda environment to construct dataset...\nYou can explicitly delete this line to try construting with cpu.")
    YOLO_model = Darknet(YOLO_CFG_FILE_PATH).to(device)
    YOLO_model.load_weights(YOLO_WEIGHT_FILE_PATH)
    YOLO_model.eval()
    print("Load YOLO Complete. Start construction ...")
    Image_dataset.construct_datasets_human(YOLO_model, list(range(21)), OUT_DIR=DATASET_HUMAN_PATH, overwrite=overwrite )

def _debug_image():
    global debug_db
    def load_image_dataset(files:list, dataset_path=DATASET_HUMAN_PATH):
        return Image_dataset.concat_datasets([torch.load(dataset_path+'/'+'video{}_db.pt'.format(file)) for file in files], TRAIN=True)
    debug_db = load_image_dataset([0])

def _debug_video():
    global debug_db
    def load_video_dataset(files:list, window:int, stride:int, dataset_path=DATASET_HUMAN_PATH):
        return Video_dataset.concat_datasets([torch.load(dataset_path+'/'+'video{}_db.pt'.format(file)) for file in files], TRAIN=True, window=window, stride=stride)
    debug_db = load_video_dataset([0], window=3, stride=2)

def _debug_class_limit():
    global curr_db
    VALID_CLASS=['Pedestrian', 'Person_sitting', 'Cyclist']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    YOLO_model = Darknet(YOLO_CFG_FILE_PATH).to(device)
    YOLO_model.load_weights(YOLO_WEIGHT_FILE_PATH)
    YOLO_model.eval()
    curr_db = Image_dataset(YOLO_model, [11], class_limit=VALID_CLASS, concat_original=True, data_augmentation=True)
    #torch.save(curr_db, OUT_DIR + "/video{}_db.pt".format(file))

def construct_and_split_image_dataset():
    def load_dataset(files:list, dataset_path=DATASET_HUMAN_PATH):
        return Image_dataset.concat_datasets([torch.load(dataset_path+'/'+'video{}_db.pt'.format(file)) for file in files], TRAIN=True)
    def train_test_split_dataset(dataset, ratio=[3,1,1], seed=123):
        assert len(ratio)==3
        total = sum(ratio)
        N = len(dataset)
        n1, n2 = int(N*ratio[0]/total), int(N*ratio[1]/total)
        n3 = N-n1-n2
        train, valid, test = torch.utils.data.random_split(dataset.data, [n1,n2,n3], torch.Generator().manual_seed(seed))
        return train, valid, test
    dataset13 = load_dataset([13]) #4800
    dataset15 = load_dataset([15]) #6220
    dataset16 = Image_dataset.remove_augmentation(load_dataset([16])) #25320 -> 12660
    dataset17 = load_dataset([17]) #5522
    dataset19 = Image_dataset.remove_augmentation(load_dataset([19])) #46032 -> 23016
    dataset16.data = dataset16.data[::2] # 12660 -> 6330
    dataset19.data = dataset19.data[::4] # 23016 -> 5754
    all_datasets = Image_dataset.concat_datasets([dataset13, dataset15, dataset16, dataset17, dataset19], True)
    global train_db, valid_db, test_db
    train, valid, test = train_test_split_dataset(all_datasets)
    all_datasets.train_data = train
    all_datasets.valid_data = valid
    all_datasets.test_data = test
    all_datasets.data = []
    torch.save(all_datasets, DATASET_HUMAN_PATH + "/all_db.pt")
    #torch.save(valid_db, DATASET_HUMAN_PATH + "/valid_db.pt")
    #torch.save(test_db, DATASET_HUMAN_PATH + "/test_db.pt")
    
def read_all_db():
    all_db = torch.load(DATASET_HUMAN_PATH+'/'+'all_db.pt')
    return all_db
    
if __name__=="__main__":
    construct_and_split_image_dataset()
    #_debug_image()
        



        
        



            
        
        
