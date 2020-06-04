#!/usr/bin/env python
# coding: utf-8

# In[1]:


##from tool.utils import *
##from tool.darknet2pytorch import Darknet
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from extract_kitti_label import *
from our_dataset import Image_dataset
from tqdm import tqdm, trange
import result_holder
import our_model
from global_variable import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[2]:


def running(dataset, epoch, mode, batch_size=3):
    
    if mode=="training":
        model.train()
        shuffle=True
    else:
        model.eval()
        shuffle=False
    #using default collate_fn, need convert from float64 to float32 explicitly for float tensor
    #Long tensor (i.e. contains int) and Bool tensor is ok by default!
    dataloader=DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=None, #use default
                            num_workers=0)
    #counter=float("inf")
    counter =  0.0500001
    counter_step = round(counter, 4)
    num_of_batch = len(dataloader)
    description="{} - Epoch {}".format(mode, epoch)
    total_loss = 0
    for bi, (file, frame, flip, bbox1, bbox2, x_inp, x_fmap, y_distance) in tqdm(enumerate(dataloader), total=len(dataloader), desc=description):
        curr_batch_size = len(file)
        # float64 -> float32
        x_inp, x_fmap, y_distance = x_inp.float().to(device), x_fmap.float().to(device), y_distance.float().to(device)          
        
        ## mask only!
        if model.in_channel==2:
            x_inp = x_inp[:,3:,:,:] #(B,5,608,608) -> (B,2,608,608)

        if mode in ["training"]:
            predicted_distance = model(x_inp, x_fmap) #(B,1)
            predicted_distance.squeeze_(-1) #(B,)
            #clamp GT to avoid gradient explode
            loss = criteria_custom_MSE(predicted_distance, y_distance.clamp(min=0.2))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                predicted_distance = model(x_inp, x_fmap) #(B,1)
                predicted_distance.squeeze_(-1) #(B,)
                loss = criteria_MSE(predicted_distance, y_distance)
        
        predicted_distance = predicted_distance.detach().cpu() #(B,)
        loss = loss.detach().cpu().item()
        total_loss += loss
        
        bbox1 = [ [int(t[i].item()) for t in bbox1] for i in range(curr_batch_size) ]
        bbox2 = [ [int(t[i].item()) for t in bbox2] for i in range(curr_batch_size) ]
        for i in range(curr_batch_size):
            predictions_holder.push( mode, (epoch, file[i].item(), frame[i].item(), flip[i].item(), str(bbox1[i]), str(bbox2[i])), (round(predicted_distance[i].item(), 5), round(y_distance[i].item(), 5)))

        if mode in ['training'] and bi/num_of_batch > counter: #breakpoint saving
            save_model( "model_e{}_f{}.pkl".format(epoch, round(counter,2)) )
            history_holder.push(mode, (epoch,round(counter,2)), total_loss/len(dataset)/counter)
            print('Saved at breakpoint: Epoch {}, frac {}, where approx_total_loss={}'.format(epoch, round(counter,2), total_loss/len(dataset)/counter))
            counter += counter_step

    history_holder.push(mode, (epoch,1.0), total_loss/len(dataset))
    
        
        
        


# In[3]:


def custom_MSELoss(predicted_distance, y_distance):
    a,b,c  = 1,4/20, 50
    norm_factor = (-0.499 + a*torch.sigmoid( y_distance*b ))*c
    loss = (predicted_distance-y_distance)**2/norm_factor
    return loss.mean()
    


# In[4]:


def load_dataset(files:list, dataset_path=DATASET_HUMAN_PATH):
    return Image_dataset.concat_datasets([torch.load(dataset_path+'/'+'video{}_db.pt'.format(file)) for file in files], TRAIN=True)


# In[5]:


def save_model(fname):
    torch.save(model.state_dict(), MODEL_WEIGHT_SAVE_PATH+"/"+fname)


# In[6]:


debug_db = load_dataset([12])


# In[7]:


## this part need roughly 4~5 GB RAM
train_db = load_dataset([13,16,19])
valid_db = load_dataset([15])
test_db = load_dataset([17])


# In[ ]:


## training preparation
POSTFIX = "_" + "maskonly_customMSE"
MODEL_WEIGHT_SAVE_PATH = "./model_weight" + POSTFIX

model = our_model.Image_model_by_distance(in_channel=2) # in_channel=2 <-> mask only
#model.load_state_dict(torch.load(MODEL_WEIGHT_SAVE_PATH+"/"))
model = model.to(device)
LR = 5e-6
#LR = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criteria_MSE = torch.nn.MSELoss().to(device)
criteria_custom_MSE = custom_MSELoss


history_holder = result_holder.holder(["Epoch","Fraction"],["Loss"], save_dir="./history"+POSTFIX)
predictions_holder = result_holder.holder(["Epoch", "File", "Frame", "Flip", "bbox1", "bbox2"],
                                            ["Distance", "Ground_Truth"], 
                                            save_dir="./prediction"+POSTFIX)
batch_size=8
START_epoch = 0
max_epoch = 101


if not os.path.exists(MODEL_WEIGHT_SAVE_PATH):
    os.mkdir(MODEL_WEIGHT_SAVE_PATH)
    
for epoch in range(max_epoch):
    epoch += START_epoch
    try:
        #running(debug_db, epoch, "training", batch_size)
        running(train_db, epoch, "training", batch_size)
        running(valid_db, epoch, "validation", batch_size)
    except:
        #save_model("model_e{}_WARN.pkl".format(epoch))
        history_holder.save()
        predictions_holder.save()
        raise
    save_model("model_e{}.pkl".format(epoch))
    history_holder.save()
    predictions_holder.save()
        
    #save_model("model_e{}.pkl".format(epoch))


# In[ ]:


raise EOFError("End of file: Normal Termination.")


# # Below is debug region ...

# In[ ]:


running(debug_db, 0, "validation", batch_size)
history_holder.save()
predictions_holder.save()


# In[ ]:


x = torch.ones((4))
sample = [x, 2*x,3*x]
print(sample)
[ [t[i].item() for t in sample] for i in range(3) ]


# In[ ]:


s#check data_augmentation order
for i in trange(0, len(train_db), 2):
    assert train_db[i][2]==train_db[i+1][2]


# In[ ]:


print(len(train_db)) 


# In[ ]:


dataloader = DataLoader(dataset=train_db,
                            batch_size=3,
                            shuffle=False,
                            num_workers=0)
for x,y,z in dataloader:
    print(x.shape)
    print(y.shape)
    print(z.shape)
    break


# In[ ]:


#train2_db = Image_dataset(YOLO_model, [0], concat_original=False, data_augmentation=True)
#print(len(train2_db)) #3898


# In[ ]:




