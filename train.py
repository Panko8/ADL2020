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
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[ ]:


##### 若訓練發生問題，通常只需要改這邊的參數
# Batch_size, GRAM不夠時下降這個，若沒有GRAM問題:8~12是正常的 [最小值為4，4以下代表不該用那台電腦train]
BATCH_SIZE = 8
# leaning_rate
LEARNING_RATE = 5e-6 
#當START_EPOCH不為0時，會嘗試從MODEL_WEIGHT_FILE_PATH去讀取[上一個]epoch的weight
START_EPOCH = 0 
#決定百分之多少個epoch要存一次weight，只決定了存檔時機
COUNTER = 0.05  

##### 不該碰的部分
MAX_EPOCH = 100 ##決定最多會連續跑多少個epoch，這只是上限值，通常不會碰到它，設越大越好
model = our_model.Image_model_by_distance(in_channel=2) ##請不要改這個
NAME = "maskonly_customMSE" 
POSTFIX = "_" + NAME

##### 其他只影響檔名的參數
MODEL_WEIGHT_SAVE_PATH = "./model_weight" + POSTFIX  #testing時，找weight file的路徑


# In[2]:


def running(dataset, epoch, mode, batch_size=3, frac=None):
    
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
    counter =  COUNTER + 0.00000001
    counter_step = round(counter, 4)
    num_of_batch = len(dataloader)
    description="{} - Epoch {}".format(mode, epoch)
    if frac!=None:
        assert mode != "training", "Training takes no frac argument!"
        description += ", Frac {}".format(frac)
    
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
        
        if frac==None:
            for i in range(curr_batch_size):
                predictions_holder.push( mode, (epoch, file[i].item(), frame[i].item(), flip[i].item(), str(bbox1[i]), str(bbox2[i])), (round(predicted_distance[i].item(), 5), round(y_distance[i].item(), 5)))
        else: # testing wrapper
            for i in range(curr_batch_size):
                predictions_holder.push( mode, (epoch+frac, file[i].item(), frame[i].item(), flip[i].item(), str(bbox1[i]), str(bbox2[i])), (round(predicted_distance[i].item(), 5), round(y_distance[i].item(), 5)))

        if mode in ['training'] and bi/num_of_batch > counter: #breakpoint saving
            save_model( "model_e{}_f{}.pkl".format(epoch, round(counter,2)) )
            history_holder.push(mode, (epoch,round(counter,2)), total_loss/len(dataset)/counter)
            print('Saved at breakpoint: Epoch {}, frac {}, where approx_total_loss={}'.format(epoch, round(counter,2), total_loss/len(dataset)/counter))
            counter += counter_step
    
    if mode=="training":
        history_holder.push(mode, (epoch, 1.0), total_loss/len(dataset))
    else:
        history_holder.push(mode, (epoch,frac), total_loss/len(dataset))
    
        
        
        


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


# In[10]:


def save_raw_holders(holders:list, fname, overwrite=False):
    if not os.path.exists("result_holders"):
        os.mkdir("result_holders")
    out_name = "result_holders" + "/" + fname
    assert not os.path.exists(out_name) or overwrite, "Can't over write {}".format(out_name)
    with open(out_name, 'wb') as f:
        pickle.dump(holders, f)


# In[ ]:


def load_raw_holders(fname):
    out_name = "result_holders" + "/" + fname
    with open(out_name, 'rb') as f:
        data = pickle.load(f)
    return data


# In[6]:


debug_db = load_dataset([12])


# In[7]:


## this part need roughly 4~5 GB RAM
train_db = load_dataset([13,16,19])


# In[7]:


valid_db = load_dataset([15])
test_db = load_dataset([17])


# # Training

# In[9]:


#model = our_model.Image_model_by_distance(in_channel=2) # in_channel=2 <-> mask only

if START_EPOCH !=0:
    model.load_state_dict(torch.load(MODEL_WEIGHT_SAVE_PATH+"/model_e{}.pkl".format(START_EPOCH-1)))

model = model.to(device)
LR = LEARNING_RATE

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criteria_MSE = torch.nn.MSELoss().to(device)
criteria_custom_MSE = custom_MSELoss


history_holder = result_holder.holder(["Epoch", "Fraction"],["Loss"], 
                                      save_dir="./history"+POSTFIX,
                                      overwrite_file=True)
predictions_holder = result_holder.holder(["Epoch", "File", "Frame", "Flip", "bbox1", "bbox2"],
                                            ["Distance", "Ground_Truth"], 
                                            save_dir="./predictions"+POSTFIX,
                                            overwrite_file=True)
batch_size=BATCH_SIZE
START_epoch = START_EPOCH
max_epoch = MAX_EPOCH


if not os.path.exists(MODEL_WEIGHT_SAVE_PATH):
    os.mkdir(MODEL_WEIGHT_SAVE_PATH)
    
for epoch in range(max_epoch):
    epoch += START_epoch
    try:
        #running(debug_db, epoch, "training", batch_size)
        running(train_db, epoch, "training", batch_size)
        running(valid_db, epoch, "validation", batch_size)
    except:
        save_model("model_e{}_WARN.pkl".format(epoch))
        save_raw_holders([history_holder, predictions_holder], NAME+"_e{}.holder".format(epoch)) #Don't allow overwrite, delete the old one instead!
        history_holder.save()
        predictions_holder.save()
        raise
    save_model("model_e{}.pkl".format(epoch))
    history_holder.save()
    predictions_holder.save()


# # Testing

# In[11]:


def testing_wrapper(fname, epo, frac, batch_size=8):
    global model, criteria_MSE
    criteria_MSE = torch.nn.MSELoss().to(device)
    model = our_model.Image_model_by_distance(in_channel=2) # in_channel=2 <-> mask only
    model.load_state_dict(torch.load(MODEL_WEIGHT_SAVE_PATH+"/"+fname))
    model = model.to(device)
    #running(debug_db, epo, "debug", batch_size, frac=frac)
    running(valid_db, epo, "validation", batch_size, frac=frac)
    running(valid_db, epo, "testing", batch_size, frac=frac)

    
#history_holder = result_holder.holder(["Epoch", "Fraction"],["Loss"], 
#                                        save_dir="./history"+POSTFIX,
#                                        overwrite_file=True)
#predictions_holder = result_holder.holder(["Epoch", "File", "Frame", "Flip", "bbox1", "bbox2"],
#                                            ["Distance", "Ground_Truth"], 
#                                            save_dir="./prediction"+POSTFIX,
#                                            overwrite_file=True)
'''
for fname in os.listdir(MODEL_WEIGHT_SAVE_PATH):
    tmp = fname[:-4].split("_")
    epo = int( tmp[1][1:] )
    if tmp[-1].endswith("WARN"):
        continue
    if len(tmp)==3:
        frac = float( tmp[2][1:])
    else:
        frac = 1
    if 2<=epo and frac in [1, 0.5]:
        testing_wrapper(fname, epo, frac, batch_size=BATCH_SIZE)
    
history_holder.save()
predictions_holder.save()
'''


# In[ ]:


#raise EOFError("End of file: Normal Termination.")


# # Below is debug region ...

# In[ ]:




