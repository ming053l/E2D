import os
import gc
import cv2
import math
import copy
import time
import random
import timm
import torch
from timm.models.efficientnet import efficientnet_b3a, tf_efficientnet_b4_ns, tf_efficientnetv2_s, tf_efficientnetv2_m,efficientnet_b7

from timm.models.convnext import *
# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score,roc_auc_score
import timm
from timm.models.efficientnet import *

# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict

#from utils.model_2dcnn_eca_nfnet_l0 import criterion, eca_nfnet_l0

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import glob

import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CONFIG = {"seed": 2022,
          "img_size":384,
          "valid_batch_size": 1,
          "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          "train_batch":8,          
          }
def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed()
test_ct_list=list(glob.glob(os.path.join("/ssd2/ming/2024COVID/test_crop", "*"))) 
df=pd.DataFrame(test_ct_list,columns=["path"])
with open("/ssd2/ming/2024COVID/filter_slice_test_dic1_05_.pickle", 'rb') as f:
    test_dic = pickle.load(f)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #to do
        #e = tf_efficientnet_b4_ns()
        e = efficientnet_b3a(pretrained=True, drop_rate=0.3, drop_path_rate=0.2)
        self.b0 = nn.Sequential(
            e.conv_stem,
            e.bn1,
            e.act1,
        )
        self.b1 = e.blocks[0]
        self.b2 = e.blocks[1]
        self.b3 = e.blocks[2]
        self.b4 = e.blocks[3]
        self.b5 = e.blocks[4]
        self.b6 = e.blocks[5]
        self.b7 = e.blocks[6]
        self.b8 = nn.Sequential(
            e.conv_head, #384, 1536
            e.bn2,
            e.act2,
        )
        self.emb = nn.Linear(1536,224)
        self.logit = nn.Linear(224,1)
    def forward(self, image):
        batch_size = len(image)
        x = 2*image-1     

        x = self.b0(x) 
        x = self.b1(x) 
        x = self.b2(x)
        x = self.b3(x) 
        x = self.b4(x) 
        x = self.b5(x) 
        x = self.b6(x) 
        x = self.b7(x) 
        x = self.b8(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)

        x = self.emb(x)
        logit = self.logit(x)
     
        return logit
        
class Covid19Dataset_valid(Dataset):
    def __init__(self, df, valid_dic, train_batch=10, img_size = 512, transforms=None):
        self.df = df
        self.valid_dic = valid_dic
        # self.file_names = df['filename'].values
        self.path = df['path'].values
        #self.labels = df['label'].values
        self.transforms = transforms
        self.img_batch=train_batch
        self.img_size = img_size
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.path[index]
        img_path_l = os.listdir(img_path)
        img_path_l_ = [file[2:] if file.startswith("._") else file for file in img_path_l]
        
        img_list = [int(i.split('.')[0]) for i in img_path_l_]
        index_sort = sorted(range(len(img_list)), key=lambda k: img_list[k])
        ct_len = len(img_list)

        start_idx,end_idx=self.valid_dic[img_path][0],self.valid_dic[img_path][1]
        
        img_sample = torch.zeros((self.img_batch, 3, self.img_size, self.img_size))
        label_sample=torch.zeros((self.img_batch, 1))
        sample_idx=[]
        if ct_len==1:
            sample_idx = [0,0,0,0,0,0,0,0]
        elif (end_idx-start_idx) > self.img_batch:
            sample_idx = self.valid_dic[img_path][3]
        print(sample_idx)
        '''
        if (end_idx-start_idx) > self.img_batch:
            sample_idx = random.sample(range(start_idx, end_idx),self.img_batch)
        elif ct_len>20:
            sample_idx = [random.choice(range(start_idx, end_idx)) for _ in range(self.img_batch)]
        else:
            sample_idx = [random.choice(range(ct_len)) for _ in range(self.img_batch)]
        # print(sample_idx)
        '''
        '''
        # Divide the range [start_idx, end_idx] into equal parts based on self.img_batch
        interval_size = (end_idx - start_idx ) // self.img_batch
        remaining_samples = (end_idx - start_idx) % self.img_batch

        sample_idx = []
        current_idx = start_idx
        if (end_idx-start_idx) > self.img_batch:
            for i in range(self.img_batch):
                # Determine the end index of the current interval
                try:
                    if i < remaining_samples:
                        current_interval_end = current_idx + interval_size
                    else:
                        current_interval_end = current_idx + interval_size
                    
                    # Add a random sample from the current interval
                    
                    sample_idx.append(random.choice(range(current_idx, current_interval_end)))
                    
                    # Move to the start of the next interval
                    current_idx = current_interval_end
                    
                except:
                    print(1)
                    pass
        elif ct_len>20:
            sample_idx = [random.choice(range(start_idx, end_idx)) for _ in range(self.img_batch)]
        elif ct_len==1:
            sample_idx = 0
        else:
            sample_idx = [random.choice(range(ct_len)) for _ in range(self.img_batch)]

        #print(sample_idx)
        '''
        for count, idx in enumerate(sample_idx):

            img_path_ = os.path.join(img_path, img_path_l_[index_sort[idx]])
            
            img = cv2.imread(img_path_)
          
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img = self.transforms(image=img)['image']
          
            
            img_sample[count] = img[:]
            
        return {
            'image': img_sample,
            'id': img_path
        }   
        
        
def prepare_loaders(CONFIG, test_df, test_dlc, data_transforms, world_seed = None, rank=None):

    valid_dataset = Covid19Dataset_valid(test_df,test_dlc,CONFIG['train_batch'], img_size=CONFIG['img_size'],
                                         transforms=data_transforms["valid"])
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG["valid_batch_size"], 
                              num_workers=15, shuffle=False, pin_memory=True)
    return  valid_loader

data_transforms = {
    "valid": A.Compose([
        A.Resize(384, 384),
        A.Normalize(),
        ToTensorV2()], p=1.)
}

@torch.inference_mode()
def inference(model, dataloader, device):
    model.eval()
    dataset_size = 0
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    IDS=[]
    pred_y=[]
    for step, data in bar:
        ids = data["id"]
        ct_b, img_b, c, h, w = data['image'].size()
        data_img = data['image'].reshape(-1, c, h, w)
        images = data_img.to(device, dtype=torch.float)
        batch_size = images.size(0)
        outputs = model(images)
        pred_y.append(torch.sigmoid(outputs).cpu().numpy())
        IDS.append(ids)
        #print(pred_y)
        #print(pred_y.shape)
    pred_y=np.concatenate(pred_y)
    IDS = np.concatenate(IDS)
    gc.collect()
    
    pred_y=np.array(pred_y).reshape(-1,1)
    pred_y=np.array(pred_y).reshape(-1,img_b)

    pred_y=pred_y.mean(axis=1)
    
    return pred_y,IDS

job=51

weights_path_list=['/ssd2/ming/2024COVID/model/auc_roc/job_58_effb7_size384_challenge[DataParallel]-fold1.bin0.9824418048946397']
for j in range(1):
    bin_save_path = "/ssd2/ming/2024COVID/model"
    #job_name = f"job_{job}_eca_nfnet_l0_size{CONFIG['img_size']}_challenge[DataParallel]-fold{j + 1}" + ".bin"
    #weights_path = f'{bin_save_path}/f1/' + f"job_{job}_eca_nfnet_l0_size{CONFIG['img_size']}_challenge[DataParallel]-fold{j + 1}" + ".bin"
    weights_path=weights_path_list[j]
    print("="*10, "loading *model*", "="*10)
    #model=eca_nfnet_l0(n_classes=2,pretrained=True)
    model=Net()
    #model = nn.DataParallel(model, device_ids=[0])
    #model = model.to(CONFIG['device'])
    scaler = amp.GradScaler()
    
    state_dict = torch.load(weights_path)  # 模型可以保存为pth文件，也可以为pt文件。
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = "module."+k[:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。 
    # load params
    
    model.load_state_dict(state_dict) # 从新加载这个模型。
    #model.load_state_dict(state_dict) # 从新加载这个模型。

    #model = nn.DataParallel(model)
    model=model.cuda()
    #test_loader=prepare_loaders()
    total_pred=[]
    test_loader = prepare_loaders(CONFIG, df, test_dic, data_transforms)

    for i in range(1):
        pred_y,name=inference(model, test_loader, device=CONFIG['device'])
        total_pred.append(pred_y)
    
    final_pred=np.mean(total_pred,axis=0)
    dict_all=dict(zip(name, final_pred))
    cnn_one_pred_df=pd.DataFrame(list(dict_all.items()),
                       columns=['path', 'pred'])
    cnn_one_pred_df.to_csv(f"/ssd2/ming/2024COVID/output/3_cnn_one_pred_{j+1}df.csv",index=False)
     
    times_list=[10]
    for times in times_list:
        total_pred=[]
        for i in range(times):
            pred_y,name=inference(model, test_loader, device=CONFIG['device'])
            total_pred.append(pred_y)
        final_pred=np.mean(total_pred,axis=0)
        dict_all=dict(zip(name, final_pred))
    
        cnn_times_pred_df=pd.DataFrame(list(dict_all.items()),
                           columns=['path', 'pred'])
        cnn_times_pred_df.to_csv(f"/ssd2/ming/2024COVID/output/3_cnn_{times}_pred_{j+1}df.csv",index=False)
        print("save")
