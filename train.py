import os, gc, cv2, math, copy, time, random
import pickle
# For data manipulation
import numpy as np, pandas as pd

# Pytorch Imports
import torch, torch.nn as nn, torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR
import json
from torch.cuda import amp

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score,roc_auc_score
# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix
#from imblearn.metrics import sensitive_score
#from imblearn.metrics import specificity_score
from sklearn.metrics import recall_score

from utils import *
from model import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()

    dataset_size = 0
    running_loss = 0.0
    
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.5, verbose=False)

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        ct_b, img_b, c, h, w = data['image'].size()
        data_img = data['image'].reshape(-1, c, h, w)
        data_label = data['label'].reshape(-1,1)
        images = data_img.to(device, dtype=torch.float)
        labels = data_label.to(device, dtype=torch.float)


        batch_size = images.size(0)

        with amp.autocast(enabled = True):
            outputs = model(images)

            loss = criterion(outputs, labels)

            loss = loss / CONFIG['n_accumulate']

        scaler.scale(loss).backward()

        if (step + 1) % CONFIG['n_accumulate'] == 0:

            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()


            if scheduler is not None:
               scheduler.step()


        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])

    gc.collect()

    return epoch_loss

@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    true_y=[]
    pred_y=[]
    for step, data in bar:
        ct_b, img_b, c, h, w = data['image'].size()
        data_img = data['image'].reshape(-1, c, h, w)
        data_label = data['label'].reshape(-1,1)

        images = data_img.to(device, dtype=torch.float)
        labels = data_label.to(device, dtype=torch.float)

        batch_size = images.size(0)

        outputs = model(images)
        loss = criterion(outputs, labels)


        true_y.append(labels.cpu().numpy())
        pred_y.append(torch.sigmoid(outputs).cpu().numpy())

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])

    true_y=np.concatenate(true_y)
    pred_y=np.concatenate(pred_y)

    gc.collect()

    true_y=np.array(true_y).reshape(-1,1)
    true_y=np.array(true_y).reshape(-1,img_b)
    true_y=true_y.mean(axis=1)

    pred_y=np.array(pred_y).reshape(-1,1)
    pred_y = torch.nan_to_num(torch.from_numpy(pred_y)).numpy()
    pred_y=np.array(pred_y).reshape(-1,img_b)
#     pred_y2=pred_y.max(axis=1)
    pred_y=pred_y.mean(axis=1)

    print(true_y.shape,pred_y.shape)
    assert (true_y.shape==pred_y.shape)
    acc_f1=f1_score(np.array(true_y),np.round(pred_y),average='macro')
    acc_f1_48=f1_score(np.array(true_y),np.where(pred_y>0.48,1,0),average='macro')
    acc_f1_51=f1_score(np.array(true_y),np.where(pred_y>0.51,1,0),average='macro')
    acc_f1_52=f1_score(np.array(true_y),np.where(pred_y>0.52,1,0),average='macro')
    acc_f1_54=f1_score(np.array(true_y),np.where(pred_y>0.54,1,0),average='macro')
    auc_roc=roc_auc_score(np.array(true_y),np.array(pred_y))
    print("acc_f1(mean) : ",round(acc_f1,4),"  auc_roc(mean) : ",round(auc_roc,4))


    return epoch_loss,acc_f1,auc_roc

def run_training(model, optimizer, scheduler, device, num_epochs):
        

    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    best_epoch_auc = 0
    best_epoch_f1 = 0
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler,
                                           dataloader=train_loader,
                                           device=CONFIG['device'], epoch=epoch)

        val_epoch_loss,acc_f1,auc_roc= valid_one_epoch(model, valid_loader, device=CONFIG['device'],
                                         epoch=epoch)

        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)


        if val_epoch_loss <= best_epoch_loss:
            print(f"Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss

            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f'{bin_save_path}/loss/'+job_name+str(best_epoch_loss)
            os.makedirs(f'{bin_save_path}/loss/', exist_ok=True)
            torch.save(model.module.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved")

        if auc_roc >= best_epoch_auc:
            print(f"Validation Auc Improved ({best_epoch_auc} ---> {auc_roc})")
            best_epoch_auc = auc_roc

            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f'{bin_save_path}/auc_roc/'+job_name+str(best_epoch_auc)
            os.makedirs(f'{bin_save_path}/auc_roc/', exist_ok=True)
            torch.save(model.module.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved")

        if acc_f1 >= best_epoch_f1:
            print(f"Validation f1 Improved ({best_epoch_f1} ---> {acc_f1})")
            best_epoch_f1 = acc_f1

            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f'{bin_save_path}/f1/'+job_name+str(best_epoch_f1)
            os.makedirs(f'{bin_save_path}/f1/', exist_ok=True)
            torch.save(model.module.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved")
        
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: {:.4f}".format(best_epoch_loss))
    return model, history, best_epoch_f1

@torch.inference_mode()
def pred_one(model, dataloader, device):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    true_y=[]
    pred_y=[]
    for step, data in bar:
        ct_b, img_b, c, h, w = data['image'].size()
        data_img = data['image'].reshape(-1, c, h, w)
        data_label = data['label'].reshape(-1,1)
        
        images = data_img.to('cuda', dtype=torch.float)
        labels = data_label.to('cuda', dtype=torch.float)
        
        batch_size = images.size(0)

        outputs = model(images)
        loss = criterion(outputs, labels)

        
        #print(images.shape)

        true_y.append(labels.cpu().numpy())
        pred_y.append(torch.sigmoid(outputs).cpu().numpy())
        #print(true_y.shape)

    

    true_y=np.concatenate(true_y)
    pred_y=np.concatenate(pred_y)
    
    
    #print(true_y.shape)
   
    gc.collect()
    
    true_y=np.array(true_y).reshape(-1,1)

    true_y=np.array(true_y).reshape(-1,img_b)
    true_y=true_y.mean(axis=1)


    #print(true_y.shape)
  
    pred_y=np.array(pred_y).reshape(-1,1)
    
    #print(pred_y)
    pred_y=np.array(pred_y).reshape(-1,img_b)

    #print(pred_y)
    pred_y=pred_y.mean(axis=1)
    
    #print(pred_y)
    return true_y,pred_y
    
if __name__ == '__main__':
    # Config
    set_seed()
    job=60
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    CONFIG = {"seed": 2022,
            "epochs": 3,  #24
            "img_size": 384, #512

            "train_batch_size": 20, #16
            "valid_batch_size": 1,
            "learning_rate": 0.0001,

            "weight_decay": 0.0005, 
        
            "n_accumulate": 1, #2
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            
            "train_batch":8,
            }
    # img size = 256; batch=8; f1-score mean: 0.9142
    # Data Augmenation
    data_transforms = {
    "train": A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.ShiftScaleRotate(shift_limit=0.2, 
                           scale_limit=0.2, 
                           rotate_limit=30, 
                           p=0.5),
        A.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5 
            ),
        A.RandomBrightnessContrast(
                brightness_limit=(-0.2,0.2), #0.2
                contrast_limit=(-0.2, 0.2),  #0.2
                p=0.5 
            ),
        A.dropout.coarse_dropout.CoarseDropout(p=0.2),
        A.Normalize(),
        ToTensorV2()], p=1.),
    
    "valid": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),

        A.Normalize(),
        ToTensorV2()], p=1.)
}
    
    # Get data dict
    #with open('/ssd2/ming/2024COVID/filter_slice_train_dic1_05_challenge.pickle', 'rb') as f:
    #    train_dic = pickle.load(f)
    with open('/ssd2/ming/2024COVID/kde_train_dic_challenge.pickle', 'rb') as f:
         train_dic = pickle.load(f)
    #train_dic = {key.replace('train_pure_crop_challenge', 'train'): value for key, value in train_dic.items()}

    with open('/ssd2/ming/2024COVID/kde_valid_dic_challenge.pickle', 'rb') as f:
        valid_dlc = pickle.load(f)
    
    #valid_dlc = {key.replace('valid_pure_crop_challenge', 'valid'): value for key, value in valid_dlc.items()}


    #with open('ssd8/2023COVID19/Train_Valid_dataset/test_dic1_05.pickle', 'rb') as f:
    #    test_dlc = pickle.load(f)
    not_allow=['/ssd2/ming/2024COVID/train_pure_crop_challenge/negative/ct_scan_292','/ssd2/ming/2024COVID/train_pure_crop_challenge/negative/ct_scan_354',
                '/ssd2/ming/2024COVID/train_pure_crop_challenge/negative/ct_scan_449', '/ssd2/ming/2024COVID/train_pure_crop_challenge/negative/ct_scan_538',
                '/ssd2/ming/2024COVID/train_pure_crop_challenge/positive/ct_scan_107','/ssd2/ming/2024COVID/train_pure_crop_challenge/positive/ct_scan_306',
                '/ssd2/ming/2024COVID/train_pure_crop_challenge/positive/ct_scan_31','/ssd2/ming/2024COVID/train_pure_crop_challenge/positive/ct_scan_47',
                '/ssd2/ming/2024COVID/train_pure_crop_challenge/positive/ct_scan_64','/ssd2/ming/2024COVID/valid_pure_crop_challenge/negative/ct_scan_101',
                '/ssd2/ming/2024COVID/valid_pure_crop_challenge/negative/ct_scan_130','/ssd2/ming/2024COVID/valid_pure_crop_challenge/negative/ct_scan_15',
                '/ssd2/ming/2024COVID/valid_pure_crop_challenge/positive/ct_scan_101','/ssd2/ming/2024COVID/valid_pure_crop_challenge/positive/ct_scan_18',
                '/ssd2/ming/2024COVID/valid_pure_crop_challenge/positive/ct_scan_40','/ssd2/ming/2024COVID/valid_pure_crop_challenge/positive/ct_scan_46',
                '/ssd2/ming/2024COVID/valid_pure_crop_challenge/positive/ct_scan_48','/ssd2/ming/2024COVID/valid_pure_crop_challenge/positive/ct_scan_60']
    train_df = pd.read_csv('/ssd2/ming/2024COVID/filter_slice_train_df_challenge.csv')
    train_df = train_df[~train_df['path'].isin(not_allow)]
    valid_df = pd.read_csv('/ssd2/ming/2024COVID/filter_slice_valid_df_challenge.csv') 
    valid_df = valid_df[~valid_df['path'].isin(not_allow)]  
    #test_df = pd.read_csv('')
    
    #train_df['path'] = train_df['path'].str.replace('train_pure_crop_challenge', 'train')

    #valid_df['path'] = valid_df['path'].str.replace('valid_pure_crop_challenge', 'valid')

    print(len(train_df),len(valid_df))
    fold_df = pd.read_csv('/ssd2/ming/2024COVID/chih_full_replaced_df.csv')
    fold_df = fold_df[~fold_df['path'].isin(not_allow)]  
    
    #fold_df['path'] = fold_df['path'].str.replace('train_pure_crop_challenge', 'train')
    #fold_df['path'] = fold_df['path'].str.replace('valid_pure_crop_challenge', 'valid')
    
    total_dic = {**train_dic, **valid_dlc}
    total_df = pd.concat([train_df, valid_df])  
    # fold loader
    job=60
    lst = [1,2,3,4,5]
    for i in range(len(lst)):
        print("fold {}的順序：{}".format(i+1, lst[i:]+lst[:i]),'kds')
        train_lst = (lst[i:]+lst[:i])[0:4] 
        train_fold = fold_df[fold_df.fold.isin(train_lst)]
        #train_fold = train_fold.sample(frac=0.03, random_state=42)

        valid_lst = (lst[i:]+lst[:i])[-1]
        valid_fold = fold_df[fold_df.fold.isin([valid_lst])]
        print(valid_fold.values.tolist()[0])
        print("Train: {} || Valid: {}".format(train_fold.shape, valid_fold.shape))
        
        train_loader, valid_loader = prepare_loaders(CONFIG, train_df, train_dic, valid_df, valid_dlc, data_transforms)
        bin_save_path = "/ssd2/ming/2024COVID/model"
        job_name = f"job_{job}_effb7_size{CONFIG['img_size']}_challenge[DataParallel]-fold{i+1}"+".bin"
        
        print("="*10, "loading *model*", "="*10)
        #model=eca_nfnet_l0(n_classes=2,pretrained=True)
        model=Net()
        model = nn.DataParallel(model, device_ids=[0])
        model = model.to(CONFIG['device'])
        scaler = amp.GradScaler()
        train_loader, valid_loader = prepare_loaders(CONFIG, train_fold, total_dic, valid_fold, total_dic, data_transforms)
        
        print("="*10, "*model* setting", "="*10)
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], 
                        weight_decay=CONFIG['weight_decay'])
        
        print("="*10, "Start Train", "="*10)
        

        model, history, best_epoch_f1= run_training(model, optimizer,None,
                                device=CONFIG['device'],
                                num_epochs=CONFIG['epochs'])
      
        


        
        # Modify the keys in the checkpoint dictionary
        #new_checkpoint = {k.replace('model', 'module.model'): v for k, v in checkpoint.items()}
        new_checkpoint = {'module.'+k: v for k, v in checkpoint.items()}

        # Load the modified state dictionary into the model
        model.load_state_dict(new_checkpoint)

        # Move the model to CUDA if available
        model.to('cuda')

        #train_loader, valid_loader = prepare_loaders(CONFIG, train_df, train_dic, valid_df, valid_dlc, data_transforms)
        for j in range(5):
            valid_lst = (lst[j:]+lst[:j])[-1]
            valid_fold = fold_df[fold_df.fold.isin([valid_lst])]
            #valid_fold = valid_fold.sample(frac=0.03, random_state=42)
            #print(valid_fold.values.tolist()[0])
            print("Valid: {},{}-th fold || Training dataset {}-th fold".format(valid_fold.shape, j+1, i+1))
        
            #_, valid_loader = prepare_loaders(CONFIG, train_df, train_dic, valid_df, valid_dlc, data_transforms)
            
            _, valid_loader = prepare_loaders(CONFIG, train_df, train_dic, valid_fold, valid_dlc, data_transforms)
            true_y, pred_y = pred_one(model, valid_loader, device=CONFIG['device'])
            #print(true_y.shape, pred_y.shape)
            #print(true_y, pred_y)
            print(f1_score(np.array(true_y),np.round(pred_y),average='macro'))  