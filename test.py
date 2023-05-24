import glob

import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from torchvision import transforms
import torchvision
from configs import args

import os
from osgeo import gdal

from sklearn.model_selection import train_test_split
# class HarvestPatches(Dataset):
#     """
#     Patches of harvest piles each

#     Input (x):
#         32 x 32 x 3  satellite image.
#     Label (y):
#         y is binary label whether this patch is pile or not
#     Metadata:
#         each image is annotated with a location coordinate corners (x1,x2,y1,y2).
    
#     """

#     def __init__(self, datadir, csv_dir, augment, normalize, clipn, patch_size, label_name,resize=True,crop=True,resize_size=224,
#                  time=False,rescale=False):
#         '''
#         Args

#         datadir:str of images dir
#         csv_dir:str of csv file directory
#         transform:boolean
#         patch_size:int
#         TODO, provide path for validation and testing rather than splitting manually
#         '''
#         self.datadir = datadir
#         self.files=glob.glob(os.path.join(self.datadir,'*'))[90000:100000]
#         self.data = pd.read_csv(csv_dir) if csv_dir else None
       
#         self.label_name = label_name
#         self.patch_size = patch_size
#         self.augment = augment
#         self.normalize = normalize
#         self.clipn = clipn
#         self.rescale = rescale
#         if self.normalize:
#              # self.means, self.stds = self.norm(datadir,csv_dir)
#              # print(self.means, self.stds)
#             self.means,self.stds=np.array([0.4355515,  0.32816476, 0.22282955]), np.array([0.13697046, 0.09374066, 0.07623406])
#              # self.means, self.stds = np.array([110.55637167  ,83.66213759,  56.63501254]),np.array([35.87896582, 24.70206238, 20.74427246])
#              # self.means, self.stds = np.array( [103.94648461 , 92.20671153 , 81.00511347] ), np.array(
#              #     [26.13480798, 22.04760537, 21.29621976])
       
        
#         self.crop=crop
#         self.resize=resize
#         self.resize_size=resize_size
#         self.extract_time=time
#         # self.metadata
#         #time stamp:
#         if time:
#             #extract
#             self.data['year']=self.data['filename'].apply(lambda x:x.split('_')[0][:4])
#             self.data['month']=self.data['filename'].apply(lambda x:x.split('_')[0][4:6])
#             #encode:
#             self.data['year']=self.data['year'].apply(encode_year)
#             self.data['month']=self.data['month'].apply(encode_month)
            

#     def __len__(self):
#         if self.data:
#             return len(self.data)
#         else:
            
#             return len(self.files)

#     def __getitem__(self, idx):
#         # read the images
#         # the metadata
#         # the labels
#         if self.data is not None:
#             img_filename = os.path.join(self.datadir, str(self.data.loc[idx, 'filename']))
#         else:
            
#             img_filename=self.files[idx]
#         # image = self.load_geotiff(img_filename)

#         try:
            
#             image = self.load_geotiff(img_filename)

       
          
#             #preprocess the image to match the desired image size required
#             if self.crop:
#                         image=transforms.CenterCrop((self.patch_size,self.patch_size))(image)

#             if self.resize:
#                         image = transforms.Resize((self.resize_size,self.resize_size))(image)
#                         image=transforms.ToTensor()(image).numpy()
#             # preprocessing
#             if self.clipn:
#                 image = np.clip(image, a_min=0., a_max=1e20)
#             if self.rescale:
#                 image = (image + 1.0) / 2.0
#             elif self.normalize:
#                 # per channel normalization
#                 self.means = self.means.reshape(-1, 1, 1)
#                 self.stds = self.stds.reshape(-1, 1, 1)
#                 image = (image - self.means) / self.stds

#             if self.augment:
#                 image = torch.tensor(image)
#                 image = self.transform()(image)
#                 image = image.numpy()
                
#             labels=0
#             if  self.data is not None:
#                 labels = self.data.loc[idx, self.label_name]
#                 if not isinstance(labels, bool):
#                       labels = labels.astype(np.bool_)
            
       
#         except:
#             return None

        
#         return image, labels

#     def transform(self):
#         # pass
#         if self.augment:
#             aug = transforms.Compose([
#                 # transforms.ToPILImage(),
#                 # transforms.ColorJitter(brightness=(0.05, 0.95), contrast=(0.05, 0.95)),
#                 # transforms.RandomRotation(15),
#                 transforms.RandomHorizontalFlip(),

#                 # transforms.ToTensor(),
#             ])
#         return aug
#     def collate_fn(self, batch):
#         batch = list(filter(lambda x: x is not None, batch))
#         return torch.utils.data.dataloader.default_collate(batch)
#     def load_geotiff(self, file):

#         ds = gdal.Open(file)
#         if not ds:
#             return None
#         r, g, b = np.array(ds.GetRasterBand(1).ReadAsArray()), np.array(ds.GetRasterBand(2).ReadAsArray()), np.array(
#             ds.GetRasterBand(3).ReadAsArray())

#         channels = [r, g, b]
#         image = np.stack(channels, axis=0)
#         image=image.transpose(1,2,0)
#         image=transforms.ToPILImage()(image)
#         return image
#         # finidng the mean and standard deviation of all the images

#     def norm(self,imgs_root_dir,csv_dir):
         
#         data = pd.read_csv(csv_dir)
       
#         data['filename']= str(imgs_root_dir) + data['filename'].astype(str)
#         files= data['filename'].tolist()
#         print(len(files))
     
#         # img_filename = os.path.join(imgs_root_dir, str(self.data.loc[idx, 'filename']))
#         # files = glob.glob(os.path.join(imgs_root_dir, '*.tif'))
#         img_list = []
#         i=0
#         for file in files:
#             img = self.load_geotiff(file)
#             if img is not None:
#                 print(img.shape)
#                 img =transforms.CenterCrop((self.patch_size,self.patch_size))(torch.tensor(img)).numpy()
#                 print('after',img.shape)
              
#             #print(img.shape)
#             if img is  None  or img.shape !=(3,self.patch_size,self.patch_size):
#                 i+=1
#                 # print(img.shape)
#                 continue
#             img_list.append(img)
#         print('i', i)
#         imgs = np.stack(img_list, axis=0)
#         means = np.mean(imgs, axis=(0, 2, 3))
#         stds = np.std(imgs, axis=(0, 2, 3))
#         return means, stds

# min_year=2007
# def encode_month(data):
#     return np.sin(2 * np.pi * data/12)
# def encode_year(data):
#     data=data.astype(np.int_)-min_year
#     return data
# def make_balanced_weights(dataset):
#     '''
#     dataset: pd.Dataframe of imgs, labels('piles')
#     return
#     Weights: tensor of shape len(dataset)
#     '''
#     pos = 0
#     neg = 0
#     #preprocess the labels to boolen type
#     dataset['has_piles']=dataset['has_piles'].astype(np.bool_)
#     for idx, row in enumerate(dataset.iterrows()):
#         # print('item',dataset.loc[idx, args.label_name])
#         if dataset.loc[idx, 'has_piles']:
#             # print('in positives',pos)
#             pos = pos + 1
#         else:
#             # print('in negatives',neg)
#             neg = neg + 1
#     print(pos, neg)
#     N = len(dataset)
#     weights = {'pos': N / pos, 'neg': N / neg}
#     weight = [0] * len(dataset)
#     for idx, row in enumerate(dataset.iterrows()):
#         weight[idx] = weights['pos'] if dataset.loc[idx, args.label_name] else weights['neg']
#     return torch.tensor(weight, dtype=torch.float32)


# def generate_random_splits(dataset, val_size, test_size):
#     train_size = int((1 - val_size - test_size) * len(dataset))
#     val_size = int(val_size * len(dataset))
#     test_size = len(dataset) - (train_size + val_size)
#     train, val_test = torch.utils.data.random_split(dataset, [train_size, (test_size + val_size)])
#     val, test = torch.utils.data.random_split(val_test, [val_size, test_size])
#     return train, val, test


# def generate_stratified_splits(X, y, val_size, test_size, stratify=True):
#     # todo: does test set need to have the same distribution as train? No => need a fix
#     if stratify:
#         X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=(val_size + test_size), stratify=y)
#         test_size = int(len(X) * test_size)
#         X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=test_size, stratify=y_val)
#     else:
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
#         X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, stratify=y_train)

#     return X_train, X_val, X_test, y_train, y_val, y_test

class HarvestPatches(Dataset):
    """
    Patches of harvest piles each

    Input (x):
        32 x 32 x 3  satellite image.
    Label (y):
        y is binary label whether this patch is pile or not
    Metadata:
        each image is annotated with a location coordinate corners (x1,x2,y1,y2).
    
    """

    def __init__(self, datadir, csv_dir, augment, normalize, clipn, patch_size, label_name,resize=True,crop=True,resize_size=224,
                 time=False,rescale=False):
        '''
        Args

        datadir:str of images dir
        csv_dir:str of csv file directory
        transform:boolean
        patch_size:int
        TODO, provide path for validation and testing rather than splitting manually
        '''
        self.files=None
        
        self.datadir = datadir
        # with open('/atlas/u/amna/harvest_piles/harvest_piles/highlandsS_unq_stacked.pkl','rb') as f:
        #     self.files=pickle.load(f)[:,-1]
        
        if csv_dir:
            self.data = pd.read_csv(csv_dir)
       
        self.label_name = label_name
        self.patch_size = patch_size
        self.augment = augment
        self.normalize = normalize
        self.clipn = clipn
        self.rescale = rescale
        if self.normalize:
           self.means, self.stds=np.array( [0.40763366 ,0.3615943 , 0.31766695]),np.array( [0.1024891 , 0.08646118, 0.08351463])
       
             # self.means, self.stds = np.array([110.55637167  ,83.66213759,  56.63501254]),np.array([35.87896582, 24.70206238, 20.74427246])
              # self.means,self.stds=np.array([0.4002362 , 0.3010103,  0.20120342]) ,np.array([0.14284575, 0.10003384, 0.07724908])
             # self.means, self.stds = np.array( [103.94648461 , 92.20671153 , 81.00511347] ), np.array(
             #     [26.13480798, 22.04760537, 21.29621976])
            # self.means, self.stds=np.array( [0.40763366 ,0.3615943 , 0.31766695]),np.array( [0.1024891 , 0.08646118, 0.08351463])
           

       
        
        self.crop=crop
        self.resize=resize
        self.resize_size=resize_size
        self.extract_time=time
        # self.metadata
        #time stamp:
        

    def __len__(self):
        if self.files is not None:
            return len(self.files)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        # read the images
        # the metadata
        # the labels
        
        # try:
#             labels = self.data.loc[idx, self.label_name]
             
#             labels = labels.astype(np.float32)
            # if not pd.isna(self.data.loc[idx, 'filename']):
            #         if ',' in self.data.loc[idx, 'filename']:
            labels=0 #labels are None
            if self.files is not None or ','  in self.data.loc[idx, 'filename']:
                        if self.files:
                            files=self.files[idx].split(',')
                        elif ','  in self.data.loc[idx, 'filename']:
                          
                            files=self.data.loc[idx, 'filename'].split(',')
                        img_list=[]
                        
                        for file in files:
                            img_filename = os.path.join(self.datadir, str(file))
                            image = self.load_geotiff(img_filename)
                            # if image is  None:
                            #     image=np.empty((3,self.patch_size,self.patch_size))
                            #preprocess the image to match the desired image size required
                            if self.crop:
                                image=transforms.CenterCrop((self.patch_size,self.patch_size))(image)

                            if self.resize:
                                image = transforms.Resize((self.resize_size,self.resize_size))(image)
                                image=transforms.ToTensor()(image).numpy()
                            # preprocessing
                            if self.clipn:
                                image = np.clip(image, a_min=0., a_max=1e20)

                            if self.normalize:
                                # per channel normalization
                                self.means = self.means.reshape(-1, 1, 1)
                                self.stds = self.stds.reshape(-1, 1, 1)
                                image = (image - self.means) / self.stds
                            if self.augment:
                                image = torch.tensor(image)
                                image = self.transform()(image)
                                image = image.numpy()
                            img_list.append(image)
                            
                        return img_list,labels
                       
            else:
                        img_filename = os.path.join(self.datadir, str(self.data.loc[idx, 'filename']))
                        if '(1).tif' in img_filename:
                            return None 
                        image = self.load_geotiff(img_filename)

                        #preprocess the image to match the desired image size required
                        if self.crop:
                            image=transforms.CenterCrop((self.patch_size,self.patch_size))(image)

                        if self.resize:
                            image = transforms.Resize((self.resize_size,self.resize_size))(image)
                            image=transforms.ToTensor()(image).numpy()
                        # preprocessing
                        if self.clipn:
                            image = np.clip(image, a_min=0., a_max=1e20)
                        if self.rescale:
                            image = (image + 1.0) / 2.0
                        elif self.normalize:
                            # per channel normalization
                            self.means = self.means.reshape(-1, 1, 1)
                            self.stds = self.stds.reshape(-1, 1, 1)

                            image = (image - self.means) / self.stds
                            # image=torchvision.transforms.Normalize(self.means,self.stds)(image)

                        if self.augment:
                            image = torch.tensor(image)
                            image = self.transform()(image)
                            image = image.numpy()
                            
                        return image ,labels


                   
        # except:
        #     return None

        
      
       

    def transform(self):
        # pass
        if self.augment:
            aug = transforms.Compose([
                # transforms.ToPILImage(),
                # transforms.ColorJitter(brightness=(0.05, 0.95), contrast=(0.05, 0.95)),
                # transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),

                # transforms.ToTensor(),
            ])
        return aug

    def load_geotiff(self, file):
       
#         image = cv2.imread(file)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image=transforms.ToPILImage()(image)
        # image=transforms.ToTensor()(image)
        # print(image.permute(2,0,1))
        
        ds = gdal.Open(file)
        if not ds:
            return None
        r, g, b = np.array(ds.GetRasterBand(1).ReadAsArray()), np.array(ds.GetRasterBand(2).ReadAsArray()), np.array(
            ds.GetRasterBand(3).ReadAsArray())

        channels = [r, g, b]
        image = np.stack(channels, axis=0)
        image=image.transpose(1,2,0)
        image=transforms.ToPILImage()(image)
          

        return image
    
    def collate_fn(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

        # finidng the mean and standard deviation of all the images
    

    def norm(self,imgs_root_dir,csv_dir):
         
        data = pd.read_csv(csv_dir)
       
        data['filename']= str(imgs_root_dir) + data['filename'].astype(str)
        files= data['filename'].tolist()
        print(len(files))
     
        # img_filename = os.path.join(imgs_root_dir, str(self.data.loc[idx, 'filename']))
        # files = glob.glob(os.path.join(imgs_root_dir, '*.tif'))
        img_list = []
        files_list=[]
        i=0
        if ',' in data.loc[0,'filename']:
            for idx,raw in data.iterrows():
                
                            files=data.loc[idx, 'filename'].split(',')
                            files_list.extend(files)
                            # for file in files:
                            #     #img_filename = os.path.join(self.datadir, str(file))
                          #     image = self.load_geotiff(file)
            files=files_list
        
        for file in files:
            img = self.load_geotiff(str(file))
            if img is not None:
    
                img =transforms.CenterCrop((self.patch_size,self.patch_size))(img)
                img=transforms.ToTensor()(img).numpy()
               
              
            #print(img.shape)
            if img is  None  or img.shape !=(3,self.patch_size,self.patch_size):
                i+=1
                # print(img.shape)
                continue
            img_list.append(img)
        imgs = np.stack(img_list, axis=0)
        means = np.mean(imgs, axis=(0, 2, 3))
        stds = np.std(imgs, axis=(0, 2, 3))
        return means, stds
def predict(batch, model):
                            
#         model(batch)
#         torch.quantization.convert(model, inplace=True)

                  
               
        outputs = model(batch)
        outputs = outputs.squeeze(dim=-1)
        preds = torch.sigmoid(outputs, )

        preds = (preds >= 0.5).type_as(batch)

        return preds

def main():
    model=torchvision.models.resnet50(pretrained=True)
    model.fc=torch.nn.Linear(model.fc.in_features,1)
    checkpoint = torch.load('/atlas/u/amna/harvest_piles/outputs/new_split/planet/planet_resnet_step_cluster2km_fullbalanced_5kfinal_b32_conv1.0_lr0001_crop53/best.ckpt')
       
        # '/atlas/u/amna/harvest_piles/outputs/new_split/skysat/fullbalanced_5kfinal_cluster300_finetuned_b32_conv1_lr0003_crop512/best.ckpt'
        #'/atlas/u/amna/harvest_piles/outputs/new_split/planet/planet_resnet_step_cluster2km_fullbalanced_5kfinal_b32_conv1.0_lr0001_crop53/best.ckpt') best planet model I guess!
        #'/atlas/u/amna/harvest_piles/outputs/moco_pretrained_expLR_b32_conv01_lr0001_crop512/best.ckpt')
        #'harvest_piles/outputs/planet_mocopretrained_expLR_b32_conv01_lr0001_crop55/best.ckpt')
        #'harvest_piles/outputs/moco_pretrained_expLR_b32_conv01_lr0001_crop512/best.ckpt')
        #'harvest_piles/outputs/imbalanced_data_sigmoid_ssl_b64_conv01_lr0001_crop512/best.ckpt')
    #torch.load('harvest_piles/outputs/imbalanced_data_sigmoid_b64_conv01_lr0001_crop512/best.ckpt')
    model.load_state_dict(checkpoint)
    # model = torch.nn.DataParallel(model)
    order='amhara_gt'
    #'south_highland'
    test_params = dict(datadir='/atlas/u/amna/harvest_piles/amhara_pile_data',
                       #'/atlas/u/amna/harvest_piles/tigray_south_256',
                          #'/atlas/u/amna/harvest_piles/black_north_v2.csv
                       csv_dir='/atlas/u/amna/harvest_piles/amhara_pile_data.csv', augment=False, normalize=True, 
                       clipn=False,label_name='has_piles',
                        patch_size=53)
 
    test = HarvestPatches(**test_params)

    test_loader = torch.utils.data.DataLoader(test, batch_size=53,
                                                  shuffle=False,num_workers=4,pin_memory=True,)
                                              #collate_fn=test.collate_fn)

    model.eval()
    # model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # torch.quantization.prepare(model, inplace=True)
    model.to(args.gpus)
    
    with torch.no_grad():

        preds=[]
        for batch in test_loader:
            if batch:
                batch_preds=[]
                
                if  isinstance(batch[0],list):
                    img_list=batch[0]
                    img_list=torch.stack(img_list,)
                    img_list=torch.tensor(img_list,device=args.gpus)
                    p=torch.empty((img_list.shape[0],img_list.shape[1]),device=args.gpus)
                    d=img_list.shape[0]
                    #make it 64x4 tensor 
                    # print(b.min(),b.max())
                    #for b in img_list:
                    for i in range(d):
                        # run calibration step

                        p[i,:]=predict(img_list[i,...].type_as(model.fc.weight),model)
                    p=torch.max(p,dim=0)[0]

                    preds.append(p)
                    print('finished preds in a batch! ',p)
                else:
                      img=torch.tensor(batch[0],device=args.gpus)
                      p=predict(img.type_as(model.fc.weight),model)
                      preds.append(p)
                      print('finished preds in a batch! ',p)
            
        

                   
          
       
        



    preds=torch.cat(preds,axis=0)

   
    torch.save(preds,f'preds_{order}_planet.pt')
    
if __name__ == "__main__":
    main()
