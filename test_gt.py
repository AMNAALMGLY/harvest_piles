import glob

import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from torchvision import transforms
import torchvision
from torchmetrics import ConfusionMatrix
import torchmetrics
import os
from torchmetrics.classification import StatScores
import tifffile as tiff
from configs import args
# from osgeo import gdal
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
        self.datadir = datadir

        self.data = pd.read_csv(csv_dir)
       
        self.label_name = label_name
        self.patch_size = patch_size
        self.augment = augment
        self.normalize = normalize
        self.clipn = clipn
        self.rescale = rescale
        if self.normalize:
            self.means, self.stds=self.norm(datadir,csv_dir)
            print(self.means,self.stds)
              # self.means,self.stds=np.array([0.4355515,  0.32816476, 0.22282955]), np.array([0.13697046, 0.09374066, 0.07623406])
              
             # self.means, self.stds = np.array( [103.94648461 , 92.20671153 , 81.00511347] ), np.array(
             #     [26.13480798, 22.04760537, 21.29621976])
            # self.means, self.stds=np.array( [0.40763366 ,0.3615943 , 0.31766695]),np.array( [0.1024891 , 0.08646118, 0.08351463])
           

       
        
        self.crop=crop
        self.resize=resize
        self.resize_size=resize_size
        self.extract_time=time
        # self.metadata
        #time stamp:
        if time:
            #extract
            self.data['year']=self.data['filename'].apply(lambda x:x.split('_')[0][:4])
            self.data['month']=self.data['filename'].apply(lambda x:x.split('_')[0][4:6])
            #encode:
            self.data['year']=self.data['year'].apply(encode_year)
            self.data['month']=self.data['month'].apply(encode_month)
            

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):
        # read the images
        # the metadata
        # the labels
        
        
            labels = self.data.loc[idx, self.label_name]
            if not isinstance(labels, bool):
                            labels = labels.astype(np.bool_)
            if not pd.isna(self.data.loc[idx, 'filename']):
                    
                    if ',' in self.data.loc[idx, 'filename']:
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


                   
      

        
                        return image, labels
       

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

    def load_geotiff( self,file):
       
        img = tiff.imread(file).astype(np.float32)[:,:,:-1]
        img = img.transpose( 2, 0, 1)
        img = img / 255
        image=transforms.ToPILImage()(torch.tensor(img,))


        return image
    
    def collate_fn(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

        # finidng the mean and standard deviation of all the images
    

    def norm(self,imgs_root_dir,csv_dir):
         
        data = pd.read_csv(csv_dir)
                            
        img_list = []
        files_list=[]
        i=0
        if ',' in data.loc[0,'filename']:
            for idx,raw in data.iterrows():
                
                            files=data.loc[idx, 'filename'].split(',')
                            files_list.extend(files)
                            
            files=files_list
        
        for file in files:
            img_filename = os.path.join(self.datadir, str(file))
            img = self.load_geotiff(img_filename)
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
        
        outputs = model(batch)
        outputs = outputs.squeeze(dim=-1)
        preds = torch.sigmoid(outputs, )

        preds = (preds >= 0.5).type_as(batch)

        return preds

# import glob

# import pandas as pd
# import torch
# from torch.utils.data import Dataset
# import pickle
# import numpy as np
# from torchvision import transforms

# import os
# from configs import args
# # from osgeo import gdal
# import tifffile as tiff

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
#                  time=args.use_time,rescale=args.rescale):
#         '''
#         Args

#         datadir:str of images dir
#         csv_dir:str of csv file directory
#         transform:boolean
#         patch_size:int
#         TODO, provide path for validation and testing rather than splitting manually
#         '''
#         self.datadir = datadir

#         self.data = pd.read_csv(csv_dir)
       
#         self.label_name = label_name
#         self.patch_size = patch_size
#         self.augment = augment
#         self.normalize = normalize
#         self.clipn = clipn
#         self.rescale = rescale
#         if self.normalize:
#              self.means, self.stds = self.norm(datadir,csv_dir)
#                 #self.norm(datadir,csv_dir)
#              print(self.means, self.stds)
# #              if 'planet' in datadir:
# #                      print('in planet norm')
# #                      self.filename='planet_folder'
# #                      self.means,self.stds=np.array([0.4355515,  0.32816476, 0.22282955]), np.array([0.13697046, 0.09374066, 0.07623406])
                     
# #              else:
# #                     self.filename='skysat_folder'
# #                     self.means, self.stds=np.array( [0.40763366 ,0.3615943 , 0.31766695]),np.array( [0.1024891 , 0.08646118, 0.08351463])
                    
          
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

#         return len(self.data)

#     def __getitem__(self, idx):
#         # read the images
#         # the metadata
#         # the labels
#            try:
#             img_filename = os.path.join(self.datadir,str(self.data.loc[idx, 'filename']))
            
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

#             if self.data is not None:
#                 labels = self.data.loc[idx, self.label_name]
#             else:
#                 labels=None
#             # if not isinstance(labels, bool):
#             labels = labels.astype(np.float32)


        
#             return image, labels
#            except: 
            
#             return None
       

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

# #     def load_geotiff(self, file):

# #         ds = gdal.Open(file)
# #         if not ds:
# #             return None
# #         r, g, b = np.array(ds.GetRasterBand(1).ReadAsArray()), np.array(ds.GetRasterBand(2).ReadAsArray()), np.array(
# #             ds.GetRasterBand(3).ReadAsArray())

# #         channels = [r, g, b]
# #         image = np.stack(channels, axis=0)
# #         image=image.transpose(1,2,0)
# #         image=transforms.ToPILImage()(image)
# #         return image
#     def load_geotiff(self,file):
       
#         img = tiff.imread(file).astype(np.float32)[:,:,:-1]
#         img = np.reshape(img, (img.shape[2], img.shape[0], img.shape[1]))
#         img = img / 255
#         image=transforms.ToPILImage()(torch.tensor(img,))
       
#         return image
    
#     def collate_fn(self, batch):
#         batch = list(filter(lambda x: x is not None, batch))
#         return torch.utils.data.dataloader.default_collate(batch)

#         # finidng the mean and standard deviation of all the images
    

#     def norm(self,imgs_root_dir,csv_dir):
         
#         data = pd.read_csv(csv_dir)
       
#         files= (data['filename'].apply(lambda x : os.path.join(imgs_root_dir,str(x)))).to_list()
        
#         #str(imgs_root_dir) + data[self.filename].astype(str)

    
     
#         # img_filename = os.path.join(imgs_root_dir, str(self.data.loc[idx, 'filename']))
#         # files = glob.glob(os.path.join(imgs_root_dir, '*.tif'))
#         img_list = []
#         i=0
#         for file in files:
#             img = self.load_geotiff(str(file))
#             if img is not None:
    
#                 img =transforms.CenterCrop((self.patch_size,self.patch_size))(img)
#                 img=transforms.ToTensor()(img).numpy()
               
              
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

def main():
    model=torchvision.models.resnet50(pretrained=True)
    model.fc=torch.nn.Linear(model.fc.in_features,1)
    checkpoint = torch.load('/atlas/u/amna/harvest_piles/outputs/paper/skysat/resnet50_paper_b64_conv01_lr0003_crop512/best_Epoch16.ckpt')
        #'/atlas/u/amna/harvest_piles/outputs/new_split/planet/planet_resnet_step_cluster2km_fullbalanced_5kfinal_b32_conv1.0_lr0001_crop53/best.ckpt')
                        
    model.load_state_dict(checkpoint)
    test_params = dict(datadir='/atlas/u/amna/harvest_piles/amhara_pile_data/',
                           csv_dir='/atlas/u/amna/harvest_piles/amhara_pile_data.csv', augment=False, normalize=True, clipn=True,label_name='has_piles',
                       #'Crop (Crop stand)',
                            patch_size=53)
    test = HarvestPatches(**test_params)

    test_loader = torch.utils.data.DataLoader(test, batch_size=64,
                                                  shuffle=False)
    model.eval()
    model.to('cuda')
    preds=[]  
    batch_preds=[]
    targets=[]
    with torch.no_grad():


        for batch in test_loader:
                t=torch.tensor(batch[1],device=args.gpus)
                targets.append(t)
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
            
        

    preds=torch.cat(preds).cpu()
    target=torch.cat(targets,axis=0).cpu()
    torch.save(preds,'tigray_pile_data.pt')
    confmat = ConfusionMatrix(num_classes=2,task='binary')
    print(confmat(preds, target.int()))
    print('acc ',torchmetrics.functional.accuracy(preds,target,task='binary'))
    print('f1 ',torchmetrics.functional.f1_score(preds,target,task='binary'))
    print('AUROC ',torchmetrics.AUROC(num_classes=2,task='binary')(preds,target))
    print('recall ',torchmetrics.Recall(num_classes=1,task='binary')(preds,target))
    print('precision ',  torchmetrics.Precision(num_classes=1,task='binary')(preds,target))
    _,fp,tn,fn,_=StatScores(num_classes=1,task='binary')(preds,target)
    tnr=tn/(tn+fn)
    print('tnr ',tnr)
if __name__ == "__main__":
    main()
