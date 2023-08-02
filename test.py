import glob

import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from torchvision import transforms

import os
from configs import args
# from osgeo import gdal
import tifffile as tiff
from src.utils import init_model, get_model, get_full_experiment_name

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
                 time=args.use_time,rescale=args.rescale):
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
             self.means, self.stds = np.array([0.42506683 ,0.38138887, 0.33894157]), np.array([0.12454753, 0.11270324 ,0.11512998])
                #self.norm(datadir,csv_dir)
             # print(self.means, self.stds)
#              if 'planet' in datadir:
#                      print('in planet norm')
#                      self.filename='planet_folder'
#                      self.means,self.stds=np.array([0.4355515,  0.32816476, 0.22282955]), np.array([0.13697046, 0.09374066, 0.07623406])
                     
#              else:
#                     self.filename='skysat_folder'
#                     self.means, self.stds=np.array( [0.40763366 ,0.3615943 , 0.31766695]),np.array( [0.1024891 , 0.08646118, 0.08351463])
                    
          
             # self.means, self.stds = np.array([110.55637167  ,83.66213759,  56.63501254]),np.array([35.87896582, 24.70206238, 20.74427246])

             # self.means, self.stds = np.array( [103.94648461 , 92.20671153 , 81.00511347] ), np.array(
             #     [26.13480798, 22.04760537, 21.29621976])
       
        
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
           try:
            img_filename = os.path.join(self.datadir,str(self.data.loc[idx, 'filename']))
            
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

            if self.augment:
                image = torch.tensor(image)
                image = self.transform()(image)
                image = image.numpy()

            if self.data is not None:
                labels = self.data.loc[idx, self.label_name]
            else:
                labels=None
            # if not isinstance(labels, bool):
            labels = labels.astype(np.float32)


        
            return image, labels
           except: 
            
            return None
       

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
    def load_geotiff(self,file):
        print(file)
        img = tiff.imread(file).astype(np.float32)[:,:,:-1]
        img = np.reshape(img, (img.shape[2], img.shape[0], img.shape[1]))
        img = img / 255
        image=transforms.ToPILImage()(torch.tensor(img,))
       
        return image
    
    def collate_fn(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

        # finidng the mean and standard deviation of all the images
    

    def norm(self,imgs_root_dir,csv_dir):
         
        data = pd.read_csv(csv_dir)
       
        files= (data['filename'].apply(lambda x : os.path.join(imgs_root_dir,str(x)))).to_list()
        
        #str(imgs_root_dir) + data[self.filename].astype(str)

    
     
        # img_filename = os.path.join(imgs_root_dir, str(self.data.loc[idx, 'filename']))
        # files = glob.glob(os.path.join(imgs_root_dir, '*.tif'))
        img_list = []
        i=0
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
        print('i', i)
        imgs = np.stack(img_list, axis=0)
        means = np.mean(imgs, axis=(0, 2, 3))
        stds = np.std(imgs, axis=(0, 2, 3))
        return means, stds
    
min_year=2007
def encode_month(data):
    return np.sin(2 * np.pi * data/12)
def encode_year(data):
    data=data.astype(np.int_)-min_year
    return data
def make_balanced_weights(dataset):
    '''
    dataset: pd.Dataframe of imgs, labels('piles')
    return
    Weights: tensor of shape len(dataset)
    '''
    pos = 0
    neg = 0
    #preprocess the labels to boolen type
    dataset[args.label_name]=dataset[args.label_name].astype(np.bool_)
    for idx, row in enumerate(dataset.iterrows()):
        # print('item',dataset.loc[idx, args.label_name])
        if dataset.loc[idx, args.label_name]:
            # print('in positives',pos)
            pos = pos + 1
        else:
            # print('in negatives',neg)
            neg = neg + 1
    print(pos, neg)
    N = len(dataset)
    weights = {'pos': N / pos, 'neg': N / neg}
    weight = [0] * len(dataset)
    for idx, row in enumerate(dataset.iterrows()):
        weight[idx] = weights['pos'] if dataset.loc[idx, args.label_name] else weights['neg']
    return torch.tensor(weight, dtype=torch.float32)


def generate_random_splits(dataset, val_size, test_size):
    train_size = int((1 - val_size - test_size) * len(dataset))
    val_size = int(val_size * len(dataset))
    test_size = len(dataset) - (train_size + val_size)
    train, val_test = torch.utils.data.random_split(dataset, [train_size, (test_size + val_size)])
    val, test = torch.utils.data.random_split(val_test, [val_size, test_size])
    return train, val, test

def predict(batch, model):
                            
#         model(batch)
#         torch.quantization.convert(model, inplace=True)

                  
               
        outputs = model(batch)
        outputs = outputs.squeeze(dim=-1)
        preds = torch.sigmoid(outputs, )

        preds = (preds >= 0.5).type_as(batch)

        return preds
def main():
    model=get_model("satlas",3,False,"")
    test_params = dict(datadir="/atlas2/u/amna/harvest_piles/merged/unlabelled",
                           csv_dir='/atlas2/u/amna/harvest_piles/unlabelled_merged.csv', augment=False, normalize=args.normalize, clipn=args.clipn,
                           label_name=args.label_name,
                           patch_size=args.image_size)
    test = HarvestPatches(**test_params)

    test_loader = torch.utils.data.DataLoader(test, batch_size=64,
                                                  shuffle=False,num_workers=4,pin_memory=True,)
                                              #collate_fn=test.collate_fn)
        
    model.eval()
    order="20k"
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

   
    torch.save(preds,f'preds_{order}_skysat.pt')
    
if __name__ == "__main__":
    main()

# def main():
#     model=torchvision.models.resnet50(pretrained=True)
#     model.fc=torch.nn.Linear(model.fc.in_features,1)
#     checkpoint = torch.load('/atlas/u/amna/harvest_piles/outputs/new_split/planet/planet_resnet_step_cluster2km_fullbalanced_5kfinal_b32_conv1.0_lr0001_crop53/best.ckpt')
       
        
#     model.load_state_dict(checkpoint)
#     # model = torch.nn.DataParallel(model)
#     order='amhara_gt'
#     #'south_highland'
#     test_params = dict(datadir='/atlas/u/amna/harvest_piles/amhara_pile_data',
#                        #'/atlas/u/amna/harvest_piles/tigray_south_256',
#                           #'/atlas/u/amna/harvest_piles/black_north_v2.csv
#                        csv_dir='/atlas/u/amna/harvest_piles/amhara_pile_data.csv', augment=False, normalize=True, 
#                        clipn=False,label_name='has_piles',
#                         patch_size=53)
 
#     test = HarvestPatches(**test_params)

#     test_loader = torch.utils.data.DataLoader(test, batch_size=53,
#                                                   shuffle=False,num_workers=4,pin_memory=True,)
#                                               #collate_fn=test.collate_fn)

#     model.eval()
#     # model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
#     # torch.quantization.prepare(model, inplace=True)
#     model.to(args.gpus)
    
#     with torch.no_grad():

#         preds=[]
#         for batch in test_loader:
#             if batch:
#                 batch_preds=[]
                
#                 if  isinstance(batch[0],list):
#                     img_list=batch[0]
#                     img_list=torch.stack(img_list,)
#                     img_list=torch.tensor(img_list,device=args.gpus)
#                     p=torch.empty((img_list.shape[0],img_list.shape[1]),device=args.gpus)
#                     d=img_list.shape[0]
#                     #make it 64x4 tensor 
#                     # print(b.min(),b.max())
#                     #for b in img_list:
#                     for i in range(d):
#                         # run calibration step

#                         p[i,:]=predict(img_list[i,...].type_as(model.fc.weight),model)
#                     p=torch.max(p,dim=0)[0]

#                     preds.append(p)
#                     print('finished preds in a batch! ',p)
#                 else:
#                       img=torch.tensor(batch[0],device=args.gpus)
#                       p=predict(img.type_as(model.fc.weight),model)
#                       preds.append(p)
#                       print('finished preds in a batch! ',p)
            
        

                   
          
       
        



#     preds=torch.cat(preds,axis=0)

   
#     torch.save(preds,f'preds_{order}_planet.pt')
    
# if __name__ == "__main__":
#     main()
