import glob

import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from torchvision import transforms

import os
from configs import args
from osgeo import gdal

from sklearn.model_selection import train_test_split
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
                 metalist=['lat_min', 'lat_max', 'lon_min', 'lon_max'], X=None, y=None, time=args.use_time):
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
        self.X = X
        self.y = y
        self.metalist = metalist
        self.label_name = label_name
        self.patch_size = patch_size
        self.augment = augment
        self.normalize = normalize
        self.clipn = clipn
        if self.normalize:
            # self.means, self.stds = self.norm(datadir,csv_dir)
            # print(self.means, self.stds)
             
             self.means, self.stds = np.array([104.32205321,  92.59582118,  81.39468476] ), np.array(
                 [26.22593466, 22.02386394, 21.15817767])
       
        if self.X is not None and self.y is not None:
            self.data = self.data.iloc[self.X.index, :].reset_index(drop=True)
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
        img_filename = os.path.join(self.datadir, str(self.data.loc[idx, 'filename']))
        image = self.load_geotiff(img_filename)

        if image is  None:
            image=np.empty((3,self.patch_size,self.patch_size))
        #preprocess the image to match the desired image size required
        if self.crop:
            image=transforms.CenterCrop(self.patch_size)(torch.tensor(image)).numpy()

        if self.resize:
            image = transforms.Resize(self.resize_size)(torch.tensor(image)).numpy()
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

        # locs = self.data.loc[idx, self.metalist].to_numpy()
        # locs = locs.astype(np.float32)
        labels = self.data.loc[idx, self.label_name]
        if not isinstance(labels, bool):
            labels = labels.astype(np.bool_)
        #time stamp
        if self.extract_time:
        
            ts_list =[self.data.loc[idx,'year'],self.data.loc[idx,'month']]
            ts=np.concatenate(ts_list,axis=0)
            return image, labels, ts
            
        # test
        #assert image.shape == (3, self.patch_size, self.patch_size), 'image shape is wrong'
        # assert locs.shape == (4,), 'locs shape is wrong'

        # example = {'images': image, 'locs': locs, 'labels': labels}

        # return example
        else:
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

    def load_geotiff(self, file):

        ds = gdal.Open(file)
        if not ds:
            return None
        r, g, b = np.array(ds.GetRasterBand(1).ReadAsArray()), np.array(ds.GetRasterBand(2).ReadAsArray()), np.array(
            ds.GetRasterBand(3).ReadAsArray())

        channels = [r, g, b]
        image = np.stack(channels, axis=0).astype(np.float64)

        return image

        # finidng the mean and standard deviation of all the images

    def norm(self,imgs_root_dir,csv_dir):
         
        data = pd.read_csv(csv_dir)
       
        data['filename']= str(imgs_root_dir) + data['filename'].astype(str)
        files= data['filename'].tolist()
        print(len(files))
     
        # img_filename = os.path.join(imgs_root_dir, str(self.data.loc[idx, 'filename']))
        # files = glob.glob(os.path.join(imgs_root_dir, '*.tif'))
        img_list = []
        i=0
        for file in files:
            img = self.load_geotiff(file)
            #print(img.shape)
            if img is not None  and img.shape !=(3,self.patch_size,self.patch_size):
                i+=1
                # print(img.shape)
                continue
            img_list.append(img)
        # print('i', i)
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


def generate_stratified_splits(X, y, val_size, test_size, stratify=True):
    # todo: does test set need to have the same distribution as train? No => need a fix
    if stratify:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=(val_size + test_size), stratify=y)
        test_size = int(len(X) * test_size)
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=test_size, stratify=y_val)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, stratify=y_train)

    return X_train, X_val, X_test, y_train, y_val, y_test
