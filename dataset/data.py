from pathlib import Path
import shutil
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms
import tarfile
import datetime
import pytz
import os
from PIL import Image
from tqdm import tqdm
from osgeo import gdal


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

    def __init__(self, datadir, csv_dir, augment, normalize, clipn, patch_size, label_name='piles',
                 metalist=['lat_min', 'lat_max', 'lon_min', 'lon_max']):
        '''
        Args

        datadir:str of images dir
        csv_dir:str of csv file directory
        transform:boolean
        patch_size:int
        '''
        self.datadir = datadir

        self.data = pd.read_csv(csv_dir)
        self.metalist = metalist
        self.label_name = label_name
        self.augment = augment
        self.normalize = normalize
        self.clipn = clipn
        self.patch_size = patch_size
        # self.metadata

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):
        # read the images
        # the metadata
        # the labels
        img_filename = os.path.join(self.datadir, str(self.data.loc[idx, 'filename']))

        image = self.load_geotiff(img_filename)
        # preprocessing
        if self.clipn:
            image = np.clip(image, a_min=0., a_max=1e20)

        if self.normalize:
            # per channel normalization
            mean = image.mean(axis=0)
            std = image.std(axis=0)
            image = (image - mean) / std

        if self.augment:
            image = Image.fromarray(image)
            image = self.augment()(image)
            image = image.numpy()

        locs = self.data.loc[idx, self.metalist].to_numpy()
        locs = locs.astype(np.float32)
        labels = self.data.loc[idx, self.label_name].astype(np.float32)
        # convert to binary
        labels = labels.astype(bool)
        # test
        assert image.shape == (3, self.patch_size, self.patch_size), 'image shape is wrong'
        assert locs.shape == (4,), 'locs shape is wrong'

        example = {'images': image, 'locs': locs, 'labels': labels}

        # return example
        return image, labels

    def augment(self):
        # pass
        if self.augment:
            aug = transforms.Compose([transforms.RandomRotation(90), transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(), ])
        return aug

    def load_geotiff(self, file):

        ds = gdal.Open(file)
        r, g, b = np.array(ds.GetRasterBand(1).ReadAsArray()), np.array(ds.GetRasterBand(2).ReadAsArray()), np.array(
            ds.GetRasterBand(3).ReadAsArray())

        channels = [r, g, b]
        image = np.stack(channels, axis=0).astype(np.float64)

        return image
