# configs
from argparse import Namespace

import torch
import multiprocessing
import os

ROOT_DIR = '/atlas/u/amna/harvest_piles'
args = Namespace(

    # Model

    model_name='resnet18',
    hs_weight_init='samescaled',  # [same, samescaled,random]
    model_init='imagenet',
    imagenet_weight_path='./imagenet_resnet18_tensorpack.npz',

    # Training

    scheduler='warmup_cos',  # warmup_step, #warmup_cos   #step    #cos  #exp
    lr_decay=0.96,
    batch_size=64,
    gpu=-1,
    max_epochs=200,
    epoch_thresh=150,
    patience=20,
    lr=.001,

    conv_reg=0.01,

    # data
    image_size=64,
    in_channels=3,
    data_path='/atlas2/u/amna/harvest_piles/ethiopia_cogs_split_4326_64',

    labels_path='/atlas2/u/amna/harvest_piles/imgs_64.csv',

    label_name='piles',

    augment=False,
    clipn=True,
    normalize=False,

    # Experiment
    seed=123,
    experiment_name='binary_class',
    out_dir=os.path.join(ROOT_DIR, 'outputs'),
    init_ckpt_dir=None,

    loss_type='classification',

    num_outputs=1,
    resume=None,

    accumlation_steps=1,
    metric=['f1'],

    # Visualization
    # wandb project:
    wandb_p="test-project",
    entity="bias_migitation",

)
args.num_workers = multiprocessing.cpu_count()
args.no_of_gpus = torch.cuda.device_count()
args.gpus = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
