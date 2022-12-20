# configs
from argparse import Namespace

import torch
import multiprocessing
import os

ROOT_DIR = '/atlas/u/amna/harvest_piles'
args = Namespace(

    # Model

    model_name= ['resnet50','mlp'], #'vitL', #'resnet50',
    hs_weight_init='samescaled',  # [same, samescaled,random]
    model_init= ['ckpt',None], #'ckpt',    #'imagenet',
    imagenet_weight_path='./imagenet_resnet18_tensorpack.npz', #TODO delete this!

    # Training

    scheduler='warmup_cos',  # warmup_step, #warmup_cos   #step    #cos  #exp
    lr_decay=0.96, # we are not using it
    batch_size=16,
    gpu=-1,
    max_epochs=200,
    epoch_thresh=150, #we are not using it
    patience=20,
    lr=.0001,

    conv_reg=0.01,

    # data
    
    random_split=True, 
    image_size=512,
    in_channels=3,
    use_time=True,
    data_path='/atlas2/u/amna/harvest_piles/skysat_clip_512_4326_cogs_attempt_2/',

    #not used anymore in train.py!
    labels_path= '/atlas2/u/amna/harvest_piles/processed_labels.csv',
    #'/atlas2/u/amna/harvest_piles/labels_v0.csv',
    #'/atlas2/u/amna/harvest_piles/imgs_512_processed.csv', #'/atlas2/u/amna/harvest_piles/labels_v0.csv',

    label_name='has_piles',

    augment=True,
    clipn=True,
    normalize=True,

    # Experiment
    seed=123,
    experiment_name='imbalanced_data_sigmoid_time_ssl_mlp',
    out_dir=os.path.join(ROOT_DIR, 'outputs'),
    init_ckpt_dir= '/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/fmow/resnet50/cpc_500/lr-0.03_bs-256_t-0.02_mocodim-128_temporal_224_exactly_same_as_32x32_with_add_transform_geohead_corrected/checkpoint_0200.pth.tar',

    #'/atlas/u/amna/harvest_piles/fmow_pretrain.pth',
    # 
    # 
    loss_type='classification',#'focal',  #'classification',

    num_outputs=1,
    resume=None,

    accumlation_steps=2, 
    metric=['f1','acc'],

    # Visualization
    # wandb project:
    wandb_p="test-project",
    entity="bias_migitation",

)
args.num_workers = multiprocessing.cpu_count()
args.no_of_gpus = torch.cuda.device_count()
args.gpus = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
