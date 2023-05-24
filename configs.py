# configs
from argparse import Namespace

import torch
import multiprocessing
import os

ROOT_DIR = '/atlas/u/amna/harvest_piles'
args = Namespace(

    # Model

    model_name= 'resnet50',
    #['resnet50','mlp'], #'vitL', #'resnet50',
    hs_weight_init='same',  # [same, samescaled,random]
    model_init= 'ckpt',
    #['ckpt',None], #'ckpt',    #'imagenet',
    imagenet_weight_path='./imagenet_resnet18_tensorpack.npz', #TODO delete this!
         
    # Training

    scheduler='step',  # warmup_step, #warmup_cos   #step    #cos  #exp
    lr_decay=0.97, 
    batch_size=32,
    gpu=-1,
    max_epochs=100,
    epoch_thresh=150, #we are not using it
    patience=20,
    lr=0.0003,

    conv_reg=0.1,

    # data
    rescale=False,
    balanced= True,
    random_split=False, 
    image_size=512, #512 or #55  or 53
    in_channels=3,
    use_time=False,
    data_path='/atlas2/u/amna/harvest_piles/skysat_clip_512_4326_cogs_attempt_2/',
    #'/atlas2/u/amna/harvest_piles/planetscope_skysat_location/',
    #skysat_clip_512_4326_cogs_attempt_2/',

    #not used anymore in train.py!
    labels_path= '/atlas2/u/amna/harvest_piles/balanced_final_merged (1).csv',  #this is the final merged file with ~5200 before balancing
    
    #balanced_5k (1).csv',
    #4k_labels.csv',
    #'/atlas2/u/amna/harvest_piles/processed_labels.csv',
    #'/atlas2/u/amna/harvest_piles/labels_v0.csv',
    #'/atlas2/u/amna/harvest_piles/imgs_512_processed.csv', #'/atlas2/u/amna/harvest_piles/labels_v0.csv',

    label_name='has_piles',

    augment=True,
    clipn=False,
    normalize=True,

    # Experiment
    seed=123,
    experiment_name='balanced_5kfinal_cluster300_80',
    out_dir=os.path.join(ROOT_DIR, 'outputs','new_split','lrcurve_skysat'),
    init_ckpt_dir='/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/fmow/resnet50/cpc_500/lr-0.03_bs-256_t-0.02_mocodim-128_temporal_224_exactly_same_as_32x32_with_add_transform_geohead_corrected/checkpoint_0200.pth.tar',
    #'/atlas/u/amna/harvest_piles/finetune-vit-base-e7.pth',
#/atlas/u/amna/harvest_piles/mae_pretrain_vit_base (1).pth
    #'/atlas/u/amna/harvest_piles/pretrain-vit-base-e199.pth',
    #'/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/fmow/resnet50/cpc_500/lr-0.03_bs-256_t-0.02_mocodim-128_temporal_224_exactly_same_as_32x32_with_add_transform_geohead_corrected/checkpoint_0200.pth.tar',

    #'/atlas/u/amna/harvest_piles/fmow_pretrain.pth',
    # 
    # 
    loss_type='classification',
    #'classification',#'focal',  #'classification', labelsmooth

    num_outputs=1,
    resume=None,

    accumlation_steps=2, 
    metric=['f1','acc','auroc'],

    # Visualization
    # wandb project:
    wandb_p="test-project",
    entity="bias_migitation",

)
args.num_workers = multiprocessing.cpu_count()
args.no_of_gpus = torch.cuda.device_count()
args.gpus = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
