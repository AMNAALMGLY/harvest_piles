# Main
import json
from torch.utils.data import SubsetRandomSampler

import torch

import numpy as np
import wandb

import os

from configs import args
from dataset.data import HarvestPatches, generate_random_splits, generate_stratified_splits, \
    make_balanced_weights
from src.trainer import Trainer
from src.utils import init_model, get_model, get_full_experiment_name
import random
from src.models_time import Encoder

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
def setup_experiment(model, train_loader, valid_loader, args, batcher_test=None):
    '''
   setup the experiment paramaters
   :param model: model class :PreActResnet
   :param train_loader: Batcher
   :param valid_loader: Batcher
   :param resume_checkpoints: str saved checkpoints
   :param args: configs
   :return: best score, and best path string
   '''

    # setup Trainer params
    params = dict(model=model, lr=args.lr, weight_decay=args.conv_reg, loss_type=args.loss_type,
                  num_outputs=args.num_outputs, metric=args.metric, sched=args.scheduler)
    # logging
    wandb.config.update(params)
    # setting experiment_path
    experiment = get_full_experiment_name(args.experiment_name, args.batch_size,
                                          args.conv_reg, args.lr, args.image_size)
    wandb.config.update({'exp_name':experiment})

    # output directory
    dirpath = os.path.join(args.out_dir, experiment)
    print(f'checkpoints directory: {dirpath}')
    os.makedirs(dirpath, exist_ok=True)

    # Trainer
    trainer = Trainer(save_dir=dirpath, **params)

    best_loss, path = trainer.fit(train_loader, valid_loader, batcher_test, max_epochs=args.max_epochs, gpus=args.gpus,
                                  args=args)
    # score = trainer.test(batcher_test)

    return best_loss, path 
        # score


def main(args):
    # seed_everything(args.seed)
    # setting experiment_path
    experiment = get_full_experiment_name(args.experiment_name, args.batch_size,
                                          args.conv_reg, args.lr, args.image_size)

    dirpath = os.path.join(args.out_dir, experiment)
    os.makedirs(dirpath, exist_ok=True)

    # # save data_params for later use
    data_params = dict(datadir=args.data_path,
                       csv_dir=args.labels_path, augment=args.augment, normalize=args.normalize, clipn=args.clipn,
                       label_name=args.label_name,
                       patch_size=args.image_size)
    #                    label_name=args.label_name,
    #                    nl_label=args.nl_label, include_buildings=args.include_buildings, batch_size=args.batch_size,
    #                    groupby=args.group,img_size=args.image_size,crop=args.crop,rand_crop=args.rand_crop,offset=args.offset)

    params_filepath = os.path.join(dirpath, 'data_params.json')

    # #save the experiment path also
    # data_params.update(exp_path=dirpath)
    with open(params_filepath, 'w') as config_file:
        json.dump(data_params, config_file, indent=4)

    wandb.config.update(data_params)

    # Todo: define dataset /dataloader

    # Creating data indices for training and validation splits:
    if not args.random_split:
        dataset = HarvestPatches(**data_params)
        train_params = dict(datadir=args.data_path,
                            csv_dir='/atlas2/u/amna/harvest_piles/eighty_clust33.csv', augment=args.augment, normalize=args.normalize, clipn=args.clipn,
                            label_name=args.label_name,
                            patch_size=args.image_size)
#         '/atlas2/u/amna/harvest_piles/ninety_train_clust.csv'
# '/atlas2/u/amna/harvest_piles/sixty_train_clust.csv'
# '/atlas2/u/amna/harvest_piles/two_clust.csv'
# '/atlas2/u/amna/harvest_piles/eighty_clust.csv'
# '/atlas2/u/amna/harvest_piles/forty_clust.csv'
        val_params = dict(datadir=args.data_path,
                          csv_dir='/atlas2/u/amna/harvest_piles/valid_cluster_cent33.csv', augment=False, normalize=args.normalize, clipn=args.clipn,
                          label_name=args.label_name,
                          patch_size=args.image_size)
          #should be a fixed test set 
        test_params = dict(datadir=args.data_path,
                           csv_dir='/atlas2/u/amna/harvest_piles/test_cluster_cent33.csv', augment=False, normalize=args.normalize, clipn=args.clipn,
                           label_name=args.label_name,
                           patch_size=args.image_size)
    else:
           train_params = dict(datadir=args.data_path,
                            csv_dir='/atlas2/u/amna/harvest_piles/train2.csv', augment=args.augment, normalize=args.normalize, clipn=args.clipn,
                            label_name=args.label_name,
                            patch_size=args.image_size)
           val_params = dict(datadir=args.data_path,
                          csv_dir='/atlas2/u/amna/harvest_piles/val2.csv', augment=False, normalize=args.normalize, clipn=args.clipn,
                          label_name=args.label_name,
                          patch_size=args.image_size)
          #should be a fixed test set 
           test_params = dict(datadir=args.data_path,
                           csv_dir='/atlas2/u/amna/harvest_piles/test2.csv', augment=False, normalize=args.normalize, clipn=args.clipn,
                           label_name=args.label_name,
                           patch_size=args.image_size)
        
    train = HarvestPatches(**train_params)

    val = HarvestPatches(**val_params)
    
  
    test = HarvestPatches(**test_params)


    
    train_df = train.data
    if args.balanced:
        weights = make_balanced_weights(train_df)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights),replacement=False)
        train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size,
                                                   sampler=sampler,collate_fn=train.collate_fn)
    else:
        train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size,
                                                   shuffle=True,collate_fn=train.collate_fn,
                                            num_workers=args.num_workers, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size,
                                                    num_workers=args.num_workers,
                                                    shuffle=False,collate_fn=val.collate_fn,
                                                    pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              shuffle=False,collate_fn=test.collate_fn,
                                              pin_memory=True)

    ckpt, pretrained = init_model(args.model_init, args.init_ckpt_dir, )

    params = dict(model_name=args.model_name, in_channels=args.in_channels, ckpt_path=ckpt)

    encoder_params_filepath = os.path.join(dirpath, 'encoder_params.json')
    
    with open(encoder_params_filepath, 'w') as config_file:
         # save the encoder_params
        json.dump(params, config_file, indent=4)

  
    if args.use_time:
        models=[]
        for i,name in enumerate(args.model_name): #assume model name and model_init are both lists ( should do assert) 
            ckpt, pretrained = init_model(args.model_init[i], args.init_ckpt_dir,)
            params = dict(model_name=name, in_channels=args.in_channels, ckpt_path=ckpt)
            models[i]=get_model(**params)
        encoder=Encoder(models[0],models[1])
    else:
        encoder = get_model(**params)
    # encoder=Encoder(self_attn=args.self_attn,**model_dict)
    # config = {"lr": args.lr, "wd": args.conv_reg}  # you can remove this now it is for raytune
    
    best_loss, best_path = setup_experiment(encoder, train_loader, validation_loader, args,
                                            batcher_test=test_loader)


if __name__ == "__main__":
    wandb.init(project="harvest_piles", entity="amna", config={})
    print('GPUS:', torch.cuda.device_count())

    main(args)
