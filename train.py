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

    # output directory
    dirpath = os.path.join(args.out_dir, experiment)
    print(f'checkpoints directory: {dirpath}')
    os.makedirs(dirpath, exist_ok=True)

    # Trainer
    trainer = Trainer(save_dir=dirpath, **params)

    best_loss, path = trainer.fit(train_loader, valid_loader, batcher_test, max_epochs=args.max_epochs, gpus=args.gpus,
                                  args=args)
    #score = trainer.test(batcher_test)

    return best_loss, path, \
           #score


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
    dataset = HarvestPatches(**data_params)
    if args.random_split:

        train, val, test = generate_random_splits(dataset, val_size=0.2, test_size=0.2)


    else:

        x_tr, x_val, x_test, y_tr, y_val, y_test = generate_stratified_splits(dataset.data['filename'],
                                                                              dataset.data['piles'].astype(bool),
                                                                              val_size=0.2, test_size=0.2)
        train = HarvestPatches(**data_params, X=x_tr, y=y_tr)
        val = HarvestPatches(**data_params, X=x_val, y=y_val)
        test = HarvestPatches(**data_params, X=x_test, y=y_test)
    # dataset_size = len(dataset)
    # validation_split = 0.2
    # indices = list(range(dataset_size))
    # split = int(np.floor(validation_split * dataset_size))
    #
    # np.random.seed(123)
    # np.random.shuffle(indices)
    # train_indices, val_indices = indices[split:], indices[:split]
    #
    # # Creating PT data samplers and loaders:
    # train_sampler = SubsetRandomSampler(train_indices)
    # valid_sampler = SubsetRandomSampler(val_indices)

    weights = make_balanced_weights(dataset.data)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size,
                                               sampler=sampler)
    validation_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size,
                                                    shuffle=False)
    test_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size,
                                              shuffle=False)
    # for i in train_loader:
    #     print(i)
    #     break
    ckpt, pretrained = init_model(args.model_init, args.init_ckpt_dir, )

    params = dict(model_name=args.model_name, in_channels=args.in_channels, ckpt_path=ckpt)

    encoder_params_filepath = os.path.join(dirpath, 'encoder_params.json')
    with open(encoder_params_filepath, 'w') as config_file:
        # json.dump(saved_encoder_params, config_file, indent=4)
        json.dump(params, config_file, indent=4)

    # save the encoder_params

    encoder = get_model(**params)
    # encoder=Encoder(self_attn=args.self_attn,**model_dict)
    # config = {"lr": args.lr, "wd": args.conv_reg}  # you can remove this now it is for raytune
    best_loss, best_path = setup_experiment(encoder, train_loader, validation_loader, args,
                                                   batcher_test=test_loader)


if __name__ == "__main__":
    wandb.init(project=args.wandp, entity=args.entity, config={})
    print('GPUS:', torch.cuda.device_count())

    main(args)