# utils
import copy
import math

import numpy as np
import torch
import os
import random
import torchvision
import warnings
from typing import Optional

#from configs import args

from src.models import resnet18, resnet34, resnet50, resnext50_32x4d
from src.models_vit import vit_base_patch16,vit_large_patch16, ViTFinetune
from src.models_mae import mae_vit_base_patch16_dec512d8b,mae_vit_large_patch16_dec512d8b,MaskedAutoencoderViT
import torch.nn.functional as F
import torchmetrics
# from configs import args
model_type = dict(
    resnet18=resnet18,
    resnet34=resnet34,
    resnet50=resnet50,
    resnext=resnext50_32x4d,
    vit=vit_base_patch16,
    vitL=vit_large_patch16,
    satlas=vit_base_patch16,
    mae=mae_vit_large_patch16_dec512d8b
)


def init_model(method, ckpt_path=None):
    '''
    :param method: str one of ['ckpt', 'imagenet', 'random']
    :param ckpt_path: str checkpoint path
    :return: tuple (ckpt_path:str , pretrained:bool)
    '''
    if method == 'ckpt':
        if ckpt_path:
            return ckpt_path, False
        else:
            raise ValueError('checkpoint path isnot provided')
    elif method == 'imagenet':
        return None, True
    else:
        return None, False


def get_model(model_name, in_channels, pretrained=False, ckpt_path=None):
    #TODO refactor this
    
    if 'vit' in model_name:
        #model=model_fn()
        print('in vit finetune')
        model = ViTFinetune(
                img_size=224,
                patch_size=16,
                in_chans=3,
                num_classes=2,
                embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
                drop_rate=0.1,
            )
    elif model_name=='satlas':
        model = torchvision.models.swin_transformer.swin_v2_b()
            

            # for param in self.feature_extractor.parameters():
            #     param.requires_grad = False
            # self.model = nn.Linear(1000, self.target_size)
    elif model_name=="mae":
        model= MaskedAutoencoderViT(
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4,
        )
    else:
        model_fn = model_type[model_name]
        model = model_fn(in_channels, pretrained)
    
    if ckpt_path:
        print(f'loading model from checkpoint{ckpt_path}')
        model = load_from_checkpoint(ckpt_path, model)
    return model


class Metric:
    "Metrics dispatcher. Adapted from answer at https://stackoverflow.com/a/58923974"

    def __init__(self, num_classes=None):
        self.num_classes = num_classes

    def get_metric(self, metric='r2'):
        """Dispatch metric with method"""

        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, metric, lambda: "Metric not implemented yet")

        return method()

    def acc(self):
        return torchmetrics.Accuracy(num_classes=self.num_classes,task="binary")

    def mse(self):
        return torchmetrics.MeanSquaredError()

    def f1(self):
        return torchmetrics.F1Score(num_classes=self.num_classes,task="binary")
    def auroc(self):
        return  torchmetrics.AUROC(num_classes=self.num_classes,task="binary")
#     def p(self):
#         return torchmetrics.F1Score(num_classes=self.num_classes,task='binary')
#     def recall(self):
#         return torchmetrics.AUROC(num_classes=self.num_classes,task='binary')



def seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:
    """Helper functions to help with reproducibility of models.
    from pytorch_lightning/utilities/seed.py
    Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random
    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
        workers: if set to ``True``, will properly configure all dataloaders passed to the
            Trainer with a ``worker_init_fn``. If the user already provides such a function
            for their dataloaders, setting this argument will have no influence. See also:
            :func:`~pytorch_lightning.utilities.seed.pl_worker_init_function`.
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = _select_seed_randomly(min_seed_value, max_seed_value)
            rank_zero_warn(f"No seed found, seed set to {seed}")
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = _select_seed_randomly(min_seed_value, max_seed_value)
                rank_zero_warn(f"Invalid seed found: {repr(env_seed)}, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        rank_zero_warn(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    # using `log.info` instead of `rank_zero_info`,
    # so users can verify the seed is properly set in distributed training.
    log.info(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed


def _warn(*args, stacklevel: int = 2, **kwargs):
    warnings.warn(*args, stacklevel=stacklevel, **kwargs)


def rank_zero_warn(*args, stacklevel: int = 4, **kwargs):
    _warn(*args, stacklevel=stacklevel, **kwargs)


def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    return random.randint(min_seed_value, max_seed_value)


def load_from_checkpoint(path, model):
    print(f'initializing model from pretrained weights at {path}')
    if 'moco' in path:
        # moco pretrained models need some weights renaming
        checkpoint = torch.load(path,map_location=torch.device('cpu'))
        # model.fc = torch.nn.Sequential()
        loaded_dict = checkpoint['state_dict']
       
        model_dict = model.state_dict()
        del loaded_dict["module.queue"]
        del loaded_dict["module.queue_ptr"]
        
        # load state dict keys
        for key_model, key_seco in zip(model_dict.keys(), loaded_dict.keys()):
            if 'fc' in key_model:
                #ignore fc weight
                continue
            model_dict[key_model] = loaded_dict[key_seco]
        model.load_state_dict(model_dict)
    elif 'fmow' in path or "mae" in path:
        checkpoint=torch.load(path)
        if 'fmow' in path:
            checkpoint_model=checkpoint['model']
        else:
            checkpoint_model=checkpoint
        print('fmow keys', checkpoint_model.keys())
   
        state_dict = model.state_dict()
        print('model keys',state_dict.keys())

        loaded_dict = checkpoint_model
        model_dict = model.state_dict()
        # del loaded_dict["channel_embed"]
        del loaded_dict['mask_token']
        del loaded_dict['decoder_pos_embed']
        # del loaded_dict['decoder_channel_embed']

        for key_model in model_dict.keys():
         
         if 'fc'  in key_model or 'head' in key_model :
#                 #ignore fc weight
              model_dict[key_model]= model_dict[key_model]
         else:
             model_dict[key_model]=loaded_dict[key_model]
          
        model.load_state_dict(model_dict)

        
    elif 'satlas' in path:
        full_state_dict = torch.load(path,map_location=torch.device('cpu'))
        swin_prefix = 'backbone.backbone.'
        swin_state_dict = {k[len(swin_prefix):]: v for k, v in full_state_dict.items() if k.startswith(swin_prefix)}

        
#             print(swin_state_dict.keys())
        model.load_state_dict(swin_state_dict)
    elif 'pretrain' in path:
        checkpoint=torch.load(path)
        checkpoint_model=checkpoint['model']
        print('pretrain keys', checkpoint_model.keys())
   
        state_dict = model.state_dict()
        print('model keys',state_dict.keys())

        loaded_dict = checkpoint_model
        model_dict = model.state_dict()
        del loaded_dict["decoder_channel_embed"]
        del loaded_dict['patch_embed.1.proj.weight']
        del loaded_dict['patch_embed.0.proj.weight']
        del loaded_dict['patch_embed.1.proj.bias']
        del loaded_dict['patch_embed.0.proj.bias']
        del loaded_dict['pos_embed']
        del loaded_dict['decoder_pos_embed']
  

        for key_model in model_dict.keys():
            if 'patch_embed' in key_model or 'decoder_pred.weight' in key_model or 'decoder_pred.bias' in key_model or 'pos_embed' in key_model or 'decoder_pos_embed' in key_model  :
                continue
         
            model_dict[key_model]=loaded_dict[key_model]
          
        model.load_state_dict(model_dict)
        
    else:
        ckpt = torch.load(path)

        model.load_state_dict(ckpt)
    model.to('cuda')
    #model.eval()
    return model


def get_full_experiment_name(experiment_name: str, batch_size: int,
                             conv_reg: float, lr: float, patch_size, ):
    if conv_reg < 1:
        conv_str = str(conv_reg).replace('.', '')
        conv_str = conv_str[1:]
    else:
        conv_str = str(conv_reg)
    if lr < 1:
        lr_str = str(lr).replace('.', '')
        lr_str = lr_str[1:]
    else:
        lr_str = str(lr)
    return f'{experiment_name}_b{batch_size}_conv{conv_str}_lr{lr_str}_crop{str(patch_size)}'


# cls_token', 'pos_embed', 'channel_embed', 'mask_token', 'decoder_pos_embed', 'decoder_channel_embed', 'patch_embed.0.proj.weight'

# 'cls_token', 'pos_embed', 'channel_embed', 'channel_cls_embed', 'patch_embed.0.proj.weight'
