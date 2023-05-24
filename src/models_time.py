from typing import Type, Any, Callable, Union, List, Optional, no_type_check
import torch
from torch import Tensor
import torch.nn as nn

import time

class MLP(nn.Module):
   def __init__(self, input_dim, output_dim=512):
      super().__init__()
      self.input_dim=input_dim
      self.output_dim=output_dim
      self.layer1=nn.Linear(input_dim,output_dim *2)
      self.layer2=nn.Linear(output_dim *2,output_dim)
      #self.layer3=nn.Linear(output_dim//4 , output_dim)
      self.fc=nn.Linear(output_dim,1)
      #self.layer3=nn.Linear(output_dim,1)
      self.relu=nn.ReLU()
   def forward(self,x):
       return self.fc(self.relu(self.layer2( self.relu(self.layer1(x)))))
              #self.layer2( self.relu(self.layer1(x)))

def mlp(in_channels: int, pretrained: bool = False, progress: bool = True, **kwargs: Any) :
    """Mlp model for encoding time
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """


    return MLP(in_channels)

class Encoder(nn.Module):
   def __init__(self,img_model,time_model):
        in_features=img_model.fc.in_features + time_model.fc.in_features
        out_features=img_model.fc.out_features
        self.fc=nn.Linear(in_features,out_features)
        img_model.fc=nn.Sequential()
        time_model.fc=nn.Sequential()
        self.model=nn.Module_dict({'img':img_model,'time':time_model,'fc':self.fc})
        
        
   def forward(self,x,t):
       img=self.model['img'](x)
       time=self.model['fc'](t)
       return self.model['fc'](torch.cat((img,time),dim=-1))
    
         
    
    
# def encoder(in_channels: int = 3, pretrained: bool = False, progress: bool = True, **kwargs: Any) :
#     r"""Encoder
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """


#     return Encoder()





