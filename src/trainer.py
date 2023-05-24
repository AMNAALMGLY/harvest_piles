import os
import shutil
import time
from collections import defaultdict

from torch.optim.lr_scheduler import ExponentialLR
import torch
import torch.nn as nn
from tqdm import tqdm

import wandb
from pl_bolts import optimizers
import torch.nn.functional as F
from configs import args
from src.utils import Metric
from timm.models.layers import trunc_normal_
patience = args.patience
import numpy as np
from timm.loss import LabelSmoothingCrossEntropy

class Trainer:
    """A trainer class for model training
     ...
    Attributes
    ----------
       - model
       - lr
       - weight_decay
       -loss_type
       -num_outputs
       -metric
       -optimizor
       -scheduler
       -criterion
    Methods:
    --------
        -init :initalization
        -shared_step: shared training step between training, validation and testing
        -training_step: shared step implemented for training data
        -validation_step: shared step implemented for validation data
        -fit : do the training loop over the dataloaders
        -setup criterion: sets loss function
        -configure optimizor:setup optim and scheduler
    """

    def __init__(self, model, lr, weight_decay, loss_type, num_outputs, metric, save_dir, sched, train_loader=None
                 , valid_loader=None, test_loader=None, **kwargs):

        '''Initializes the Trainer.
        Args
        - model:PreAct Model class
        - lr: int learning_rate
        - weight_decay: int
        - loss_type: str, one of ['classification', 'focal','cp']
        - num_outputs:output class  one of [None ,num_classes]
        - metric:List[str]  one of ['acc','f1' ,'percison', 'recall']
        '''
        super().__init__()

        self.model = model
        # init fc layer
        fc_in_dim = self.model.fc.in_features
        self.model.fc = nn.Linear(fc_in_dim, num_outputs)
          #   # manually initialize fc layer
        trunc_normal_(model.fc.weight, std=2e-5)
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        self.save_dir = save_dir
        self.num_outputs = num_outputs

        if args.no_of_gpus > 1:
            self.model = nn.DataParallel(self.model)
            self.typeAs = self.model.module.fc
        else:
            self.typeAs = self.model.fc
        self.model.to(args.gpus)

        self.metric_str = metric
        self.metric = []
        for m in metric:
            self.metric.append(Metric(self.num_outputs).get_metric(m))

        self.scheduler = self.configure_optimizers()['lr_scheduler'][sched]

        self.opt = self.configure_optimizers()['optimizer']

        self.setup_criterion()

    def _shared_step(self, batch, metric_fn):

        trainloss = 0

        x = torch.tensor(batch[0])
        # print(torch.isnan(batch[0]).any(),batch[1])
        if x is not None:
            label = batch[1]
            target = torch.tensor(label, )
            target = target.type_as(self.typeAs.weight)

            x = x.type_as(self.typeAs.weight)

           
            if len(batch)>2:
                #time extract
                ts=torch.tensor(batch[2]).type_as(self.typeAs.weight)
                outputs=self.model(x,ts)
            else:
                outputs = self.model(x)
            
            outputs = outputs.squeeze(dim=-1)

            # Loss

            trainloss = self.criterion(outputs, target)

            # Metric calculation
            if self.loss_type in ['classification' ,'focal','cb'] and self.num_outputs > 1:

                preds = nn.functional.softmax(outputs, dim=1)
                target = target.long()

            elif self.loss_type in ['classification' ,'focal','cb','labelsmooth']   and self.num_outputs == 1:
                preds = torch.sigmoid(outputs )
                target = target.long()

            else:
                preds = torch.tensor(outputs, device=args.gpus)

            for fn in metric_fn:
                fn.to(args.gpus)
                fn.update(preds, target)
        else:
            print('found nan img')
        return trainloss

    def training_step(self, batch, ):

        train_loss = self._shared_step(batch, self.metric)
        self.opt.zero_grad()

        train_loss.backward()

        self.opt.step()

        # log the outputs!

        return train_loss

    def validation_step(self, batch, ):
        loss = self._shared_step(batch, self.metric, )

        return loss

    def test_step(self, batch, ):
        loss = self._shared_step(batch, self.metric, )

        return loss

    def predict(self,batch, model):
        if model is None:
            model = self.model
        outputs = model(batch)
        outputs = outputs.squeeze(dim=-1)
        preds = torch.sigmoid(outputs, )

        preds = (preds > 0.5).type_as(batch)

        return preds

    def fit(self, trainloader, validloader, batcher_test, max_epochs, gpus, early_stopping=True,
            save_every=25, args=args):

        wandb.watch(self.model, log='all', log_freq=5)

        train_steps = len(trainloader)

        valid_steps = len(validloader)  # the number of batches
        best_loss = float('inf')
        count2 = 0  # count loss improving times

        resume_path = None
        val_list = defaultdict(lambda x: '')  #val loss after best epochs
        acc_list= defaultdict(lambda x: '')
        start = time.time()

        for epoch in range(max_epochs):

            epoch_start = time.time()

            with tqdm(trainloader, unit="batch") as tepoch:
                train_step = 0
                epoch_loss = 0

                print('-----------------------Training--------------------------------')
                self.model.train()
                self.opt.zero_grad()
                for record in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    train_loss = self._shared_step(record, self.metric)
                    train_loss.backward()
                    # Implementing gradient accumlation
                    if (train_step + 1) % args.accumlation_steps == 0:
                        self.opt.step()
                        self.opt.zero_grad()

                    epoch_loss += train_loss.item()
                    train_step += 1
                    # print statistics
                    print(f'Epoch {epoch} training Step {train_step}/{train_steps} train_loss {train_loss.item():.2f}')
                    if (train_step + 1) % 20 == 0:
                        running_train = epoch_loss / (train_step)
                        wandb.log({"train_loss": running_train, 'epoch': epoch})

                    tepoch.set_postfix(loss=train_loss.item())
                    time.sleep(0.1)

            # Metric calulation and average loss
            for i,m in enumerate(self.metric):
                 value = self.metric[i].compute()
                 wandb.log({f'{m} train': value, 'epoch': epoch})
            avgloss = epoch_loss / train_steps
            wandb.log({"Epoch_train_loss": avgloss, 'epoch': epoch})
            print(f'End of Epoch training average Loss is {avgloss:.2f}')
            for i in range(len(self.metric)):
                self.metric[i].reset()

            with torch.no_grad():
                valid_step = 0
                valid_epoch_loss = 0
                print('--------------------------Validation-------------------- ')
                self.model.eval()
                for record in validloader:

                    valid_loss = self.validation_step(record)
                    valid_epoch_loss += valid_loss.item()
                    valid_step += 1
                    print(
                        f'Epoch {epoch} validation Step {valid_step}/{valid_steps} validation_loss {valid_loss.item():.2f}')
                    if (valid_step + 1) % 20 == 0:
                        running_loss = valid_epoch_loss / (valid_step)
                        wandb.log({"valid_loss": running_loss, 'epoch': epoch})

                avg_valid_loss = valid_epoch_loss / valid_steps

                for i,m in enumerate(self.metric):
                     value = self.metric[i].compute()
                     if i==1:
                        acc_valid=value
                     wandb.log({f'{m} validation': value, 'epoch': epoch})
                for i in range(len(self.metric)):
                     self.metric[i].reset()
                print(f'Validation loss {avg_valid_loss}')
                
                wandb.log({"Epoch_valid_loss": avg_valid_loss, 'epoch': epoch})

                # AGAINST ML RULES : moniter test values
                _, _ = self.test(batcher_test)

                # early stopping with loss
                if best_loss - avg_valid_loss >= 0:
                    print('in loss improving loop by ')
                    print(best_loss - avg_valid_loss)
                    # loss is improving
                    counter = 0
                    count2 += 1
                    best_loss = avg_valid_loss
                    # start saving after a threshold of epochs and a patience of improvement
                    if count2 >= 1:
                        print('in best path saving')
                        save_path = os.path.join(self.save_dir, f'best_Epoch{epoch}.ckpt')
                        torch.save(self.model.state_dict(), save_path)
                        acc_list[acc_valid] = save_path
                        val_list[avg_valid_loss] = save_path
                        print(f'best model  in loss at Epoch {epoch} loss {avg_valid_loss} ')
                        print(f'Path to best model at loss found during training: \n{save_path}')



                elif best_loss - avg_valid_loss < 0:
                    # loss is degrading
                    print('in loss degrading loop by :')
                    print(best_loss - avg_valid_loss)
                    counter += 1  # degrading tracker
                    count2 = 0  # improving tracker
                    if counter >= patience and early_stopping:
                        print(f'.................Early stopping can be in this Epoch{epoch}.....................')
                        # break

            # Saving the model for later use every 10 epochs:
            if epoch % save_every == 0:
                resume_dir = os.path.join(self.save_dir, 'resume_points')
                os.makedirs(resume_dir, exist_ok=True)
                resume_path = os.path.join(resume_dir, f'Epoch{epoch}.ckpt')
                torch.save(self.model.state_dict(), resume_path)
                print(f'Saving model to {resume_path}')

            for i in range(len(self.metric)):
                     self.metric[i].reset()
            self.scheduler.step()

            print("Time Elapsed for one epochs : {:.2f}m".format((time.time() - epoch_start) / 60))

        # choose the best model between the saved models  minimum loss
#         if len(acc_list.keys()) > 0:
#             best_path = acc_list[max(acc_list.keys())]
#             print(f'{self.metric_str[0]} of best model saved is {max(acc_list.keys())} , path {best_path}')

#             shutil.move(best_path,
#                         os.path.join(self.save_dir, 'best.ckpt'))

        if len(val_list.keys()) > 0:
            best_path = val_list[min(val_list.keys())]
            print(f'loss of best model saved is {min(val_list.keys())} , path {best_path}')

            shutil.move(best_path,
                        os.path.join(self.save_dir, 'best.ckpt'))


        else:
            # best path is the last path which is saved at resume_points dir
            best_path = resume_path
            print(f'loss of best model saved from resume_point is {avg_valid_loss}')
            shutil.move(os.path.join(self.save_dir, best_path.split('/')[-2], best_path.split('/')[-1]),
                        os.path.join(self.save_dir, 'best.ckpt'))

            # better_path=best_path

            print("Time Elapsed for all epochs : {:.2} H".format((time.time() - start) / 120))
        best_path = os.path.join(self.save_dir, 'best.ckpt')
        return best_loss, best_path,
        # TODO implement overfit batches
        # TODO savelast

    def test(self, batcher_test):

        with torch.no_grad():
            test_step = 0
            test_epoch_loss = 0
            print('--------------------------Testing-------------------- ')
            self.model.eval()
            metrics = []
            for record in batcher_test:
                test_epoch_loss += self.test_step(record).item()
                test_step += 1

            for i, m in enumerate(self.metric):
                metrics.append((m.compute()))

                wandb.log({f'{m} Test': metrics[i], })

        return metrics, (test_epoch_loss / test_step)

    def configure_optimizers(self):
        if args.scheduler == 'cyclic':
            opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                   weight_decay=self.weight_decay)

        return {
            'optimizer': opt,
            'lr_scheduler': {
                'exp': ExponentialLR(opt,
                                     gamma=args.lr_decay),
                'cos': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=args.max_epochs),
                'warmup_cos': optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(opt, warmup_epochs=5,
                                                                                    max_epochs=200,
                                                                                    warmup_start_lr=1e-8),

                'step': torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=args.lr_decay, ),
                'cyclic': torch.optim.lr_scheduler.CyclicLR(opt, base_lr=1e-4, max_lr=self.lr, cycle_momentum=False),

            }
        }

    def setup_criterion(self):

        if self.loss_type == 'classification' and self.num_outputs > 1:
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss_type == 'classification' and self.num_outputs == 1:
            self.criterion = nn.BCEWithLogitsLoss()

        elif self.loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif self.loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif self.loss_type == 'smoothL1':
            self.criterion = nn.SmoothL1Loss(beta=3)
        elif self.loss_type=='labelsmooth':
             self.criterion = LabelSmoothingLoss()
        elif self.loss_type == 'focal':
            self.criterion = Sigmoid_Focal_Loss()
        
        elif self.loss_type == 'cb':
            self.criterion = CBSigmoid_Focal_Loss()
    



# def sigmoid_focal_loss(
#     inputs: torch.Tensor,
#     targets: torch.Tensor,
#     alpha: float = 0.1,
#     gamma: float = 2,
#     reduction: str = "none",
# ):
#     """
#     Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
#     Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
#
#     Args:
#         inputs: A float tensor of arbitrary shape.
#                 The predictions for each example.
#         targets: A float tensor with the same shape as inputs. Stores the binary
#                 classification label for each element in inputs
#                 (0 for the negative class and 1 for the positive class).
#         alpha: (optional) Weighting factor in range (0,1) to balance
#                 positive vs negative examples or -1 for ignore. Default = 0.25
#         gamma: Exponent of the modulating factor (1 - p_t) to
#                balance easy vs hard examples.
#         reduction: 'none' | 'mean' | 'sum'
#                  'none': No reduction will be applied to the output.
#                  'mean': The output will be averaged.
#                  'sum': The output will be summed.
#     Returns:
#         Loss tensor with the reduction option applied.
#     """
#
#     p = torch.sigmoid(inputs)
#     ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
#     p_t = p * targets + (1 - p) * (1 - targets)
#     loss = ce_loss * ((1 - p_t) ** gamma)
#
#     if alpha >= 0:
#         alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
#         loss = alpha_t * loss
#
#     if reduction == "mean":
#         loss = loss.mean()
#     elif reduction == "sum":
#         loss = loss.sum()
#
#     return loss

# class BinaryCrossEntropy(nn.Module):
#     """ BCE with optional one-hot from dense targets, label smoothing, thresholding
#     NOTE for experiments comparing CE to BCE /w label smoothing, may remove
#     """
#     def __init__(
#             self, smoothing=0.1, target_threshold = None, weight = None,
#             reduction: str = 'mean', pos_weight = None):
#         super(BinaryCrossEntropy, self).__init__()
#         assert 0. <= smoothing < 1.0
#         self.smoothing = smoothing
#         self.target_threshold = target_threshold
#         self.reduction = reduction
#         self.register_buffer('weight', weight)
#         self.register_buffer('pos_weight', pos_weight)

#     def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#             assert x.shape[0] == target.shape[0]
#         # if target.shape != x.shape:
#             # NOTE currently assume smoothing or other label softening is applied upstream if targets are already sparse
#             num_classes = 2
#             # FIXME should off/on be different for smoothing w/ BCE? Other impl out there differ
#             x= torch.nn.functional.one_hot(x)
#             off_value = self.smoothing / num_classes
#             on_value = 1. - self.smoothing + off_value
#             target = target.long().view(-1, 1)
#             target = torch.full(
#                 (target.size()[0], num_classes),
#                 off_value,
#                 device=x.device, dtype=x.dtype).scatter_(1, target, on_value)
            
#             # target=torch.argmax(target, dim=1)
#             if self.target_threshold is not None:
#             # Make target 0, or 1 if threshold set
#                 target = target.gt(self.target_threshold).to(dtype=target.dtype)
#             return F.binary_cross_entropy_with_logits(
#                 x, target,
#                 self.weight,
#                 pos_weight=self.pos_weight,
#                 reduction=self.reduction)
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing= 0.1, 
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing   = smoothing
        self.reduction = reduction
        self.weight    = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.sigmoid(preds)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.binary_cross_entropy(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)
class Sigmoid_Focal_Loss():
    
    """
#     Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
#     Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
#
#     Args:
#         inputs: A float tensor of arbitrary shape.
#                 The predictions for each example.
#         targets: A float tensor with the same shape as inputs. Stores the binary
#                 classification label for each element in inputs
#                 (0 for the negative class and 1 for the positive class).
#         alpha: (optional) Weighting factor in range (0,1) to balance
#                 positive vs negative examples or -1 for ignore. Default = 0.25
#         gamma: Exponent of the modulating factor (1 - p_t) to
#                balance easy vs hard examples.
#         reduction: 'none' | 'mean' | 'sum'
#                  'none': No reduction will be applied to the output.
#                  'mean': The output will be averaged.
#                  'sum': The output will be summed.
#     Returns:
#         Loss tensor with the reduction option applied.
#     """

    def __init__(
            self,
            alpha: float = -1,
            gamma: float = 1.5,
            reduction: str = "mean",
            **kwargs,
    ):
        """
        Quantile loss
        Args:
            quantiles: quantiles for metric
        """

        super(Sigmoid_Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma=gamma
        self.reduction=reduction

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
    
    
    
    
class CBSigmoid_Focal_Loss():
    
    """
#     Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
#     Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
#
#     Args:
#         inputs: A float tensor of arbitrary shape.
#                 The predictions for each example.
#         targets: A float tensor with the same shape as inputs. Stores the binary
#                 classification label for each element in inputs
#                 (0 for the negative class and 1 for the positive class).
#         alpha: (optional) Weighting factor in range (0,1) to balance
#                 positive vs negative examples or -1 for ignore. Default = 0.25
#         gamma: Exponent of the modulating factor (1 - p_t) to
#                balance easy vs hard examples.
#         reduction: 'none' | 'mean' | 'sum'
#                  'none': No reduction will be applied to the output.
#                  'mean': The output will be averaged.
#                  'sum': The output will be summed.
#     Returns:
#         Loss tensor with the reduction option applied.
#     """

    def __init__(
            self,
            alpha: float = 0.2,
            gamma: float = 1.5,
            reduction: str = "mean",
            **kwargs,
    ):
        """
        Quantile loss
        Args:
            quantiles: quantiles for metric
        """

        super(CBSigmoid_Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma=gamma
        self.reduction=reduction

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        

        return self.CB_loss(inputs,targets)
    
    def focal_loss(self,labels, logits, alpha,):
        """Compute the focal loss between `logits` and the ground truth `labels`.
        Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
        where pt is the probability of being classified to the true class.
        pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
        Args:
          labels: A float tensor of size [batch, num_classes].
          logits: A float tensor of size [batch, num_classes].
          alpha: A float tensor of size [batch_size]
            specifying per-example weight for balanced cross entropy.
          gamma: A float scalar modulating loss from hard and easy examples.
        Returns:
          focal_loss: A float32 scalar representing normalized total loss.
        """    
        BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * labels * logits - self.gamma * torch.log(1 + 
                torch.exp(-1.0 * logits)))

        loss = modulator * BCLoss

        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)

        focal_loss /= torch.sum(labels)
        return focal_loss
    
    def CB_loss(self,labels, logits, samples_per_cls=[141,672], no_of_classes=2, loss_type='sigmoid', beta=0.99):
        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        Args:
          labels: A int tensor of size [batch].
          logits: A float tensor of size [batch, no_of_classes].
          samples_per_cls: A python list of size [no_of_classes].
          no_of_classes: total number of classes. int
          loss_type: string. One of "sigmoid", "focal", "softmax".
          beta: float. Hyperparameter for Class balanced loss.
          gamma: float. Hyperparameter for Focal loss.
        Returns:
          cb_loss: A float tensor representing class balanced loss
        """
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes

        labels_one_hot = F.one_hot(labels.int(), no_of_classes).float()
        labels_one_hot=labels
        print('labels_one_hot',labels_one_hot.shape)
        weights = torch.tensor(weights,device=args.gpus).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1,no_of_classes)
        
        if loss_type == "focal":
            cb_loss = self.focal_loss(labels_one_hot, logits, weights)
        elif loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
        elif loss_type == "softmax":
            pred = logits.softmax(dim = 1)
            cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
        return cb_loss



