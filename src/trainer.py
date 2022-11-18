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

patience = args.patience


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
        - loss_type: str, one of ['classification', 'mse','smoothL1']
        - num_outputs:output class  one of [None ,num_classes]
        - metric:List[str]  one of ['r2','R2' ,'mse', 'rank']
        '''
        super().__init__()

        self.label_name = args.label_name
        self.model = model
        # init fc layer
        fc_in_dim = self.model.fc.in_features
        self.model.fc = nn.Linear(fc_in_dim, num_outputs)
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

        x = torch.tensor(batch[0][:, :, :, :])

        if x is not None:
            # lats = batch[2][:, :2]
            # longs = batch[2][:, 2:]

            label = batch[1]
            target = torch.tensor(label, )
            target = target.type_as(self.typeAs.weight)

            x = x.type_as(self.typeAs.weight)

            # reshaping

            outputs = self.model(x)
            outputs = outputs.squeeze(dim=-1)

            # Loss

            trainloss = self.criterion(outputs, target)

            # Metric calculation
            if self.loss_type == 'classification' and self.num_outputs > 1:

                preds = nn.functional.softmax(outputs, dim=1)
                target = target.long()

            elif self.loss_type == 'classification' and self.num_outputs == 1:
                preds = torch.sigmoid(outputs, )
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

        preds = (preds >= 0.5).type_as(batch)

        return preds

    def fit(self, trainloader, validloader, batcher_test, max_epochs, gpus, class_model=None, early_stopping=True,
            save_every=25, args=args):

        wandb.watch(self.model, log='all', log_freq=5)

        train_steps = len(trainloader)

        valid_steps = len(validloader)  # the number of batches
        best_loss = float('inf')
        count2 = 0  # count loss improving times

        r2_dict = defaultdict(lambda x: '')
        resume_path = None
        val_list = defaultdict(lambda x: '')
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
            r2 = self.metric[0].compute()
            wandb.log({f'{self.metric_str[0]} train': r2, 'epoch': epoch})
            avgloss = epoch_loss / train_steps
            wandb.log({"Epoch_train_loss": avgloss, 'epoch': epoch})
            print(f'End of Epoch training average Loss is {avgloss:.2f} and {self.metric_str[0]} is {r2:.2f}')
            self.metric[0].reset()

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

                r2_valid = self.metric[0].compute()
                self.metric[0].reset()
                print(f'Validation {self.metric_str[0]} is {r2_valid:.2f} and loss {avg_valid_loss}')
                wandb.log({f'{self.metric_str[0]} valid': r2_valid, 'epoch': epoch})
                wandb.log({"Epoch_valid_loss": avg_valid_loss, 'epoch': epoch})

                # AGAINST ML RULES : moniter test values
                r2_test, test_loss = self.test(batcher_test)

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
                        # save r2 values and loss values
                        r2_dict[r2_valid] = save_path
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

            self.metric[0].reset()
            self.scheduler.step()

            print("Time Elapsed for one epochs : {:.2f}m".format((time.time() - epoch_start) / 60))

        # choose the best model between the saved models in regard to r2 value or minimum loss
        if len(r2_dict.keys()) > 0:
            best_path = r2_dict[max(r2_dict.keys())]
            print(f'{self.metric_str[0]} of best model saved is {max(r2_dict.keys())} , path {best_path}')

            shutil.move(best_path,
                        os.path.join(self.save_dir, 'best.ckpt'))

        elif len(val_list.keys()) > 0:
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
            r2_test = []
            for record in batcher_test:
                test_epoch_loss += self.test_step(record).item()
                test_step += 1

            for i, m in enumerate(self.metric):
                r2_test.append((m.compute()))

                wandb.log({f'{self.metric_str[i]} Test': r2_test[i], })

        return r2_test[0], (test_epoch_loss / test_step)

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
        elif self.loss_type == 'focal':
            self.criterion = Sigmoid_Focal_Loss()



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


class Sigmoid_Focal_Loss():
    """
    Quantile loss, i.e. a quantile of ``q=0.5`` will give half of the mean absolute error as it is calcualted as
    Defined as ``max(q * (y-y_pred), (1-q) * (y_pred-y))``
    """

    def __init__(
            self,
            alpha: float = 0.1,
            gamma: float = 2,
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
