from libmodel import simple_autoPrelu

from torchvision import transforms,datasets

from torch.optim import Optimizer

import numpy as np

import torch

from pytorch_msssim import *


import os

from libcustomDataset import ImageFolderWithPaths


class CyclicLR(object):
    """
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for eachparam groups.
            Default: 0.001
        max_lr (float or list): Upper boundaries in the cycle for
            each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function. Default: 0.006
        step_size (int): Number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch. Default: 2000
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        last_batch_iteration (int): The index of the last batch. Default: -1
    Example:
         optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
         scheduler = torch.optim.CyclicLR(optimizer)
         data_loader = torch.utils.data.DataLoader(...)
         for epoch in range(10):
             for batch in data_loader:
                 scheduler.batch_step()
                 train_batch(...)
    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration



    def batch_step(self, batch_iteration=None):
        """
        questa funzione e richiamata all nell loop di training 
        quando eseguo le diverse iterazioni
        
        
        :param batch_iteration: valore int del numero batch_size
        :return: 
        """

        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    # metodi che possono essere scelti per cambiare
    #la scelta progeressiva del lr
    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)



    def get_lr(self):

        """
           funzione richiamata da batch_step che calcola per ogni batch_size
           il valore del lr che causa una variazione nella loss maggiore

           :return: 
        """
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs


def training(dataset_path, folder_save):


    array=[1,2,3,4]

    for i in array:

        #Data preparation
        MODE="train{}".format(i)
        data_transforms = {
            MODE: transforms.Compose([
                transforms.RandomCrop(size=(64,64)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ]),
        }

        #data_dir="/media/velab/dati/Difetti/DatasetPalline/dataset1/"
        data_dir="/media/velab/dati/Difetti/PallineScritte/64SizeNoBatchLinear/"



        dset = {x: ImageFolderWithPaths(os.path.join(data_dir, x), data_transforms[x])
             for x in [MODE]}


        dset_loaders = {x: torch.utils.data.DataLoader(dset[x], batch_size=1,
                                                       shuffle=True, num_workers=4)
                        for x in [MODE]}


        #find_normalize(dset_loaders)
        #criterion= torch.nn.MSELoss()

        #criterion=torch.nn.L1Loss()
        #criterion=PercentileLoss()

        criterion = SSIM(data_range=1.0)





        # i risultati con la batch norm sono migliori rispetto al group norm
        #model = CasaeGroup()
        #model=Casae()

        #model=AutoencoderSSIM()

        model=simple_autoPrelu()


        #model= torch.load("/media/velab/dati/Difetti/PallineScritte/64SizeNoBatch/Model/{}/ConvAuto300.pth".format(MODE))

        optimizer=torch.optim.SGD(model.parameters(),lr=0.0001,momentum=0.9)

        scheduler = CyclicLR(optimizer)


        criterion.cuda()
        model.cuda()

        
        loss=None
      
    
      

        for epoch in range(0,250,1):
            train_loss = 0.0
            loss_final=0.0

            val_loss = 0.0
            val_loss_final = 0.0
            array_loss = []
            print("EPOCH {}".format(epoch))
            if training:
                model.train()
                for data in dset_loaders[MODE]:


                    scheduler.batch_step()

                    image,_,path=data


                    optimizer.zero_grad()

                    in_data= image.cuda()


                    out=model(in_data)

                    out=out.view(1,1,64,64)


                    loss=1- criterion(out,image.cuda())


                    array_loss.append(loss.item())

               

                    train_loss += loss.item()
                    loss.backward()

                    optimizer.step()


            train_loss = train_loss / len(dset_loaders[MODE])

          
           

            print("LOSS; {};".format(train_loss))
  


            if epoch%10==0:
                torch.save(model, "/media/velab/dati/Difetti/PallineScritte/64SizeNoBatchLinear/Model/train{}/ConvAuto{}.pth".format(i,epoch))
            else:
                #torch.save(model, "/media/velab/dati/Difetti/Texture/7/Random/Model/ConvAuto{}.pth".format(epoch))
                pass



