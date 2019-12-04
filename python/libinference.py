from torch.nn import MSELoss

from torchvision import transforms,datasets

import matplotlib.pyplot as plt

import torch.nn as nn
import os
import torch

from libmodel import simple_autoPrelu

from libcustomDataset import ImageFolderWithPaths

import numpy as np

import time

import matplotlib.pyplot as plt

from pytorch_msssim import *
import cv2

from PIL import Image
import shutil


import torch.nn.functional as F

import skimage



def inference(dataset_path):

    MODE="train1"
    data_transforms = {
        MODE: transforms.Compose([
            #transforms.CenterCrop(300),
            # transforms.Resize((64, 64)),
            #transforms.RandomCrop(size=(64,64)),
            #transforms.Resize((256, 256)),
            transforms.Grayscale(num_output_channels=1),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),

            #transforms.Normalize([0.54431105, 0.5803863, 0.53637147], [0.00527598, 0.00665597, 0.00575981])
        ])
    }

    data_dir="/media/velab/dati/Difetti/PallineScritte/64SizeNoBatchLinear/"

    dset = {x: ImageFolderWithPaths(os.path.join(data_dir, x), data_transforms[x])
         for x in [MODE]}


    dset_loaders = {x: torch.utils.data.DataLoader(dset[x], batch_size=1,
                                                   shuffle=False, num_workers=4)
                    for x in [MODE]}

    criterion= SSIM(data_range=1.0)

    criterion2 = torch.nn.MSELoss()

    loss=None

    ROOT_DATASET_RESIZE = "/media/velab/dati/Difetti/PallineScritte/64SizeNoBatch/out4/"




    # inference
    #model_inference = torch.load("./model/CsaeTrainingRandomCrop/ConvAuto800.pth")
    model_inference = torch.load("/media/velab/dati/Difetti/PallineScritte/64SizeNoBatchLinear/Model/{}/ConvAuto240.pth".format(MODE))


    
    model_inference.cuda()
    model_inference.train()
    train_loss=0.0
    index=0
    max=0.0

    max2=0.0
    max3=0.0

    array_loss=[]
    index2=0
    for i,data in enumerate(dset_loaders[MODE]):
        image, _ ,path= data


        data_image = image.unfold(2, 64, 64).unfold(3, 64, 64)

        data_image=data_image.contiguous().view(-1, 1, 64, 64)

        print(data_image.size())

       

        #input_image= output.unfold(0, 1, 1).unfold(1, 64, 64).unfold(2, 64, 64)

        #print(data_image.size())

        #input()

        # ciclo for su tutte le immagini 64x64

        index = 0
        #tensor=torch.zeros((1,512,512))

        #data = np.random.random((512, 512))


        #blocks = skimage.util.shape.view_as_blocks(data, (64, 64))



        # Do the processing on the blocks here.









        
        
        index_patch=0
        for k in range(0,2,1):
            for j in range(0,2,1):


                #tensor_image_out = data_image[0][k][j].view(-1,1,64,64)


                 




                tensor_image_in = data_image[index_patch].view(-1,1,64,64)


                output=model_inference(tensor_image_in.cuda())

                print(output.size())

               

               

                print(loss)
               



                #loss=1-criterion(tensor_image_in.cuda(),tensor_image_out.cuda())


                #data_image[0][k][j] = torch.sqrt((tensor_image_in.cpu().view(-1,64,64) - output.cpu().view(-1,64,64)) ** 2)


                # trova la differenza delle immagini
                data_image[index_patch] = torch.sqrt((tensor_image_in.cpu().view(-1,64,64) - output.cpu().view(-1,64,64)) ** 2)


                #data_image[index_patch] =  output.cpu().view(-1,64,64)



                #print(image_diff.size())


                #blocks[k][j]=image_diff.view(-1,64).detach().numpy()


                #single_pache=blocks[k][j]*255


                #img_block = Image.fromarray(single_pache)



                #print("NEW IMAGE")
                #print(image_diff.view(-1,64).detach().numpy())
                #input()

                image2_diff = transforms.ToPILImage()(data_image[index_patch].cpu().view(-1, 64, 64))

                #image2_diff.show()
                #input()

                index_patch=index_patch+1


                #image2_diff.show()

                #img_block.show()



        #print (data_image.size())


        tmp = data_image.view(1, 4, 64 * 64).permute(0, 2, 1)

        im = F.fold(tmp, (128, 128), (64, 64), 1, 0, (64, 64))


        image_reconstruct = transforms.ToPILImage()(im.cpu().view(-1, 128, 128))

        image_original= transforms.ToPILImage()(image.cpu().view(-1, 128, 128))

        image_reconstruct.show()

        image_original.show()

        # image_reconstruct.save(
        #                  ROOT_DATASET_RESIZE +"{}/".format(MODE)+"{}".format(os.path.basename(path[0])),
        #                  "PNG")

        # image_original.save("/media/velab/dati/Difetti/Texture/7/Random/original/val/{}_{}".format(index, os.path.basename(path[0])),
        #                 "PNG")

        input("image")