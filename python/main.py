import torch
import torch.nn as nn
#from mymodule1 import simple_autoPrelu
from libmodel import simple_autoPrelu
from libtrain import training



if __name__=="__main__":




    training("","")



    ten=torch.rand((1,1,64,64))

    model=  simple_autoPrelu()

    print(model)

    out=model(ten)

    print(out)


