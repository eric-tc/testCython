import torch

import torch.nn as nn


class Mcython(nn.Module):

    def __init__(self):

        super(Mcython,self).__init__()
        self.linear=nn.Linear(3,3)

    def forward(self,x):

        return self.linear(x)


class simple_autoPrelu(nn.Module):


    def __init__(self):

        super(simple_autoPrelu, self).__init__()

        self.Conv1 = nn.Sequential(nn.Conv2d(1, 32, 4, stride=2,padding=1),
                               nn.BatchNorm2d(32),
                               nn.PReLU())

        self.Conv2 = nn.Sequential(nn.Conv2d(32, 64, 4, stride=2,padding=1),
                               nn.BatchNorm2d(64),
                               nn.PReLU())

        self.Conv3 = nn.Sequential(nn.Conv2d(64, 128,5, stride=1),
                                   nn.BatchNorm2d(128),
                                   nn.PReLU())

        self.Conv4 = nn.Sequential(nn.Conv2d(128, 256, 4, stride=2,padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.PReLU())

        self.Conv5 = nn.Sequential(nn.Conv2d(256, 512, 5, stride=1,padding=0),
                                   nn.BatchNorm2d(512),
                                   nn.PReLU())

        self.Conv6 = nn.Sequential(nn.Conv2d(512, 512, 1, stride=1,padding=0))




        self.ConvTranspose6 = nn.Sequential(nn.Conv2d(512, 512, 1, stride=1,padding=0),
                                            nn.BatchNorm2d(512))


        #self.middle=nn.Sequential(nn.Conv2d(32, 1, 1, stride=1, padding=1))
        self.ConvTranspose5 = nn.Sequential(nn.ConvTranspose2d(512, 256, 5, stride=1,padding=0),
                                            nn.BatchNorm2d(256),
                                            nn.PReLU())

        self.ConvTranspose4 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2,padding=1),
                                            nn.BatchNorm2d(128),
                                            nn.PReLU())

        self.ConvTranspose3 = nn.Sequential(nn.ConvTranspose2d(128,64,5,stride=1,padding=0),
                                            nn.BatchNorm2d(64),
                                            nn.PReLU())

        self.ConvTranspose2 = nn.Sequential(nn.ConvTranspose2d(64,32,4,stride=2,padding=1),
                                            nn.BatchNorm2d(32),
                                            nn.PReLU())

        self.ConvTranspose1 = nn.Sequential(nn.ConvTranspose2d(32, 1, 4, stride=2,padding=1))



    def forward(self, x,y=None):
        #print(x.size())
        x=self.Conv1(x)
        #print(x.size())
        x=self.Conv2(x)
        #print(x.size())
        x=self.Conv3(x)
        #print(x.size())
        x=self.Conv4(x)
        #print(x.size())
        x=self.Conv5(x)
        #print(x.size())
        x_top=self.Conv6(x)
        #print(x_top.size())

        if y == None:
            x = self.ConvTranspose6(x_top)
        else:
            x = self.ConvTranspose6(y)

        #print(x.size())
        x=self.ConvTranspose5(x)
        #print(x.size())
        x=self.ConvTranspose4(x)
        #print(x.size())
        x=self.ConvTranspose3(x)
        #print(x.size())
        x=self.ConvTranspose2(x)
        #print(x.size())
        x=self.ConvTranspose1(x)
        #print(x.size())

        return x
