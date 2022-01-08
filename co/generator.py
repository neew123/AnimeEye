import torch
import torch.nn as nn
from miscc.config import cfg
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        # input shape : B * 6 * h * w
        self.conv1 = nn.Conv2d(6,16,kernel_size=7,stride=1,padding = 2)
        self.in1 = nn.InstanceNorm2d(16)
        self.act1 = nn.ReLU()
        # input shape : B * 16 * h * w
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding = 1)
        self.in2 = nn.InstanceNorm2d(32)
        self.act2 = nn.ReLU()
        # input shape : B * 32 * h/2 * w/2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding = 1)
        self.in3 = nn.InstanceNorm2d(64)
        self.act3 = nn.ReLU()
        # input shape : B * 64 * h/4 * w/4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding = 1)
        self.in4 = nn.InstanceNorm2d(128)
        self.act4 = nn.ReLU()
        # input shape : B * 128 * h/8 * w/8
        self.conv5 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding = 1)
        self.in5 = nn.InstanceNorm2d(256)
        self.act5 = nn.ReLU()
        # input shape : B * 256 * h/16 * w/16
        self.conv5 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding = 1)
        self.in5 = nn.InstanceNorm2d(256)
        self.act5 = nn.ReLU()

        # input shape : 256 * h/32 * w/32
        self.ngf = 4
        ngf = self.ngf
        self.fc1 = nn.Linear(in_features=256 * ngf * ngf,out_features=256)
        self.fc2 = nn.Linear(in_features=512,out_features=256 * ngf * ngf)
        self.act = nn.ReLU()
        # out shape : 256 * h/32 * w/32

        # input shape : B * 256 * h/32 * w/32
        self.deconv1 = nn.ConvTranspose2d(256,256,kernel_size=3,stride=1,padding=1)
        self.de_in1 = nn.InstanceNorm2d(256)
        self.de_act1 = nn.ReLU()
        # input shape : B * 256 * h/16 * w/16
        self.deconv2 = nn.ConvTranspose2d(256,128,kernel_size=3,stride=1,padding=1)
        self.de_in2 = nn.InstanceNorm2d(128)
        self.de_act2 = nn.ReLU()
        # input shape : B * 128 * h/8 * w/8
        self.deconv3 = nn.ConvTranspose2d(128,64,kernel_size=3,stride=1,padding=1)
        self.de_in3 = nn.InstanceNorm2d(64)
        self.de_act3 = nn.ReLU()
        # input shape : B * 64 * h/4 * w/4
        self.deconv4 = nn.ConvTranspose2d(64,32,kernel_size=3,stride=1,padding=1)
        self.de_in4 = nn.InstanceNorm2d(32)
        self.de_act4 = nn.ReLU()
        # input shape : B * 32 * h/2 * w/2
        self.deconv5 = nn.ConvTranspose2d(32,16,kernel_size=3,stride=1,padding=1)
        self.de_in5 = nn.InstanceNorm2d(16)
        self.de_act5 = nn.ReLU()
        # input shape : B * 16 * h * w
        self.deconv6 = nn.Conv2d(16,3,kernel_size=7,stride=1,padding=1)
        self.de_act6 = nn.Tanh()
        # output shape : B * 3 * h * w

    def forward(self,xc,xm,left_latent,right_latent):
        x = torch.cat((xc,xm),dim=1) # B * C * H * W
        x = self.act1(self.in1(self.conv1(x)))
        x = self.act2(self.in2(self.conv2(x)))
        x = self.act3(self.in3(self.conv3(x)))
        x = self.act4(self.in4(self.conv4(x)))
        x = self.act5(self.in5(self.conv5(x)))

        x = torch.flatten(x,1) # batch * 256*h/32*w/32
        x = self.fc1(x) # output : 256 * 1
        x = torch.concat((x,left_latent,right_latent),dim=1) # 512 * 1
        x = self.act(self.fc2(x))
        x = x.view(-1,256,self.ngf,self.ngf)

        x = self.de_act1(self.de_in1(self.deconv1(x)))
        x = self.de_act2(self.de_in2(self.deconv2(x)))
        x = self.de_act3(self.de_in3(self.deconv3(x)))
        x = self.de_act4(self.de_in4(self.deconv4(x)))
        x = self.de_act5(self.de_in5(self.deconv5(x)))
        x = self.de_act6(self.de_in6(self.deconv6(x)))

        return x


