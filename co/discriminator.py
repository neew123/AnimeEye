import torch
import torch.nn as nn
from miscc.config import cfg
import torch.nn.functional as F


# 3 * 64 * 64
class LocalDiscriminator(nn.Module):
    def __init__(self,input_shape):
        super(LocalDiscriminator, self).__init__()
        self.input_shape = input_shape
        self.img_c = input_shape[0]
        self.img_h = input_shape[1]
        self.img_w = input_shape[2]

        # input: 3 * h * w
        self.conv1 = nn.Conv2d(self.img_c, 32, kernel_size=5, stride=2, padding=2)
        self.act1 = nn.LeakyReLU()
        # input: 32 * h/2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.act2 = nn.LeakyReLU()
        # input: 64 * h/4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.act3 = nn.LeakyReLU()
        # input: 128 * h/8
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.act4 = nn.LeakyReLU()
        # input: 256 * h/16
        self.conv5 = nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)
        self.act5 = nn.LeakyReLU()
        # input: 256 * h/32
        ndf = self.img_h//32
        self.fc = nn.Linear(in_features=256 * ndf * ndf,out_features=256)

    def forward(self,x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.act5(self.conv5(x))

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 3 * 128 * 128
class GlobalDiscriminator(nn.Module):
    def __init__(self,input_shape):
        super(GlobalDiscriminator, self).__init__()
        self.input_shape = input_shape
        self.out_shape = (1024,)

        self.img_c = input_shape[0]
        self.img_h = input_shape[1]
        self.img_w = input_shape[2]
        # input shape : 3 * h * w
        self.conv1 = nn.Conv2d(self.img_c, 32, kernel_size=5, stride=2, padding=2)
        self.act1 = nn.LeakyReLU()
        # input shape : 32 * h/2 * w/2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.act2 = nn.LeakyReLU()
        # input shape : 64 * h/4 * w/4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.act3 = nn.LeakyReLU()
        # input shape : 128 * h/8 * w/8
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.act4 = nn.LeakyReLU()
        # input shape : 256 * h/16 * w/16
        self.conv5 = nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)
        self.act5 = nn.LeakyReLU()
        # input shape : 256 * h/32 * w/32
        self.conv6 = nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)
        self.act6 = nn.LeakyReLU()
        # input shape : 256 * h/64 * w/64
        ndf = self.img_h//64
        self.fc = nn.Linear(in_features=256 * ndf * ndf,out_features=256)

    def forward(self,x):
         x = self.act1(self.conv1(x))
         x = self.act2(self.conv2(x))
         x = self.act3(self.conv3(x))
         x = self.act4(self.conv4(x))
         x = self.act5(self.conv5(x))
         x = self.act6(self.conv6(x))
         x = torch.flatten(x,1)
         x = self.fc(x)
         return x

class ContextDiscriminator(nn.Module):
    def __init__(self):
        super(ContextDiscriminator,self).__init__()
        infeature = 256 + 256 + 256 + 256
        local_shape = [3,64,64]
        global_shape = [3,128,128]

        self.model_ld = LocalDiscriminator(local_shape)
        self.model_gd = GlobalDiscriminator(global_shape)

        self.fc1 = nn.Linear(infeature,out_features=512)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=512,out_features=1)

    def forward(self,x,xl_left,xl_right,latent_left,latent_right):
        x_ld = self.model_ld(xl_left)
        x_rd = self.model_ld(xl_right)

        x_d = torch.concat((x_ld,x_rd),dim=1)
        x_gd = self.model_gd(x)
        out = torch.concat((x_d,x_gd,latent_right,latent_left),dim=1)

        out = self.act1(self.fc1(out))
        out = self.fc2(out)
        return out



