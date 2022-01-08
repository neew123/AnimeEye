import itertools
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from auto_encoder_networks import EncoderGenerator_Res , DecoderGenerator_image_Res

class AutoEncoderModel(nn.Module):
    """
    AutoEncoder
    this model will encode region to a vector
    """
    def __init__(self,):
        super(AutoEncoderModel, self).__init__()

        self.image_size = 128
        self.input_nc = 3
        self.latent_dim = 256
        # base
        self.device = torch.device('cuda:7')
        self.load_path = ''
        self.loss_names = []
        self.model_names = []
        self.schedulers = []
        self.optimizers = []

        self.face_part = {'bg': [0, 0, 128],
                          'left_eye': [54, 78, 32],
                          'right_eye': [128, 78, 32],
                          'nose': [91, 116, 16],
                          'mouth': [84, 150, 32]}

        # 5 encoders
        self.net_encoder_bg = EncoderGenerator_Res(image_size=self.face_part['bg'][2],
                                                   input_nc=self.input_nc,
                                                   latent_dim=self.latent_dim)
        self.net_encoder_bg.to(self.device)
        self.model_names += ['net_encoder_bg']

        self.net_encoder_left_eye = EncoderGenerator_Res(image_size=self.face_part['left_eye'][2],
                                                         input_nc=self.input_nc,
                                                         latent_dim=self.latent_dim)
        self.net_encoder_left_eye.to(self.device)
        self.model_names += ['net_encoder_left_eye']

        self.net_encoder_right_eye = EncoderGenerator_Res(image_size=self.face_part['right_eye'][2],
                                                          input_nc=self.input_nc,
                                                          latent_dim=self.latent_dim)
        self.net_encoder_right_eye.to(self.device)
        self.model_names += ['net_encoder_right_eye']

        self.net_encoder_nose = EncoderGenerator_Res(image_size=self.face_part['nose'][2],
                                                     input_nc=self.input_nc,
                                                     latent_dim=self.latent_dim)
        self.net_encoder_nose.to(self.device)
        self.model_names += ['net_encoder_nose']

        self.net_encoder_mouth = EncoderGenerator_Res(image_size=self.face_part['mouth'][2],
                                                      input_nc=self.input_nc,
                                                      latent_dim=self.latent_dim)
        self.net_encoder_mouth.to(self.device)
        self.model_names += ['net_encoder_mouth']

    def name(self):
        return "AutoEncoder"

    def set_input(self, left,right):
        # self.image = input['image'][0].to(self.device) # Batch * C * H * W
        self.left_eye = left.to(self.device)
        self.right_eye = right.to(self.device)
        # self.nose = input['nose'].to(self.device)
        # self.mouth = input['mouth'].to(self.device)
        # self.bg = input['image'].to(self.device)

    def forward(self):
        # self.latent_vector = self.net_encoder(self.image)
        # self.fake = self.net_decoder(self.latent_vector)
        # get latent
        self.bg_latent = self.net_encoder_bg(self.bg)
        self.left_eye_latent = self.net_encoder_left_eye(self.left_eye)
        self.right_eye_latent = self.net_encoder_right_eye(self.right_eye)
        self.nose_latent = self.net_encoder_nose(self.nose)
        self.mouth_latent = self.net_encoder_mouth(self.mouth)

        # decode
        self.bg_fake = self.net_decoder_bg(self.bg_latent)
        self.left_eye_fake = self.net_decoder_left_eye(self.left_eye_latent)
        self.right_eye_fake = self.net_decoder_right_eye(self.right_eye_latent)
        self.nose_fake = self.net_decoder_nose(self.nose_latent)
        self.mouth_fake = self.net_decoder_mouth(self.mouth_latent)

        self.fake = self.bg_fake.clone()
        #for key in self.face_part:
            #self.fake[:, :, self.face_part[key][0]:self.face_part[key][0] + self.face_part[key][2],
            #self.face_part[key][1]:self.face_part[key][1] + self.face_part[key][2]] = getattr(self, f'{key}_fake')

    def backward(self):
        bg_mse = F.mse_loss(self.bg, self.bg_fake)
        left_eye_mse = F.mse_loss(self.left_eye, self.left_eye_fake)
        right_eye_mse = F.mse_loss(self.right_eye, self.right_eye_fake)
        nose_mse = F.mse_loss(self.nose, self.nose_fake)
        mouth_mse = F.mse_loss(self.mouth, self.mouth_fake)
        self.loss_mse = bg_mse + left_eye_mse + right_eye_mse + nose_mse + mouth_mse
        self.loss = self.opt.mse_weight * self.loss_mse
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def set_test_input(self, input):
        self.image = input['image'].to(self.device)


    def get_latent(self, left,right):
        #self.image = input['image'][0].to(self.device)
        self.left_eye = left.to(self.device)
        self.right_eye = right.to(self.device)
        # self.nose = input['nose'].to(self.device)
        # self.mouth = input['mouth'].to(self.device)
        # self.bg = input['image'].to(self.device)

        return {
                'left_eye_latent': self.net_encoder_left_eye(self.left_eye),
                 'right_eye_latent': self.net_encoder_right_eye(self.right_eye)}
        # return {'bg_latent': self.net_encoder_bg(self.bg),
        #         'left_eye_latent': self.net_encoder_left_eye(self.left_eye),
        #         'right_eye_latent': self.net_encoder_right_eye(self.right_eye),
        #         'nose_latent': self.net_encoder_nose(self.nose),
        #         'mouth_latent': self.net_encoder_mouth(self.mouth)}

    def eval(self):
        """把本model内所有神经网络进入评估模式"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def test(self):
        """在no_grad下运行一次forward"""
        with torch.no_grad():
            self.forward()

    def load_networks(self, load_path):
        """加载模型
        @:param epoch (int) -- load的目标epoch数; 读取save_dir中的文件 '%s_%s.pth' % (name, epoch)
        """
        print("loading model with [%s]" % load_path)
        state_dict = torch.load(load_path, map_location=self.device)
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                net.load_state_dict(state_dict[name])

    def print_networks(self, verbose):
        """打印模型参数总数，if verbose then 打印模型架构
        @:param verbose (bool): 为True打印模型架构，False不打印
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] 参数总数 : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')