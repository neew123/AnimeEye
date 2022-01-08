from __future__ import print_function
from six.moves import range

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import os
import time

from PIL import Image, ImageFont, ImageDraw



from miscc.config import cfg
from miscc.utils import mkdir_p


from tensorboardX import summary
from tensorboardX import FileWriter

#from model import G_NET, D_NET64, D_NET128, D_NET256, D_NET512, D_NET1024, INCEPTION_V3
from generator import Generator
from discriminator import ContextDiscriminator
from auto_encoder_model import AutoEncoderModel
from loss import loss_hinge_dis,loss_hinge_gen,ReconstructionLoss


# ################## Shared functions ###################
def compute_mean_covariance(img):
    batch_size = img.size(0)
    channel_num = img.size(1)
    height = img.size(2)
    width = img.size(3)
    num_pixels = height * width

    # batch_size * channel_num * 1 * 1
    mu = img.mean(2, keepdim=True).mean(3, keepdim=True)

    # batch_size * channel_num * num_pixels
    img_hat = img - mu.expand_as(img)
    img_hat = img_hat.view(batch_size, channel_num, num_pixels)
    # batch_size * num_pixels * channel_num
    img_hat_transpose = img_hat.transpose(1, 2)
    # batch_size * channel_num * channel_num
    covariance = torch.bmm(img_hat, img_hat_transpose)
    covariance = covariance / num_pixels

    return mu, covariance

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)



def load_network(gpus):
    netG = Generator()
    netG.apply(weights_init)
    netG = torch.nn.DataParallel(netG, device_ids=gpus)
    # print(netG)
    netEncoder = AutoEncoderModel()
    netEncoder.load_networks('./AutoEncoder_490.pth')
    netEncoder.eval()
    #
    netD = ContextDiscriminator()
    netG.apply(weights_init)
    netD = torch.nn.DataParallel(netD, device_ids=gpus)

    count = 0
    if cfg.TRAIN.NET_G != '':
        state_dict = torch.load(cfg.TRAIN.NET_G)
        netG.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_G)
        istart = cfg.TRAIN.NET_G.rfind('_') + 1
        iend = cfg.TRAIN.NET_G.rfind('.')
        count = cfg.TRAIN.NET_G[istart:iend]
        count = int(count) + 1


    if cfg.TRAIN.NET_D != '':
        print('Load %s.pth' % cfg.TRAIN.NET_D)
        state_dict = torch.load('%s.pth' % cfg.TRAIN.NET_D)
        netD.load_state_dict(state_dict)


    # inception_model = INCEPTION_V3()

    if cfg.CUDA:
        netG.cuda()
        netEncoder.cuda()
        netD.cuda()

        # inception_model = inception_model.cuda()
    # inception_model.eval()
    # return netG, vgg ,netsD, len(netsD), inception_model, count
    return netG,count,netEncoder,netD


def define_optimizers(netG, netD):
    optimizerD = optim.Adam( netD.parameters(),
                            lr=cfg.TRAIN.DISCRIMINATOR_LR,
                            betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(),
                            lr=cfg.TRAIN.GENERATOR_LR,
                            betas=(0.5, 0.999))

    return optimizerG,optimizerD


def save_model(netG, netD,epoch, model_dir):
    torch.save(
        netG.state_dict(),
        '%s/netG_%d.pth' % (model_dir, epoch))
    torch.save(
        netD.state_dict(),
        '%s/netD_%d.pth' % (model_dir, epoch))
    print('Save G/Ds models.')


def save_img_results(imgs_tcpu, fake_imgs,
                     count, image_dir, summary_writer):
    num = cfg.TRAIN.VIS_COUNT
    # The range of real_img (i.e., self.imgs_tcpu[i][0:num])
    # is changed to [0, 1] by function vutils.save_image
    real_img = imgs_tcpu[-1][0:num]
    vutils.save_image(
        real_img, '%s/real_samples.png' % (image_dir),
        normalize=True)
    real_img_set = vutils.make_grid(real_img).numpy()
    real_img_set = np.transpose(real_img_set, (1, 2, 0))
    real_img_set = real_img_set * 255  #numpy.ndarray object
    real_img_set = real_img_set.astype(np.uint8) #(776,2066,3) (H,W,C)
    #print("real image set1", real_img_set.shape)
    #sup_real_img = summary.image('real_img', real_img_set)
    sup_real_img = summary.image('real_img', real_img_set,dataformats='HWC')
    #print("real image set2",real_img_set.shape)
    summary_writer.add_summary(sup_real_img, count)

    fake_img = fake_imgs[0:num]
    # The range of fake_img.data (i.e., self.fake_imgs[i][0:num])
    # is still [-1. 1]...
    vutils.save_image(
            fake_img.data, '%s/count_%09d_fake_samples.png' %
            (image_dir, count), normalize=True)

    fake_img_set = vutils.make_grid(fake_img.data).cpu().numpy()

    fake_img_set = np.transpose(fake_img_set, (1, 2, 0))
    fake_img_set = (fake_img_set + 1) * 255 / 2
    fake_img_set = fake_img_set.astype(np.uint8)

    sup_fake_img = summary.image('fake_img', fake_img_set,dataformats='HWC')
    summary_writer.add_summary(sup_fake_img, count)
    summary_writer.flush()

def gram_matrix(feature):
    (b, ch, h, w) = feature.size()
    t1 = feature.view(b, ch, w*h)
    t2 = t1.transpose(1, 2)
    gram = t1.bmm(t2) / (ch * w * h)
    return gram

def calc_vgg(fake_imgs,real_imgs,vgg):
    errG_vgg = 0.0
    _, _, _, fake_feature, _, _, _, _ = vgg(fake_imgs)
    _, _, _, real_feature, _, _, _, _ = vgg(real_imgs)
    fake_feature = gram_matrix(fake_feature)
    real_feature = gram_matrix(real_feature)
    #print("diff")

    errG_vgg += torch.norm(real_feature - fake_feature) * 1
    return errG_vgg



# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, imsize):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = FileWriter(self.log_dir)

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

    def crop_resize(self,img,pos):
        patch = img[:,pos[0]-16:pos[0]+16,pos[1]-16:pos[1]+16]
        return patch
    def get_mask(self , img, eye_pos):
        batch_mask = []
        batch_miss = []
        for i in range(self.batch_size):
            current_img = img[i]
            current_p = eye_pos[i]
            #
            current_img = current_img.numpy()  # C H W
            current_img = current_img.transpose((2, 0, 1)) # H W C
            #
            mask = np.zeros(shape=[128, 128, 3])
            # for left eye
            scale = current_p[0] - 16
            down_scale = current_p[0] + 16
            l1_1 = int(scale)
            u1_1 = int(down_scale)
            scale = current_p[1] - 16
            down_scale = current_p[1] + 16
            l1_2 = int(scale)
            u1_2 = int(down_scale)
            mask[l1_1:u1_1, l1_2:u1_2, :] = 1.0
            # for right eye
            scale = current_p[2] - 16
            down_scale = current_p[2] + 16
            l2_1 = int(scale)
            u2_1 = int(down_scale)
            scale = current_p[3] - 16
            down_scale = current_p[3] + 16
            l2_2 = int(scale)
            u2_2 = int(down_scale)
            mask[l2_1:u2_1, l2_2:u2_2, :] = 1.0
            img_miss = current_img * (1 - mask) # H W C
            img_miss = img_miss.transpose((2,0,1))
            batch_mask.append(mask)
            batch_miss.append(img_miss)
        batch_mask = torch.tensor(batch_mask)
        batch_miss = torch.tensor(batch_miss)
        return batch_mask,batch_miss
    def prepare_data(self, data):
        img, eye_pos  = data
        N,C,H,W = img.shape
        xm,xc = self.get_mask(img,eye_pos)
        if cfg.CUDA:
            x = Variable(img).cuda()
            xm = Variable(xm).cuda()
            xc = Variable(xc).cuda()
        return x,xm,xc

    def train_Dnet(self, count):
        flag = count % 100

        # for real img
        label_real_right = self.netD(self.x,self.xl_left,self.xl_right,self.xl_left_fp,self.xl_right_fp)
        # for fake img
        label_fake_right = self.netD(self.y,self.yl_left,self.yl_right,self.yl_left_fp,self.yl_right_fp)

        netD.zero_grad()

        errD = loss_higen_dis(label_real_right,label_fake_right)

        errD.backward()
        self.optimizerD.step()
        # log
        if flag == 0:
            #summary_D = summary.scalar('D_loss%d' % idx, errD.data[0])
            summary_D = summary.scalar('D_right_loss%d' , errD.data.item())
            self.summary_writer.add_summary(summary_D, count)
        return errD



    def train_Gnet(self, count):
        self.netG.zero_grad()
        flag = count % 100
        # adversarial loss
        label_fake_right = self.netD(self.y,self.yl_left,self.yl_right,self.yl_left_fp,self.yl_right_fp)
        errG = loss_hinge_gen(label_fake_right)

        # recon loss: lam_r
        recon_loss = cfg.TRAIN.Lambda_recon * ReconstructionLoss(self.y,self.x)
        # percep loss: lam_p
        percep_loss = self.percep_loss(self.xl_left_fp,self.yl_left_fp)+self.percep_loss(self.xl_right_fp,self.yl_right_fp)
        percep_loss = cfg.TRAIN.Lambda_recon * percep_loss
        #
        errG_total = errG + recon_loss+ percep_loss
        ######################

        ######################
        if flag == 0:
            # summary_D = summary.scalar('G_loss%d' % i, errG.data[0])
            summary_D = summary.scalar('G_loss%d' , errG_total.data.item())
            self.summary_writer.add_summary(summary_D, count)
        errG_total.backward()
        self.optimizerG.step()
        return errG_total

    def train(self):
        self.netG,start_count,self.netEncoder,self.netD= load_network(self.gpus)
        #self.optimizerG, self.optimizersD = define_optimizers(self.netG, self.netsD)
        self.optimizerG,self.optimizerD= define_optimizers(self.netG,self.netD)
        # loss
        self.criterion = nn.BCELoss()
        self.recon_loss = ReconstructionLoss()
        self.percep_loss = nn.L1Loss()


        if cfg.CUDA:
            self.criterion.cuda()
            self.recon_loss.cuda()
            self.percep_loss.cuda()

        count = start_count
        start_epoch = start_count // (self.num_batches)
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            for step, data in enumerate(self.data_loader,0):
                #######################################################
                # (0) Prepare training data
                ######################################################
                self.x, self.xm,self.xc= self.prepare_data(data)

                img, eye_pos = data
                self.x_left_p = eye_pos[:,0:2]
                self.x_right_p = eye_pos[:,2:4]
                self.xl_right = self.crop_resize(self.x,self.x_right_p)
                self.xl_left = self.crop_resize(self.x,self.x_left_p)
                if cfg.CUDA:
                    self.xl_right = Variable(self.xl_right).cuda()
                    self.xl_left = Variable(self.xl_left).cuda()
                    self.x_left_p = Variable(self.x_left_p).cuda()
                    self.x_right_p = Variable(self.x_right_p).cuda()

                #######################################################
                # (1) Generate fake images
                ######################################################
                self.netEncoder.set_input(self.xl_left,self.xl_right)
                latent_dict = self.netEncoder.get_latent(self.xl_left,self.xl_right)
                self.xl_left_fp, self.xl_right_fp = latent_dict['left_eye_latent'],latent_dict['right_eye_latent']
                #self.xl_left_fp ,self.xl_right_fp= self.netEncoder(self.xl_left,self.xl_right)
                self.yo = self.netG(self.xc,self.xm,self.xl_left_fp,self.xl_right_fp)

                self.yl_left = self.crop_resize(self.yo,self.x_left_p)
                self.yl_right = self.crop_resize(self.yo,self.x_right_p)
                #self.yl_left_fp ,self.yl_right_fp = self.netEncoder(self.yl_left,self.yl_right)
                self.netEncoder.set_input(self.yl_left,self.yl_right)
                latent_dict = self.netEncoder.get_latent(self.xl_left,self.xl_right)
                self.yl_left_fp, self.yl_right_fp = latent_dict['left_eye_latent'],latent_dict['right_eye_latent']

                self.y = self.xc + self.yo * self.xm
                #######################################################
                # (2) Update D network
                ######################################################
                errD_total = self.train_Dnet_right(count)
                #######################################################
                # (3) Update G network: maximize log(D(G(z)))
                ######################################################
                errG_total = self.train_Gnet(count)

                if count % 100 == 0:
                    summary_D = summary.scalar('D_loss', errD_total.data.item())
                    summary_G = summary.scalar('G_loss', errG_total.data.item())
                    #summary_KL = summary.scalar('KL_loss', kl_loss.data.item())
                    self.summary_writer.add_summary(summary_D, count)
                    self.summary_writer.add_summary(summary_G, count)
                    #self.summary_writer.add_summary(summary_KL, count)

                count = count + 1

                if count % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                    save_model(self.netG,self.netD, count, self.model_dir)
                    # Save images
                    self.yo = self.netG(self.xc, self.xm, self.xl_left_fp, self.xl_right_fp)
                    self.fake_imgs = self.xc + self.yo * self.xm

                    save_img_results(self.imgs_tcpu, self.fake_imgs,count,self.image_dir, self.summary_writer)

            end_t = time.time()
            print('''[%d/%d][%d]
                         Loss_D: %.2f  Loss_G: %.2f Time: %.2fs
                      '''  # D(real): %.4f D(wrong):%.4f  D(fake) %.4f
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.data.item(), errG_total.data.item(), end_t - start_t))

        # save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
        save_model(self.netG, self.netD, count, self.model_dir)
        self.summary_writer.close()

    def save_superimages(self, images_list, filenames,
                         save_dir, split_dir, imsize):
        batch_size = images_list[0].size(0)
        num_sentences = len(images_list)
        for i in range(batch_size):
            s_tmp = '%s/super/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            #
            savename = '%s_%d.png' % (s_tmp, imsize)
            super_img = []
            for j in range(num_sentences):
                img = images_list[j][i]
                # print(img.size())
                img = img.view(1, 3, imsize, imsize)
                # print(img.size())
                super_img.append(img)
                # break
            super_img = torch.cat(super_img, 0)
            vutils.save_image(super_img, savename, nrow=10, normalize=True)

    def save_singleimages(self, images, filenames,
                          save_dir, split_dir, sentenceID, imsize):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames)
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d_sentence%d.png' % (s_tmp, imsize, sentenceID)
            # range from [-1, 1] to [0, 255]
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def evaluate(self, split_dir):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            if split_dir == 'eval':
                split_dir = 'eval'
            netG = Generator()
            netG.apply(weights_init)
            netG = torch.nn.DataParallel(netG, device_ids=self.gpus)
            print(netG)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            state_dict = \
                torch.load(cfg.TRAIN.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load ', cfg.TRAIN.NET_G)
            # load Encoder

            # the path to save generated images
            s_tmp = cfg.TRAIN.NET_G
            istart = s_tmp.rfind('_') + 1
            iend = s_tmp.rfind('.')
            iteration = int(s_tmp[istart:iend])
            s_tmp = s_tmp[:s_tmp.rfind('/')]
            save_dir = '%s/iteration%d' % (s_tmp, iteration)

            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(1 , nz))
            if cfg.CUDA:
                netG.cuda()
                noise = noise.cuda()

            noise.data.resize_(1, nz)
            noise.data.normal_(0, 1)
            tag = [
               ['red eyes', 'white hair', 'long hair'],
                ['red eyes', 'silver hair', 'long hair'],
                ['red eyes', 'gray hair', 'long hair'],
                ['red eyes', 'black hair', 'long hair'],
                # ['yellow eyes', 'black hair', 'long hair'],
                # ['pink eyes', 'green hair', 'short hair'],
                # ['black eyes', 'red hair', 'long hair'],
                # ['green eyes', 'blue hair', 'short hair'],
                ]
            '''
            inter_tag = [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

                [0.0, 0.2, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.3, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.3, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.6, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.4, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.3, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.8, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
            '''
            # switch to evaluate mode
            netG.eval()
            # i = 0
            # for tags in tag:
            #    i += 1
            for i in range(0,100):
                noise.data.resize_(1, nz)
                noise.data.normal_(0, 1)
                #tags = ['brown hair','black eyes']
                #tags = ['glasses']
                #tags = ['long hair']
                tags = ['blue eyes','black hair','glasses']
                # tags = ['blue eyes', 'purple hair', 'glasses']
                #print(utils.get_one_hot(source_tag)) #ndarray
                #t_embeddings = Variable(torch.FloatTensor([tags])) #for interploate
                t_embeddings = Variable(torch.FloatTensor([utils.get_one_hot(tags)]))
                # t_embeddings = Variable(torch.FloatTensor(tags))
                if cfg.CUDA:
                    t_embeddings = Variable(t_embeddings).cuda()
                else:
                    t_embeddings = Variable(t_embeddings)

                # noise.data.resize_(1, nz)
                # noise.data.normal_(0, 1)
                fake_img_list = []
                # print("noise shape",noise.shape)
                # print("tags shape",t_embeddings.shape)
                fake_imgs = netG(noise, t_embeddings)
                if cfg.TEST.B_EXAMPLE:
                        # fake_img_list.append(fake_imgs[0].data.cpu())
                        # fake_img_list.append(fake_imgs[1].data.cpu())
                    fake_img_list.append(fake_imgs[1].data.cpu())
                else:

                        #self.save_singleimages(fake_imgs[-1], filenames,
                                               #save_dir, split_dir, i, 256)
                    filenames = 'test'
                    self.save_singleimages(fake_imgs[-1], filenames,
                                            save_dir, split_dir, i, 128)
                        # self.save_singleimages(fake_imgs[-3], filenames,
                        #                        save_dir, split_dir, i, 64)
                    # break
                if cfg.TEST.B_EXAMPLE:
                    # self.save_superimages(fake_img_list, filenames,
                    #                       save_dir, split_dir, 64)
                     self.save_superimafake_one_hotges(fake_img_list, filenames,
                                           save_dir, split_dir, 128)
                    #self.save_superimages(fake_img_list, filenames,
                                          #save_dir, split_dir, 256)