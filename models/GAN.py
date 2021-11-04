
import os
import datetime
import time
import timeit
import copy
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

import models.Losses as Losses
from data import get_image_batches_test_random, get_data_loader, get_image_batches, get_image_batches_test
from models import update_average
from models.Blocks import GResiduleBlock, PerceptualLoss, Vgg_face_dag, load_lightCNN, Resnet50_ferplus_dag, DiscriminatorTop, DiscriminatorBlock, InputBlock, GSynthesisBlock
from models.CustomLayers import EqualizedConv2d, PixelNormLayer, EqualizedLinear, Truncation
import pdb
import scipy.io as sio

import torch.nn.functional as F
from torch.autograd import Variable
# import torch.nn as nn
from torchvision.utils import save_image

from pytorch_msssim import ssim

bs =8
styleDim=2048
num_ref = 3



class x2gen(nn.Module):
    def __init__(self):
        super().__init__()

        # 3 16 16 -> 32 16 16 -> 32 32 32 -> 3 32 32
        use_wscale = True
        gain = np.sqrt(2)
        self.conv_0 = EqualizedConv2d(3, 32, 3, gain=gain, use_wscale=use_wscale)
        self.conv_1 = EqualizedConv2d(32, 32, 3, gain=gain, use_wscale=use_wscale, upscale=True)
        self.conv_2 = EqualizedConv2d(32, 3, 3, gain=gain, use_wscale=use_wscale)

        self.act = nn.ReLU()
        self.tahn = nn.Tanh()


    def forward(self, x):
        x = self.act(self.conv_0(x))
        x = self.act(self.conv_1(x))
        y = self.tahn(self.conv_2(x))

        return y


class qnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_1 = nn.Conv2d(64+3, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.feature_3 = nn.Conv2d(256, 1, kernel_size=1, stride=1)
        self.feature_4 = nn.Sigmoid()

    def forward(self, data, gt, train=True):

        if train is True:
            data = data.view(bs, num_ref, 64, 16,16)
            for i in range(bs):

                feature_1 = self.feature_1(torch.cat((data[i], gt[i].unsqueeze(0).repeat(num_ref,1,1,1)), dim=1))
                feature_3 = self.feature_3(self.relu(feature_1))
                w_ = self.feature_4(feature_3)
                
                w = w_ / torch.sum(w_, dim=0)

                if i == 0:
                    z = torch.sum(data[i] * w, dim=0).unsqueeze(0)
                    ww = w.unsqueeze(0)
                else:
                    z = torch.cat((z, torch.sum(data[i] * w, dim=0).unsqueeze(0)),dim=0)
                    ww = torch.cat((ww, w.unsqueeze(0)),0)
        else:
            feature_1 = self.feature_1(torch.cat((data, gt.repeat(num_ref,1,1,1)), dim=1))
            feature_3 = self.feature_3(self.relu(feature_1))
            w_ = self.feature_4(feature_3)

            w = w_ / torch.sum(w_, dim=0)

            z = torch.sum(data * w, dim=0).unsqueeze(0)

            ww = w


        return z, ww



class qnn_32(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_1 = nn.Conv2d(128+3, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.feature_3 = nn.Conv2d(256, 1, kernel_size=1, stride=1)
        self.feature_4 = nn.Sigmoid()

    def forward(self, data, gt, train=True):

        if train is True:
            data = data.view(bs, num_ref, 128, 32,32)
            for i in range(bs):

                feature_1 = self.feature_1(torch.cat((data[i], gt[i].unsqueeze(0).repeat(num_ref,1,1,1)), dim=1))
                feature_3 = self.feature_3(self.relu(feature_1))
                w_ = self.feature_4(feature_3)
                
                w = w_ / torch.sum(w_, dim=0)

                if i == 0:
                    z = torch.sum(data[i] * w, dim=0).unsqueeze(0)
                    ww = w.unsqueeze(0)
                else:
                    z = torch.cat((z, torch.sum(data[i] * w, dim=0).unsqueeze(0)),dim=0)
                    ww = torch.cat((ww, w.unsqueeze(0)),0)
        else:
            feature_1 = self.feature_1(torch.cat((data, gt.repeat(num_ref,1,1,1)), dim=1))
            feature_3 = self.feature_3(self.relu(feature_1))  
            w_ = self.feature_4(feature_3)

            w = w_ / torch.sum(w_, dim=0)

            z = torch.sum(data * w, dim=0).unsqueeze(0)

            ww = w
            # print(w_)            
        # print(ww)

        return z, ww



class Ref_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        use_wscale = True
        self.down_0 = EqualizedConv2d(3, 64, 3, gain=1, use_wscale=use_wscale)
        self.down_1 = EqualizedConv2d(64, 128, 3, gain=1, use_wscale=use_wscale)
        self.down_2 = EqualizedConv2d(128, 256, 3, gain=1, use_wscale=use_wscale, downscale=True)
        self.down_3 = EqualizedConv2d(256, 128, 3, gain=1, use_wscale=use_wscale,downscale=True)
        self.down_4 = EqualizedConv2d(128, 64, 3, gain=1, use_wscale=use_wscale,downscale=True)

        self.act = nn.ReLU()
        # self.qnn = qnn

    def forward(self, x):
        # print(x.shape)
        x = self.act(self.down_0(x))
        # print(x.shape)
        x = self.act(self.down_1(x))
        # print(x.shape)
        x = self.act(self.down_2(x))
        # print(x.shape)
        y1 = self.act(self.down_3(x))
        # print(y1.shape)
        y2 = self.act(self.down_4(y1)) 
        # print(y2.shape) 

        # pdb.set_trace()
        return y2, y1




class GSynthesis(nn.Module):

    def __init__(self, num_channels=3, resolution=1024,
                 fmap_base=8192, fmap_decay=1.0, fmap_max=512,
                 use_wscale=True, blur_filter=None,
                  **kwargs):


        super().__init__()

        # pdb.set_trace()

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self._channels = [64, 128, 256, 128]

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.depth = resolution_log2 - 1
        gain = np.sqrt(2)

        # Early layers.
        self.init_block = InputBlock(nf(1), gain, use_wscale)
        # create the ToRGB layers for various outputs
        rgb_converters = [EqualizedConv2d(nf(1), num_channels, 1, gain=1, use_wscale=use_wscale)]

        # Building blocks for remaining layers.
        blocks = []
        blocks.append(GSynthesisBlock(self._channels[0]*2, self._channels[1], blur_filter, gain, use_wscale,
                                          ))
        blocks.append(GSynthesisBlock(self._channels[1]*2, self._channels[2], blur_filter, gain, use_wscale,
                                          ))

        for res in range(2, len(self._channels)-1):

            last_channels = self._channels[res]
            channels = self._channels[res+1]

            blocks.append(GSynthesisBlock(last_channels, channels, blur_filter, gain, use_wscale,
                                          ))
            rgb_converters.append(EqualizedConv2d(64, num_channels, 1, gain=1, use_wscale=use_wscale))

        blocks.append(GResiduleBlock(128, 64, gain, use_wscale,
                                          ))        
        self.blocks = nn.ModuleList(blocks)
        self.to_rgb = nn.ModuleList(rgb_converters)


    def forward(self, im_lr, ref_fmap_16, ref_fmap_32):


        x = self.init_block(im_lr)

        x = torch.cat((x, ref_fmap_16),1)
        for i, block in enumerate(self.blocks):
            if i == 1:
                x = torch.cat((x, ref_fmap_32),1)
                x = block(x)
            else:

                x = block(x)

        images_out = nn.Tanh()(self.to_rgb[-1](x))
  


        return images_out


class Generator(nn.Module):

    def __init__(self, resolution, **kwargs):

        super(Generator, self).__init__()

        self.qnn = qnn()
        self.qnn_32 = qnn_32() 

        # Setup components.
        self.num_layers = (int(np.log2(resolution)) - 1) * 2
        self.g_synthesis = GSynthesis(resolution=resolution, dlatent_size=styleDim, **kwargs)
        self.encoder = Ref_encoder()

    def forward(self, im_lr, im_lr_32, im_hr, labels_in=None, train=True):

        ref_fmap_16,  ref_fmap_32 = self.encoder(im_hr)
        ref_fmap_16, w_16 = self.qnn(ref_fmap_16, im_lr,train)
        ref_fmap_32, w_32 = self.qnn_32(ref_fmap_32, im_lr_32,train)



        fake_images = self.g_synthesis(im_lr, ref_fmap_16, ref_fmap_32)
        return fake_images, w_16, w_32


class Discriminator(nn.Module):

    def __init__(self, resolution, num_channels=3, fmap_base=8192, fmap_decay=1.0, fmap_max=512,
                 nonlinearity='lrelu', use_wscale=True, mbstd_group_size=4, mbstd_num_features=1,
                 blur_filter=None, **kwargs):

        super(Discriminator, self).__init__()
        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.mbstd_num_features = mbstd_num_features
        self.mbstd_group_size = mbstd_group_size

        # if blur_filter is None:
        #     blur_filter = [1, 2, 1]

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.depth = resolution_log2 - 1

        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]

        # create the remaining layers
        blocks = []
        blocks_2 = []
        from_rgb = []
        from_rgb_2 = []
        # pdb.set_trace()

        for res in range(resolution_log2, 2, -1):
            # name = '{s}x{s}'.format(s=2 ** res)
            blocks.append(DiscriminatorBlock(nf(res - 1), nf(res - 2),
                                             gain=gain, use_wscale=use_wscale, activation_layer=act,
                                             blur_kernel=blur_filter))
            # create the fromRGB layers for various inputs:
            from_rgb.append(EqualizedConv2d(num_channels, nf(res - 1), kernel_size=1,
                                            gain=gain, use_wscale=use_wscale))
        self.blocks = nn.ModuleList(blocks)


        self.final_block = DiscriminatorTop(self.mbstd_group_size, self.mbstd_num_features,
                                            in_channels=nf(2), intermediate_channels=nf(2),
                                            gain=gain, use_wscale=use_wscale, activation_layer=act)
        from_rgb.append(EqualizedConv2d(num_channels, nf(2), kernel_size=1,
                                        gain=gain, use_wscale=use_wscale))
        self.from_rgb = nn.ModuleList(from_rgb)

        # register the temporary downSampler
        self.temporaryDownsampler = nn.AvgPool2d(2)

    def forward(self, images_in, labels_in=None):


        # assert depth < self.depth, "Requested output depth cannot be produced"

        x = self.from_rgb[0](images_in)
        for i, block in enumerate(self.blocks):
            x = block(x)

        scores_out = self.final_block(x)
        

        return scores_out

def faceNN_preprocess(img):
    # pdb.set_trace()
    tensortype = type(img.data)
    if len(img.shape) == 3:
        img = F.interpolate(img.unsqueeze(0), size=(224,224), mode='bicubic')
    else:
        img = F.interpolate(img, size=(224,224), mode='bicubic')
    img = (img + 1)*127.5
    mean = tensortype(img.data.size())
    mean[:, 0, :, :] = 131.0912
    mean[:, 1, :, :] = 103.8827
    mean[:, 2, :, :] = 91.4953

    # mean[:, 0, :, :] = 129.186279296875
    # mean[:, 1, :, :] = 104.76238250732422
    # mean[:, 2, :, :] = 93.59396362304688


    img = img.sub(Variable(mean.cuda()))
    return img




def recon_criterion(predict, target):
    # pdb.set_trace()
    return torch.mean(torch.abs(predict - target))

class SRGAN:

    def __init__(self, resolution, num_channels,
                 g_args, d_args, g_opt_args, d_opt_args,
                 d_repeats=1, device=torch.device("cpu")):

        # state of the object
        self.depth = int(np.log2(resolution)) - 1
        self.device = device
        self.d_repeats = d_repeats



        self.faceNN = Resnet50_ferplus_dag()
        state_dict = torch.load('./resnet50_ferplus_dag.pth')

        # self.faceNN = Vgg_face_dag()


        self.faceNN.load_state_dict(state_dict)
        for p in self.faceNN.parameters():
            p.requires_grad = False
        self.faceNN.eval()
        self.faceNN = self.faceNN.cuda()

        self.PerceptualLoss = PerceptualLoss()



        # Create the Generator and the Discriminator
        self.gen = Generator(num_channels=num_channels,
                             resolution=resolution,
                             **g_args).to(self.device)
        self.dis = Discriminator(num_channels=num_channels,
                                 resolution=resolution,
                                 **d_args).to(self.device)
        self.x2gen = x2gen().cuda()

 
        # pdb.set_trace()
        # define the optimizers for the discriminator and generator
        self.__setup_gen_optim(**g_opt_args)
        self.__setup_dis_optim(**d_opt_args)
        self.__setup_x2gen_optim(**g_opt_args)

        # define the loss function used for training the GAN
        self.loss = Losses.LogisticGAN(self.dis)

        # # Use of ema
        # if self.use_ema:
        #     # create a shadow copy of the generator
        #     self.gen_shadow = copy.deepcopy(self.gen)
        #     # updater function:
        #     self.ema_updater = update_average
        #     # initialize the gen_shadow weights equal to the weights of gen
        #     self.ema_updater(self.gen_shadow, self.gen, beta=0)


    def __setup_gen_optim(self, learning_rate, beta_1, beta_2, eps):
        # pdb.set_trace()
        # self.gen_optim = torch.optim.Adam(self.gen.parameters(), lr=learning_rate, betas=(beta_1, beta_2), eps=eps)
        self.gen_optim = torch.optim.Adam([
            {'params': self.gen.g_synthesis.parameters()},
            {'params': self.gen.encoder.parameters()},
            {'params': self.gen.qnn.parameters(),'lr':1e-4},
            {'params': self.gen.qnn_32.parameters(),'lr':1e-4},
            ], lr=learning_rate, betas=(beta_1, beta_2), eps=eps)

    def __setup_dis_optim(self, learning_rate, beta_1, beta_2, eps):
        self.dis_optim = torch.optim.Adam(self.dis.parameters(), lr=learning_rate, betas=(beta_1, beta_2), eps=eps)

    def __setup_x2gen_optim(self, learning_rate, beta_1, beta_2, eps):
        self.x2gen_optim = torch.optim.Adam(self.x2gen.parameters(), lr=learning_rate, betas=(beta_1, beta_2), eps=eps)






    def optimize_discriminator(self, real_samples, loss_weight):

        loss_val = 0
        # pdb.set_trace()

        idx_lr = np.arange(0,real_samples.shape[0],num_ref+1)
        idx_ref = np.setdiff1d(np.arange(real_samples.shape[0]),idx_lr)
        im_lr = F.interpolate(real_samples[idx_lr],size=(16,16),mode='bicubic')
        im_lr_32 = F.interpolate(im_lr,size=(32,32),mode='bicubic')

        im_32_down = F.interpolate(real_samples[idx_lr], size=(32,32),mode='bicubic')


        # pdb.set_trace()
        # for _ in range(self.d_repeats):
            # generate a batch of samples
            # pdb.set_trace()
        im_32_gen = self.x2gen(im_lr)
        im_32_gen = im_32_gen.detach()

        if torch.mean(torch.abs(im_32_gen - im_32_down)) < torch.mean(torch.abs(im_lr_32 - im_32_down)):
            # print('=====Good=====')
            fake_samples, _,_ = self.gen(im_lr, im_32_gen, real_samples[idx_ref])
        else:
            fake_samples, _,_ = self.gen(im_lr, im_lr_32, real_samples[idx_ref])
              # pdb.set_trace()
        fake_samples = fake_samples.detach()

        loss = loss_weight['w_adv']*self.loss.dis_loss(real_samples[idx_lr], fake_samples)


        self.dis_optim.zero_grad()
        loss.backward()
        self.dis_optim.step()

        loss_val += loss.item()

        return loss_val 

    def optimize_generator(self, real_samples,loss_weight):

        idx_lr = np.arange(0,real_samples.shape[0],num_ref+1)
        idx_ref = np.setdiff1d(np.arange(real_samples.shape[0]),idx_lr)

        im_lr = F.interpolate(real_samples[idx_lr],size=(16,16),mode='bicubic')
        im_lr_32 = F.interpolate(im_lr, size=(32,32),mode='bicubic')
        im_32_down = F.interpolate(real_samples[idx_lr], size=(32,32),mode='bicubic')

        im_32_gen = self.x2gen(im_lr)
        loss_32 = 10*(torch.mean(torch.abs(im_32_gen - im_32_down))) + loss_weight['w_id']*torch.mean(self.PerceptualLoss.forward(im_32_gen, im_32_down))

        self.x2gen_optim.zero_grad()
        loss_32.backward()
        nn.utils.clip_grad_norm_(self.x2gen.parameters(), max_norm=10.)
        self.x2gen_optim.step()

        im_32_gen = im_32_gen.detach()


        print('====={}-{}====='.format(torch.mean(torch.abs(im_32_gen - im_32_down)).item(), torch.mean(torch.abs(im_lr_32 - im_32_down)).item()))
        

        if torch.mean(torch.abs(im_32_gen - im_32_down)) < torch.mean(torch.abs(im_lr_32 - im_32_down)):
            print('=====Good=====')
            fake_samples, ws,_ = self.gen(im_lr, im_32_gen, real_samples[idx_ref])
        else:
            fake_samples, ws,_ = self.gen(im_lr, im_lr_32, real_samples[idx_ref])

        loss_adv = self.loss.gen_loss(real_samples[idx_lr], fake_samples)
        loss_lr_recons = torch.mean(torch.abs((real_samples[idx_lr] - fake_samples))) 
        # + torch.mean((real_samples[idx_lr] - fake_samples)**2))/2.

        loss = loss_weight['w_lrRecon']*loss_lr_recons + loss_weight['w_adv']*loss_adv

        # pdb.set_trace()
        # ref_feature = self.faceNN(faceNN_preprocess(real_samples[idx_ref]))
        # ref_feature = ref_feature.view((bs, int(real_samples[idx_ref].shape[0]/bs), ref_feature.shape[1]))
        # for ww in range(bs):
        #     if ww == 0:
        #         # pdb.set_trace()
        #         ref_feature_ = torch.transpose(torch.mm(torch.transpose(ref_feature[ww],0,1), ws[ww]), 0,1)
        #     else:
        #         ref_feature_ = torch.cat((ref_feature_, torch.transpose(torch.mm(torch.transpose(ref_feature[ww],0,1), ws[ww]), 0,1)),0)
        # ref_feature = torch.mean(ref_feature, dim=1)
        # pdb.set_trace()

        ref_feature = self.faceNN(faceNN_preprocess(real_samples[idx_lr]))
        ref_feature = ref_feature.view(ref_feature.shape[0],ref_feature.shape[1])
        fake_feature = self.faceNN(faceNN_preprocess(fake_samples))
        fake_feature = fake_feature.view(fake_feature.shape[0],fake_feature.shape[1])
        loss_identity = recon_criterion(ref_feature, fake_feature)
        loss_identity = loss_identity + torch.mean(self.PerceptualLoss.forward(real_samples[idx_lr], fake_samples))
        # pdb.set_trace()
        loss = loss+loss_weight['w_id']*loss_identity

        # pdb.set_trace()

        loss_ssim = 1-ssim( (real_samples[idx_lr]+1)*127.5, (fake_samples+1)*127.5, data_range=255, size_average=True)
        # loss += loss_weight['w_ssim']*loss_ssim
        # optimize the generator
        self.gen_optim.zero_grad()
        loss.backward()
        # Gradient Clipping
        nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=10.)
        self.gen_optim.step()

        # return the loss value
        return loss.item(), loss_weight['w_adv']*loss_adv.item(), loss_weight['w_lrRecon']*loss_lr_recons.item(), loss_weight['w_id']*loss_identity.item(), loss_32.item(), 1-loss_ssim.item()

    @staticmethod
    def create_grid(samples, scale_factor, img_file):
        """
        utility function to create a grid of GAN samples

        :param samples: generated samples for storing
        :param scale_factor: factor for upscaling the image
        :param img_file: name of file to write
        :return: None (saves a file)
        """
        from torchvision.utils import save_image
        from torch.nn.functional import interpolate

        # upsample the image
        if scale_factor > 1:
            samples = interpolate(samples, scale_factor=scale_factor)

        # save the images:
        # save_image(samples, img_file, nrow=int(np.sqrt(len(samples))),
        #            normalize=True, scale_each=True, pad_value=128, padding=1)
        save_image(samples, img_file, nrow=num_ref+1+2,
                   normalize=True, scale_each=True, pad_value=128, padding=1)

    def train(self, dataset, num_workers, epochs, batch_sizes, logger, output,loss_weight,
              num_samples=36, feedback_factor=100, checkpoint_factor=1, start_epoch=1, end_epoch=20):

        # pdb.set_trace()
        label = sio.loadmat('./data/label_celeba_128.mat')
        label_all = label['label_all'][0]
        label_train = label['label_train'][0]
        label_test = label['label_test'][0]

        unique_train_idx, counts_train = np.unique(label_train, return_counts=True)
        unique_test_idx,  counts_test = np.unique(label_test, return_counts=True)



        thresh = num_ref+1 #---th=3 === 3564 classes === 26395 images
        thresh_2 = num_ref+1
        useful_labels_train = unique_train_idx[np.where(counts_train >= thresh)[0]]
        useful_labels_test = unique_test_idx[np.where(counts_test >= thresh)[0]]


        # pdb.set_trace()

        # useful_labels_test = useful_labels[2500:]


        # turn the generator and discriminator into train mode
        self.gen.train()
        self.dis.train()
        # if self.use_ema:
        #     self.gen_shadow.train()

        # create a global time counter
        global_time = time.time()


        # config depend on structure
        logger.info("Starting the training process ... \n")

        step = 1  # counter for number of iterations
        # pdb.set_trace()
        
        data = dataset
        for epoch in range(start_epoch, end_epochs):
            start = timeit.default_timer()  # record time at the start of epoch

            logger.info("Epoch: [%d]" % epoch)
                # total_batches = len(iter(data))
                # total_batches = counts[useful_labels_train].sum()
                # pdb.set_trace()
            t = 1000 * time.time() # current time in milliseconds
            np.random.seed(int(t) % 2**32)  
            total_batches = int(counts_train[np.where(counts_train >= thresh)[0]].sum()/bs)
            idx_r = np.random.permutation(counts_train[np.where(counts_train >= thresh)[0]].sum())[0:total_batches*bs]

                # pdb.set_trace()
            for i in range(total_batches):
                t = 1000 * time.time() # current time in milliseconds
                np.random.seed(int(t) % 2**32)  

                batch = get_image_batches(data, label_all, useful_labels_train,i, thresh_2, bs,idx_r)



                # extract current batch of data for training
                images = batch.to(self.device)

                # optimize the discriminator:
                dis_loss = self.optimize_discriminator(images,loss_weight)

                # optimize the generator:
                gen_loss, adv_loss, lr_rcons_loss, id_loss, x2_loss, ssim_loss = self.optimize_generator(images,loss_weight)

                # provide a loss feedback
                # pdb.set_trace()
                elapsed = time.time() - global_time
                elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]
                logger.info(
                    "Elapsed: [%s] Step: %d  Batch: %d  D_Loss: %f | G_Loss: %f adv_Loss: %f lrRcons_Loss: %f ID_Loss: %f S_Loss: %f| x2_Loss: %f"
                    % (elapsed, step, i, dis_loss, gen_loss, adv_loss, lr_rcons_loss, id_loss, ssim_loss,x2_loss))
                
                # pdb.set_trace()

                if i % int(total_batches / feedback_factor + 1) == 0 or i == 1:
                    self.get_scores(dataset, 0, logger)
                    # create a grid of samples and save it
                    os.makedirs(os.path.join(output, 'samples'), exist_ok=True)
                    gen_img_file = os.path.join(output, 'samples', "gen_" + str(current_depth)
                                                + "_" + str(epoch) + "_" + str(i) + ".png")

                    t = 1000 * time.time() # current time in milliseconds
                    np.random.seed(int(t) % 2**32)                    
      
                    save_dir = os.path.join(output, 'models')
                    os.makedirs(save_dir, exist_ok=True)
                    gen_save_file = os.path.join(save_dir, "GAN_GEN_latest" + "_" + str(epoch) + ".pth")
                    x2gen_save_file = os.path.join(save_dir, "GAN_x2GEN_latest" + "_" + str(epoch)+ ".pth")
                    dis_save_file = os.path.join(save_dir, "GAN_DIS_latest" + "_" + str(epoch) + ".pth")
                    gen_optim_save_file = os.path.join(
                            save_dir, "GAN_GEN_OPTIM_latest" + "_" + str(epoch)+ ".pth")
                    dis_optim_save_file = os.path.join(
                            save_dir, "GAN_DIS_OPTIM_latest" + "_" + str(epoch)+ ".pth")
                    x2gen_optim_save_file = os.path.join(
                            save_dir, "GAN_x2GEN_OPTIM_latest" + "_" + str(epoch)+ ".pth")


                    torch.save(self.gen.state_dict(), gen_save_file)
                    torch.save(self.x2gen.state_dict(), x2gen_save_file)
                    logger.info("Saving the model to: %s\n" % gen_save_file)
                    torch.save(self.dis.state_dict(), dis_save_file)
                    torch.save(self.gen_optim.state_dict(), gen_optim_save_file)
                    torch.save(self.dis_optim.state_dict(), dis_optim_save_file)
                    torch.save(self.x2gen_optim.state_dict(), x2gen_optim_save_file)

                    with torch.no_grad():

                        for ii in range(8):
                            # noise = torch.randn(1, self.latent_size).to(self.device)
                            if ii < 4:
                                ii_r = np.random.permutation(useful_labels_train)[0]
                                batch = get_image_batches_test(data, label_all, useful_labels_train,ii_r, thresh_2)
                            else:
                                ii_r = np.random.permutation(useful_labels_test)[0]
                                batch = get_image_batches_test(data, label_all, useful_labels_test,ii_r, thresh_2)

                            batch = batch.cuda()
                            # pdb.set_trace()

                            im_lr = F.interpolate(batch[0].unsqueeze(0),size=(16,16),mode='bicubic')
                            im_lr_32 = F.interpolate(im_lr,size=(32,32),mode='bicubic')
                            im_32_down = F.interpolate(batch[0].unsqueeze(0),size=(32,32),mode='bicubic')
                            im_32_gen = self.x2gen(im_lr)

                            if torch.mean(torch.abs(im_32_gen - im_32_down)) < torch.mean(torch.abs(im_lr_32 - im_32_down)):
                                c_samples, _,_=self.gen(im_lr,im_32_gen, batch[1:],train=False)
                            else:
                                c_samples, _,_=self.gen(im_lr,im_lr_32, batch[1:],train=False)

                            c_samples = c_samples.detach()
                            im_lr_up = F.interpolate(im_lr,size=(128,128),mode='bicubic')
                            if ii == 0: 
                                all_sample = torch.cat((batch[1:],im_lr_up),dim=0)
                            else:
                                # pdb.set_trace()
                                all_sample = torch.cat((all_sample,batch[1:]),dim=0)
                                all_sample = torch.cat((all_sample,im_lr_up),dim=0)

                            all_sample = torch.cat((all_sample,c_samples),dim=0)
                            all_sample = torch.cat((all_sample,batch[0].unsqueeze(0)),dim=0)
                        # all_sample = all_sample.detach()
                        self.create_grid(
                            all_sample,
                            scale_factor=1,
                            img_file=gen_img_file
                        )
                    # self.get_scores(data)

                    # pdb.set_trace()
                if (epoch >= 8) and (i % int(total_batches / checkpoint_factor + 1) == 0):
                    logger.info('!!!!!!=====!!!!!!')
                    self.get_scores(dataset, 0, logger)
                    t = 1000 * time.time() # current time in milliseconds
                    np.random.seed(int(t) % 2**32) 

                    save_dir = os.path.join(output, 'models')
                    os.makedirs(save_dir, exist_ok=True)
                    gen_save_file = os.path.join(save_dir, "GAN_GEN_"  + "_" + str(epoch) + "_" + str(i) + ".pth")
                    x2gen_save_file = os.path.join(save_dir, "GAN_x2GEN_"  + "_" + str(epoch) + "_" + str(i) + ".pth")
                    dis_save_file = os.path.join(save_dir, "GAN_DIS_"  + "_" + str(epoch) + "_" + str(i) + ".pth")
                    gen_optim_save_file = os.path.join(
                            save_dir, "GAN_GEN_OPTIM_"  + "_" + str(epoch) + "_" + str(i) + ".pth")
                    dis_optim_save_file = os.path.join(
                            save_dir, "GAN_DIS_OPTIM_"  + "_" + str(epoch) + "_" + str(i) + ".pth")
                    x2gen_optim_save_file = os.path.join(
                            save_dir, "GAN_x2GEN_OPTIM_"  + "_" + str(epoch) + "_" + str(i) + ".pth")

                    torch.save(self.gen.state_dict(), gen_save_file)
                    torch.save(self.x2gen.state_dict(), x2gen_save_file)
                    logger.info("Saving the model to: %s\n" % gen_save_file)
                    torch.save(self.dis.state_dict(), dis_save_file)
                    torch.save(self.gen_optim.state_dict(), gen_optim_save_file)
                    torch.save(self.dis_optim.state_dict(), dis_optim_save_file)
                    torch.save(self.x2gen_optim.state_dict(), x2gen_optim_save_file)



            elapsed = timeit.default_timer() - start
            elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]
            logger.info("Time taken for epoch: %s\n" % elapsed)

        logger.info('Training completed.\n')

    def test(self, dataset):
        data = dataset
        label = sio.loadmat('./data/celeA_HQ/label.mat')
        label = label['label'][0]

        unique_label, counts = np.unique(label, return_counts=True)


        thresh = 4 #---th=3 === 3564 classes === 26395 images
        thresh_2 = 4
        # useful_labels = np.where(counts >= thresh)[0]
        useful_labels_train = useful_labels[0:2500]
        useful_labels_test = useful_labels[2500:]

        for i in range(20):
            ii_r = np.random.permutation(2887-2500)[0]

            batch_mix, batch = get_image_batches_test_random(data, label, useful_labels_test,ii_r, 4)

            batch, batch_mix = batch.cuda(), batch_mix.cuda()
                                # pdb.set_trace()

            im_lr = F.interpolate(batch[0].unsqueeze(0),size=(16,16),mode='bicubic')
            # im_lr_32 = F.interpolate(batch[0].unsqueeze(0),size=(32,32),mode='bicubic')
            im_lr_32 = self.x2gen(im_lr)
            c_samples, _,_=self.gen(im_lr, im_lr_32, batch[1:], 1, 5, 0,train=False)

            im_lr_mix = F.interpolate(batch_mix[0].unsqueeze(0),size=(16,16),mode='bicubic')
            # im_lr_mix_32 = F.interpolate(batch_mix[0].unsqueeze(0),size=(32,32),mode='bicubic')
            im_lr_mix_32 = self.x2gen(im_lr_mix)
            c_samples_mix,_,_=self.gen(im_lr_mix,im_lr_mix_32,batch_mix[1:], 1, 5, 0,train=False)

            c_samples = (c_samples - torch.min(c_samples))/(torch.max(c_samples) - torch.min(c_samples))
            c_samples = 2*c_samples - 1


            c_samples_mix = (c_samples_mix - torch.min(c_samples_mix))/(torch.max(c_samples_mix) - torch.min(c_samples_mix))
            c_samples_mix = 2*c_samples_mix - 1

            c_samples = torch.cat((batch,c_samples),0)
            c_samples = c_samples.detach()

            c_samples_mix = torch.cat((batch_mix,c_samples_mix),0)
            c_samples_mix = c_samples_mix.detach()

            c_samples = torch.cat((c_samples, c_samples_mix),0)

            save_image(c_samples,'./x2gen_test_{}.png'.format(i),normalize=True, nrow=5)






    def test_simple(self, dataset, num_examplar):
        num_ref = num_examplar


        r_idx = np.arange(0, num_ref)
        im_lr = data.dataset[r_idx[0]]

        im_hr = torch.zeros([num_ref,3,128,128])

        for j in range(num_ref):
            im_hr[j] = data.dataset[r_idx[1:][j]]

        im_hr = im_hr.cuda()
        im_lr = im_lr.cuda()

        im_lr_ = F.interpolate(im_lr.unsqueeze(0),size=(16,16),mode='bicubic')
        im_lr_32 = self.x2gen(im_lr_)
        c_samples, _,_=self.gen(im_lr_, im_lr_32, im_hr, 1, 5, 0,train=False)
        aa = F.interpolate(im_lr_ ,size=(128,128),mode='bicubic')

        c_samples_ = torch.cat((im_hr, aa),0)
        c_samples_ = torch.cat((c_samples_, c_samples),0)
        c_samples = torch.cat((c_samples_, im_lr.unsqueeze(0)),0)

        save_image(c_samples,'./test_cat_{}.png'.format(i),normalize=True, nrow=num_ref+3, padding=0)



    def get_scores(self, dataset, idx, logger):
        # from skimage.metrics import structural_similarity as ssim
        from math import log10
        from pytorch_msssim import ssim
        from torchvision.utils import make_grid
        data = dataset
        label = sio.loadmat('./data/label_celeba_128.mat')
        label_all = label['label_all'][0]
        # label_train = label['label_train'][0]
        label_test = label['label_test'][0]

        # _, counts_train = np.unique(label_train, return_counts=True)
        unique_label_test, counts_test = np.unique(label_test, return_counts=True)



        thresh = num_ref+1 #---th=3 === 3564 classes === 26395 images
        thresh_2 = num_ref+1
        # useful_labels_train = np.where(counts_train >= thresh)[0]
        useful_labels_test = unique_label_test[np.where(counts_test >= thresh)[0]]



        psnr = 0
        psnr_base = 0
        ssim_base = 0
        ssimm = 0
        psdis_base = 0
        psdis = 0
        with torch.no_grad():
            for i in range(len(useful_labels_test)):
                # print('{} | {}'.format(i+1, len(useful_labels_test)))
                np.random.seed(idx)
                batch = get_image_batches_test(data, label_all, useful_labels_test,i, thresh_2)
                batch = batch.cuda()
                # pdb.set_trace()
                im_lr = F.interpolate(batch[0].unsqueeze(0),size=(16,16),mode='bicubic')
                im_lr_32 = F.interpolate(im_lr,size=(32,32),mode='bicubic')
                # im_lr_bicubic = F.interpolate(batch[0].unsqueeze(0),size=(16,16),mode='bicubic')
                im_32_down = F.interpolate(batch[0].unsqueeze(0),size=(32,32),mode='bicubic')
                im_32_gen = self.x2gen(im_lr)

                if torch.mean(torch.abs(im_32_gen - im_32_down)) < torch.mean(torch.abs(im_lr_32 - im_32_down)):
                    # print('hello')
                    c_samples, _,_=self.gen(im_lr,im_32_gen, batch[1:],train=False)
                else:
                    c_samples, _,_=self.gen(im_lr,im_lr_32, batch[1:],train=False)
                


                # c_samples, weights, weights_32=self.gen(im_lr,im_32_gen, batch[1:], 1, 5, 0,train=False)
                # pdb.set_trace()
                c_samples = make_grid(c_samples, nrow=1, padding=0,normalize=True).mul(255).add_(0.5).clamp_(0, 255)
                gt = make_grid(batch[0], nrow=1, padding=0,normalize=True).mul(255).add_(0.5).clamp_(0, 255)
                im_hr_bicubic = make_grid(F.interpolate(im_lr,size=(128,128),mode='bicubic'), nrow=1, padding=0,normalize=True).mul(255).add_(0.5).clamp_(0, 255)
                # mul(255).add_(0.5).clamp_(0, 255)

     
                mse = torch.mean((c_samples - gt) ** 2)
                _psnr = 10 * log10(255*255 / mse.item())
                psnr += _psnr

                mse_base = torch.mean((im_hr_bicubic-gt) ** 2)
                _psnr_base = 10 * log10(255*255 / mse_base.item())
                psnr_base += _psnr_base

                # gene = c_samples[0].permute(2,1,0).cpu().detach().numpy()
                # ori = batch[0].permute(2,1,0).cpu().detach().numpy()
                # ups = F.interpolate(im_lr_bicubic,size=(128,128),mode='bicubic')[0].permute(2,1,0).cpu().detach().numpy()



                # gene = np.moveaxis(gene,-1,0)
                # ori = np.moveaxis(ori,-1,0)
                # ups = np.moveaxis(ups,-1,0)
                # _ssim = ssim((gene+1)/2, (ori+1)/2,multichannel=True)
                _ssim = ssim( c_samples.unsqueeze(0), gt.unsqueeze(0), data_range=255, size_average=True)
                ssimm += _ssim

                # _ssim_base = ssim((ups+1)/2, (ori+1)/2,multichannel=True)
                _ssim_base = ssim( im_hr_bicubic.unsqueeze(0),gt.unsqueeze(0) , data_range=255, size_average=True)                
                ssim_base += _ssim_base
                # pdb.set_trace()

                # save_image(c_samples,'./test_{}.png'.format(i), nrow=5)


                # psdis_ = torch.mean(self.PerceptualLoss.forward(c_samples, batch[0].unsqueeze(0)))    
                # psdis += psdis_

                # psdis_base_ = torch.mean(self.PerceptualLoss.forward(F.interpolate(im_lr,size=(128,128),mode='bicubic'), batch[0].unsqueeze(0)))    
                # psdis_base += psdis_base_               

            logger.info('\nSSIM: {}, PSNR: {}'.format(ssimm/len(useful_labels_test), psnr/len(useful_labels_test)))
            logger.info('Upsample: SSIM: {}, PSNR: {}'.format(ssim_base/len(useful_labels_test), psnr_base/len(useful_labels_test)))
            # print('{}, {}'.format(psdis/len(useful_labels_test), psdis_base/len(useful_labels_test)))
            # pdb.set_trace()
            return ssimm/len(useful_labels_test), psnr/len(useful_labels_test)







    def visualizePWAVE(self, dataset, celeA_label, visual_size):
        import cv2
        from torchvision.utils import make_grid

        data = dataset
        label = sio.loadmat(celeA_label)

        label_all = label['label_all'][0]
        label_train = label['label_train'][0]
        label_test = label['label_test'][0]

        unique_train_idx, counts_train = np.unique(label_train, return_counts=True)
        unique_test_idx,  counts_test = np.unique(label_test, return_counts=True)


        thresh = 5+1 #---th=3 === 3564 classes === 26395 images
        thresh_2 = num_ref+1
        useful_labels_train = unique_train_idx[np.where(counts_train >= thresh)[0]]
        useful_labels_test = unique_test_idx[np.where(counts_test >= thresh)[0]]

        with torch.no_grad():

            for ii in range(100):
                np.random.seed(1)
                rr = np.random.permutation(len(useful_labels_test))[0:905]
     
                ii_r = rr[ii]
                batch = get_image_batches_test(data, label_all, useful_labels_test,ii_r, thresh_2)

                batch = batch.cuda()
                # pdb.set_trace()

                im_lr = F.interpolate(batch[0].unsqueeze(0),size=(16,16),mode='bicubic')
                im_lr_32 = self.x2gen(im_lr)
                # im_lr_32 = F.interpolate(im_lr,size=(32,32),mode='bicubic')
                # im_lr_64 = F.interpolate(im_lr,size=(64,64),mode='bicubic')


                if visual_size == 16:
                    c_samples, weight, _ =self.gen(im_lr, im_lr_32, batch[1:],train=False)
                else:
                    c_samples, _, weight =self.gen(im_lr, im_lr_32, batch[1:],train=False)

                image_ref = batch[1:]

                # weight_16 = weight_16-1/num_ref

                aa = F.interpolate(weight[0].unsqueeze(0),size=(128,128),mode='bicubic') 
                bb = F.interpolate(weight[1].unsqueeze(0),size=(128,128),mode='bicubic') 
                cc = F.interpolate(weight[2].unsqueeze(0),size=(128,128),mode='bicubic')




                save_image(image_ref[0],'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=3)
                ref_1 = cv2.imread('./temp_1.png')
                heatmap = np.uint8(255 * aa[0][0].cpu().numpy())
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img_1 = heatmap * 0.4 + ref_1
                # cv2.imwrite('./test.png', superimposed_img) 

                # pdb.set_trace()
                save_image(image_ref[1],'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=3)
                ref_2 = cv2.imread('./temp_1.png')
                heatmap = np.uint8(255 * bb[0][0].cpu().numpy())
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img_2 = heatmap * 0.4 + ref_2

                save_image(image_ref[2],'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=3)
                ref_3 = cv2.imread('./temp_1.png')
                heatmap = np.uint8(255 * cc[0][0].cpu().numpy())
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img_3 = heatmap * 0.4 + ref_3

                superimposed_all = np.concatenate((superimposed_img_1, superimposed_img_2, superimposed_img_3), axis=1)


                save_image(c_samples,'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=3)
                c_samples = cv2.imread('./temp_1.png')

                im_lr_up = F.interpolate(im_lr,size=(128,128),mode='bicubic')
                save_image(im_lr_up,'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=3)
                im_lr_up = cv2.imread('./temp_1.png')

                save_image(batch[0],'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=3)
                gt = cv2.imread('./temp_1.png')


                all_images = np.concatenate((superimposed_all, im_lr_up, c_samples, gt), axis=1)

                if visual_size == 16:
                    cv2.imwrite('./output_pwave/pwaveHM_16_{}.png'.format(ii), all_images) 
                else:
                    cv2.imwrite('./output_pwave/pwaveHM_32_{}.png'.format(ii), all_images) 


 


if __name__ == '__main__':
    print('Done.')
