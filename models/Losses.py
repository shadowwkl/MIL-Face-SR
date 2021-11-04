
import numpy as np

import torch
import torch.nn as nn


# =============================================================
# Interface for the losses
# =============================================================

class GANLoss:
    """ Base class for all losses

        @args:
        dis: Discriminator used for calculating the loss
             Note this must be a part of the GAN framework
    """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")



class LogisticGAN(GANLoss):
    def __init__(self, dis):
        super().__init__(dis)

    # gradient penalty
    def R1Penalty(self, real_img):


        real_img = torch.autograd.Variable(real_img, requires_grad=True)
        real_logit = self.dis(real_img)
        # real_logit = apply_loss_scaling(torch.sum(real_logit))
        real_grads = torch.autograd.grad(outputs=real_logit, inputs=real_img,
                                         grad_outputs=torch.ones(real_logit.size()).to(real_img.device),
                                         create_graph=True, retain_graph=True)[0].view(real_img.size(0), -1)
        # real_grads = undo_loss_scaling(real_grads)
        r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
        return r1_penalty

    def dis_loss(self, real_samps, fake_samps, r1_gamma=10.0):
        # Obtain predictions
        r_preds = self.dis(real_samps)
        f_preds = self.dis(fake_samps)

        loss = torch.mean(nn.Softplus()(f_preds)) + torch.mean(nn.Softplus()(-r_preds))

        if r1_gamma != 0.0:
            r1_penalty = self.R1Penalty(real_samps.detach()) * (r1_gamma * 0.5)
            loss += r1_penalty

        return loss

    def gen_loss(self, _, fake_samps):
        f_preds = self.dis(fake_samps)

        return torch.mean(nn.Softplus()(-f_preds))

        
