"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
import os
from os.path import join as ospj
import time
import datetime
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from metrics.eval import calculate_metrics

import numpy as np
import random
import kornia
import csv
import cv2

import torchvision.utils as vutils
from metrics.fid import calculate_fid_given_paths

class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        self.nets, self.nets_ema = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]
        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), **self.nets_ema)]

        self.to(self.device)
        
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

        if os.path.isfile(args.checkpoint_unet):
            self.nets.segmentator.load_state_dict(torch.load(args.checkpoint_unet, map_location=lambda storage, loc: storage))
            print("Loading pretrained UNet...")
        else:
            print("Wrong UNet file")

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        
        mask_values = [0.85,0.90,0.95,1.0,1.05,1.10,1.15]
        
        for i in range(args.resume_iter, args.total_iters):
            
            # random mask size
            sel_attr = random.randrange(3)
            sel_mask = random.randrange(len(mask_values))
            if args.attribute_resize:
                mask_size = mask_values[sel_mask]
            else:
                mask_size = 1.0
 
            # fetch images and labels
            inputs = next(fetcher)
            x_real, y_org, m_org = inputs.x_src, inputs.y_src, inputs.m_src
            x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2
        
            masks = torch.unsqueeze(m_org[:,0,:,:],1) if args.w_hpf > 0 else None
            if masks is not None:
                masks = m_org
                masks = (masks - torch.min(masks))/(torch.max(masks) - torch.min(masks)) 
                masks[masks != masks] = 0 #avoids NaN
    
            # train the discriminator
            d_loss, d_losses_latent = compute_d_loss(
                nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks.clone(), mask_size=mask_size, attribute=sel_attr)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            d_loss, d_losses_ref = compute_d_loss(
                nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=masks.clone(), mask_size=mask_size, attribute=sel_attr)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            # train the generator
            g_loss, g_losses_latent = compute_g_loss(
                nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks.clone(), mask_size=mask_size, attribute=sel_attr)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()

            g_loss, g_losses_ref = compute_g_loss(
                nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks.clone(), mask_size=mask_size, attribute=sel_attr)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()

            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                        ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)
                
            if (i+1) % args.save_losses == 0:                
                csv_columns = ['D/latent_real', 'D/latent_fake', 'D/latent_reg', 'D/ref_real', 'D/ref_fake', 'D/ref_reg', 'G/latent_adv',
                               'G/latent_sty', 'G/latent_ds', 'G/latent_cyc', 'G/latent_mask', 'G/ref_adv', 'G/ref_sty', 'G/ref_ds', 'G/ref_cyc',
                               'G/ref_mask', 'G/lambda_ds']
                
                csv_file = args.main_dir+"/losses.csv"
                if i+1 == args.save_losses:
                    with open(csv_file, 'w') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                        writer.writeheader()
                        writer.writerow(all_losses)
                else:
                    with open(csv_file, 'a') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                        writer.writerow(all_losses)

            # generate images for debugging
            if (i+1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i+1)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)

            # compute FID and LPIPS if necessary
            if (i+1) % args.eval_every == 0:
                calculate_metrics(nets_ema, args, i+1, mode='latent')
                calculate_metrics(nets_ema, args, i+1, mode='reference')

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
        calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')

def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None, mask_size=None, attribute=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)
        x_fake = nets.generator(x_real, s_trg, masks=masks, mask_size=mask_size, attribute=attribute)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())


def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None, mask_size=None, attribute=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)
       
    x_fake = nets.generator(x_real, s_trg, masks=masks.clone(), mask_size=mask_size, attribute=attribute)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)
    
    # cycle-consistency loss
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=masks.clone(), mask_size=mask_size, attribute=attribute, cycle=True)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x_real, s_trg2, masks=masks, mask_size=mask_size, attribute=attribute)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # segmentation loss            
    masks_pred = nets.segmentator(x_fake)
   
    size_tmp = int(mask_size* x_fake.size(2))
    mask_attr = F.interpolate(torch.unsqueeze(masks[:,attribute],1), size=size_tmp, mode='bilinear')
    mask_attr = kornia.geometry.transform.center_crop(mask_attr, (x_fake.size(2), x_fake.size(2)))   
    mask_attr = torch.round(mask_attr)

    # eyes
    if attribute == 0:
        mask_attr_pred = torch.unsqueeze(masks_pred[:,1],1)
        which_eye = random.randrange(2)
        # left
        if which_eye == 0:
            #90:130, 65:115 (wihtout resizing)
            cx = 110
            cy = 90
            x_tmp = int(40*mask_size/2)
            y_tmp = int(50*mask_size/2)
        #right
        else:
            #90:130, 140:190 (wihtout resizing)
            cx = 110
            cy = 165
            x_tmp = int(40*mask_size/2)
            y_tmp = int(50*mask_size/2)
    # lips
    elif attribute == 1:
        mask_attr_pred = torch.unsqueeze(masks_pred[:,2],1)
        #175:220,85:165 (wihtout resizing)
        cx = 197 #(175+220)/2
        cy = 128 #(85+165)/2
        x_tmp = int(45*mask_size/2)  #(220-175)
        y_tmp = int(80*mask_size/2) #(165-85)
    #nose
    else:       
        mask_attr_pred = torch.unsqueeze(masks_pred[:,3],1)
        #105:180,100:155 (wihtout resizing)
        cx = 143
        cy = 128
        x_tmp = int(75*mask_size/2)
        y_tmp = int(55*mask_size/2)
         
    mask_attr = mask_attr[:,:,cx-x_tmp:cx+x_tmp,cy-y_tmp:cy+y_tmp]
    mask_attr_pred = mask_attr_pred[:,:,cx-x_tmp:cx+x_tmp,cy-y_tmp:cy+y_tmp]
    loss_mask = F.binary_cross_entropy_with_logits(mask_attr_pred, mask_attr)
    
    loss = args.lambda_adv * loss_adv + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc + args.lambda_seg * loss_mask
    
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item(),
                       mask=loss_mask.item())

def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)

def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(outputs=d_out.sum(), inputs=x_in,
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg