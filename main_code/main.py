import numpy as np
import skimage 
import os ,tqdm, glob , random
import torch
import torch.nn.functional as F
import yaml

from torch import nn, optim
from torchvision import models ,transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

#custom set#
from my_network import *
from neuron_util import *
from neuron_util import channel_wise_segmentation
from my_loss import *
import config
from mydataset import prjection_mydataset,mydataset_2d
from logger import Logger
from metrics import *
import argparse

from main_reconstruction_network import *

from main_reconsturction_torch_utils import *
import matplotlib.pyplot as plt
import csv

class main:

    #initilize Generaotor and discriminator
    def __init__(self):
        self.model = 'basic'

        # self.path = './unet_denseunet_wgangp/10percent_cascade_boost_test'
        #set optimize
        self.epoches=500
        self.batch_size=4
        self.sampling = 5
        self.parallel = False
        self.learning_rate = 1e-4
        self.end_lr = 1e-6
        self.use_scheduler = True
        self.deleteall = False
        self.knum = 1

        ############################################new dataset####################################
        self.path = '../save_model'

        self.imageDir = '../mri_dataset/real_final_train/'
        self.labelDir = '../mri_datasetreal_final_train/'
        self.maskDir = '../othermodel/RefineGAN/data/mask/cartes/mask_1/'

        # self.v_imageDir ='../mri_dataset/real_final_test/'
        # self.v_labelDir = '../mri_dataset/real_final_test/'
        # self.v_maskDir = '../othermodel/RefineGAN/data/mask/cartes/mask_1/'

        self.t_imageDir = '../mri_dataset/real_final_test/'
        self.t_labelDir = '../mri_dataset/real_final_test/'
        self.t_maskDir = '../othermodel/RefineGAN/data/mask/cartes/mask_1/'
        #set a dataset ##
        # max_grad_norm = 2.
        self.gen_num = 10
        self.logger = Logger(self.path,batch_size=self.batch_size,delete=deleteall,num=str(knum),name=model+'_'+self.model)
        
        
        self.device = torch.device('cuda:0'if torch.cuda.is_available() else "else")
        
    def load_model(self):

        if self.model == 'basic':
            print('----------generator---------')
            self.gen = reconstruction_resnet101unet().to(self.device)
            print('----------discriminator---------')
            self.dis = discriminator().to(self.device)


        self.optimizerG = torch.optim.Adam(self.gen.parameters(),lr=learning_rate)
        self.optimizerD = torch.optim.Adam(self.dis.parameters(),lr=learning_rate)

        Tensor = torch.cuda.FloatTensor if device else torch.FloatTensor
        if use_scheduler == True:
            self.schedulerG = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizerG,100,T_mult=1,eta_min=end_rate)
            self.schedulerD = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizerD,100,T_mult=1,eta_min=end_rate)

        #set multigpu ##
        if parallel == True:
            self.gen = torch.nn.DataParallel(self.dddddgen, device_ids=[0,1])    
            self.dis = torch.nn.DataParallel(self.dis, device_ids=[0,1])    
            
    def load_loss(self):
        ###loss initilize
        
        TV_loss=TVLoss().to(self.device)
        
        self.WGAN_loss = WGANLoss(device)
        self.Lloss = torch.nn.L1Loss().to(self.device)
        self.Closs = torch.nn.L1Loss().to(self.device)
        self.Floss = torch.nn.L1Loss().to(self.device)
        self.TVloss = TV_loss.to(self.device)
        
        ###trainning###
        
    def trainning(self):
        
        self.load_model()
        self.load_loss()

        print('----- Dataset-------------')
        Dataset  = {'train': DataLoader(mydataset(imageDir,labelDir,maskDir,self.sampling),
                                batch_size = self.BATCH_SIZE,
                                num_workers=8),
                    'valid': DataLoader(mydataset(v_imageDir,v_labelDir,v_maskDir,self.sampling),
                                batch_size = self.BATCH_SIZE,
                                shuffle = True, 
                                num_workers = 4)}


        for epoch in range(epoches):
            
            #set loss weight
            ALPHA = 1e+1
            GAMMA = 1e+4
            DELTA = 1e-4
            total_PSNR = []
            total_SSIM = []


            if epoch % validnum == 0:
                self.gen.eval()
                self.dis.eval()
                phase = 'valid'
                self.save_model(self.gen,self.dis,epoch,self.path)

            else :
                self.gen.train()
                self.dis.train()
                phase = 'train'
                self.schedulerG.step(epoch)
                self.schedulerD.step(epoch)

            for i, batch in enumerate(tqdm(Dataset[phase])):
                if phase == 'train'
                    # set model inputa
                    _image = Variable(batch[0]).to(self.device)
                    mask = Variable(batch[2]).to(self.device)
                    
                    self.optimizerG.zero_grad()
                    
                    ##########preprocessing FFT, iFFT ###########
                    # apply_mask function do multiple cartasian mask randomly
                    under_image,zero_image = apply_mask(_image,mask) 
                    # when apply Furier transform it change data dype double so we change floatdatatype
                    zero_image=zero_image.float()
                    
                    ###############train gen#################
                    #generator Add ZFimage + feature image 
                    trained_img = self.gen(zero_image)
                    recon_img = torch.add(trained_img , zero_image).float().to(self.device)

                    #update mask again
                    mask=mask.to(torch.float32)
                    _image=_image.to(torch.float32)
                    
                    final_img = update(recon_img,_image,mask).to(torch.float32).to(self.device)


                    #reconGAN loss
                    recon_loss = self.Lloss(_image,recon_img) 
                    error_loss = self.Closs(_image,final_img)        
                    #compare frequecy image
                    #WGan gen loss
                    freq_img,_ = apply_mask(final_img,mask)
                    freq_loss = self.Floss(under_image.to(torch.float32),freq_img).to(torch.float32)
                    
                    boost_F_loss = freq_loss
                    
                    #WGan gen loss
                    boost_dis_fake = self.dis(final_img)

                    boost_fake_A=self.WGAN_loss.loss_gen(boost_dis_fake)
                    TV_a=self.TVloss(final_img).to(torch.float32)
                    
                    #reconstruction loss
                    boost_R_loss = recon_loss + error_loss + freq_loss
                    boost_R_loss += boost_fake_A
                    boost_R_loss += TV_a
                    loss_g = boost_R_loss
                    loss_g.backward(retain_graph=True)
                    
                    self.optimizerG.step()

                    summary_val = {'recon_loss':recon_loss,
                                'error_loss':recon_loss,
                                'freq_loss':freq_loss,
                                'TV_a':TV_a}

                    ###############train discrim#################
                    
                    if i > self.gen_num:

                        dis_real_img = self.dis(_image)
                        dis_fake_img = self.dis(final_image)
                        
                        optimizerD.zero_grad()

                        #calcuate loss function
                        dis_loss = self.WGAN_loss.loss_disc(dis_fake_img,dis_real_img)
                        # loss_RMSE   = Lloss(_image,final_image)            
                        GP_loss=compute_gradient_penalty(dis, _image, final_image)
                        
                        discrim_loss = dis_loss + GP_loss
                        discrim_loss.backward(retain_graph=True)

                        optimizerD.step()
                        summary_val.update({'dis_loss':dis_loss,'GP_loss',GP_loss})

                    self.printall(summary_val,epoch)
                
                else:
                    
                    avg_psnr = 0
                    z_avg_psnr = 0
                    avg_ssim = 0
                    z_avg_ssim = 0

                    with torch.no_grad():
                        gen.eval()
                        dis.eval()

                        for i, batch in enumerate(tqdm(valid_dataset)):
                            inputa, mask = batch[0].to(self.device), batch[2].to(self.device)
                            
                            inputa=inputa.float()
                            mask = mask.float()
                            
                            under_im,inputs = apply_mask(inputa,mask)
                            
                            prediction = gen(inputs.float()).to(cuda1)
                            
                            prediction = torch.add(inputs.float() , prediction)
                            
                            
                            pred = update(prediction,inputa,mask)
                            psnr=psNR(npt2imag(inputa.cpu().numpy()),npt2imag(pred.cpu().detach().numpy()),255.0)
                            z_psnr=psNR(npt2imag(inputa.cpu().numpy()),npt2imag(inputs.cpu().detach().numpy()),255.0)
                            ssim=SSIM(npt2imag(inputa[0,0,...].cpu().numpy()),npt2imag(pred[0,0,...].cpu().detach().numpy()),255.0)
                            z_ssim=SSIM(npt2imag(inputa[0,0,...].cpu().numpy()),npt2imag(inputs[0,0,...].cpu().detach().numpy()),255.0)
                            avg_psnr += psnr
                            z_avg_psnr += z_psnr
                            # avg_ssim += ssim
                            # z_avg_ssim += z_ssim
                            # print(z_ssim.shape)

    def printall(self,summary_val,epoch):
        self.logger.print_value(summary_val,'train')
        self.logger.summary_scalars(summary_val,epoch)

    def save_csv(self,summary,epoch):
    
    def save_image(self,save_stack_image,epoch):
        save_stack_images

    def save_model(self,model):
            self.save_model(self.gen,self.dis,epoch,self.path)
            torch.save({"gen_model":self.gen.state_dict(),
                        "dis_model":self.dis.state_dict(),
                        "optimizerG":self.optimizerG.state_dict(),
                        "optimierD":self.optimizerD.state_dict(),
                        "epochs":epoch},
                        self.path+"lastsave_models{}.pth")
                
                print("======{}/{}epochs======".format(epoch,epoches))
                # print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(valid_dataset)))

                print("===> Avg. reconPSNR: {:.4f} dB".format(avg_psnr / len(valid_dataset)))
                print("===> Avg. zeroPSNR: {:.4f} dB".format(z_avg_psnr / len(valid_dataset)))
                print("===> Avg. reconSSIM: ",(avg_ssim[...,0]) / len(test_dataset))
                print("===> Avg. zeroSSIM: ",(z_avg_ssim[...,0]) / len(test_dataset))
                
                print("===> lasg PSNR: {:.4f} db".format(psnr))
                print("===> T_loss_G : {:.4f}".format(loss_G))
                print("===> T_loss_D : {:.4f}".format(T_loss_d))
                print("===> boost_loss_g : {:.4f}".format(boost_loss_g))
                print("===> R_loss : {:.4f}".format(R_loss))
                print("===> E_loss : {:.4f}".format(E_loss))
                print("===> F_loss : {:.4f}".format(F_loss))
                print("===> boost_R_loss : {:.4f}".format(boost_R_loss))
                print("===> boost_E_loss : {:.4f}".format(boost_E_loss))
                print("===> boost_F_loss : {:.4f}".format(boost_F_loss))

                # print("===> TV : {:.4f}".format(TV_loss))boost_WG_loss
                print("===> w_loss_G : {:.4f}".format(WG_loss))
                print("===> boost_WG_loss : {:.4f}".format(boost_WG_loss))
                print("===> w_loss_D : {:.4f}".format(w_loss_D))
                # print("===> l1_regul_d : {:.4f}".format(l1_regul_d))
                # print("===> l1_regu : {:.4f}".format(l1_regu))

                # print("===> WG_loss : {:.4f}".format(WG_loss))
                
                
                # optimizerG.step()
                # optimizerD.step()

                
                ###using tensorboard 

                writer.add_scalar('Avg_psnr',avg_psnr / len(valid_dataset),epoch)
                writer.add_scalar('zero_avg_psnr',z_avg_psnr / len(valid_dataset),epoch)
                # writer.add_scalar('Avg_ssim',avg_ssim / len(test_dataset),epoch)
                # writer.add_scalar('zero_avg_ssim',z_avg_ssim / len(test_dataset),epoch)

                writer.add_scalar('loss_G',loss_G,epoch)
                writer.add_scalar('psnr',psnr,epoch)
                # writer.add_scalar('loss_D',w_loss_D,epoch)
                writer.add_scalar('R_loss',R_loss,epoch)
                writer.add_scalar('E_loss',E_loss,epoch)
                writer.add_scalar('F_loss',F_loss,epoch)
                writer.add_scalar('WG_loss',WG_loss,epoch)
                writer.add_scalar('WD_loss',w_loss_D,epoch)
                # # FA = FA.to(torch.float32)
                # # print(A.shape)
                # writer.add_images('real_image', A[:,0,...],dataformats='CHW')
                # writer.add_images('imag_image', A[:,0,...],dataformats='CHW')
                # writer.add_images('mask',R[:,0,...,0],dataformats='CHW')
                # writer.add_images('sampl_mask',UA[:,0,...,0],dataformats='CHW')
                # writer.add_images('uner_image',(RA[:,0,...]),dataformats='CHW')
                # writer.add_images('recon',(R1[:,0,...]),dataformats='CHW')
                # writer.add_images('error_recon',(A[:,0,...].to(torch.float32)- FA[:,0,...].to(torch.float32)),dataformats='CHW')
                # writer.add_images('error_last',(A[:,0,...].to(torch.float32)- R1[:,0,...].to(torch.float32)),dataformats='CHW')
                # writer.add_images('feature',(FA[:,0,...]),dataformats='CHW')
                # skimage.io.imsave(path + "/real"+"_"+str(i)+".tif",np.transpose(A[0,...].cpu().numpy(),[3,0,1,2]]))
                # # skimage.io.imsave("./output_3dconv_v5_l1_contorv2/imgs/fakeb_"+"_"+str(i)+".tif",(np.transpose(fake_b[0,...,0:1],[1, 2, 3, 0])))
                # skimage.io.imsave(path + "/zero_"+"_"+str(i)+".tif",np.transpose(ZA[0,...].cpu().detach().numpy(),[3,0,1,2]]))
                # # skimage.io.imsave("./output_3dconv_v5_l1_contorv2/imgs/zerob_"+"_"+str(i)+".tif",(n(zero_b[0,...,0:1],[1, 2, 3, 0])))
                # skimage.io.imsave(path + "/fake"+"_"+str(i)+".tif",np.transpose(RA[0,...].cpu().detach().numpy(),[3,0,1,2]]))
                # skimage.io.imsave(path + "/recon"+"_"+str(i)+".tif",np.transpose(R1[0,...].cpu().detach().numpy(),[3,0,1,2]]))
                # skimage.io.imsave(path + "/feature"+"_"+str(i)+".tif",np.transpose(np.abs(FA[0,...].cpu().detach().numpy()),[3,0,1,2]]))
                # skimage.io.imsave(path + "/under_image"+"_"+str(i)+".tif",np.transpose(np.abs(UA[0,...].cpu().detach().numpy()),[3,0,1,2]]))
                
                
            #         total = [A[0,0,...,0],R[0,0,...,0],
            #                  FA[0,0,...,0],RA[0,0,...,0],R1[0,0,...,0]]        
                #update leraning rates

            if epoch%50 == 0:
                # torch.save(gen.state_dict(),path + "/save_models.pth")
                # torch.save(dis.state_dict(),path + "/save_models.pth")
                torch.save({"gen": gen.module.state_dict(),
                    "gen2":gen2.module.state_dict(),
                    "dis": dis.module.state_dict(),
                    "optimizerG": optimizerG.state_dict(),
                    "optimizerD": optimizerD.state_dict(),
                    "optimizerG2":optimizerG2.state_dict()},
                    path + "/save_models.pth")
            if epoch%37 == 0:
                # print(A.shape)
                skimage.io.imsave(path + "/Sn1_"+"_"+str(i)+".tif",np.transpose(A[0,...].cpu().numpy(),[3,0,1,2]))
                skimage.io.imsave(path + "/zero_"+"_"+str(i)+".tif",np.transpose(ZA[0,...].cpu().detach().numpy(),[3,0,1,2]))
                skimage.io.imsave(path + "/Sp1_"+"_"+str(i)+".tif",np.transpose(RA[0,...].cpu().detach().numpy(),[3,0,1,2]))
                skimage.io.imsave(path + "/S1_"+"_"+str(i)+".tif",np.transpose(R1[0,...].cpu().detach().numpy(),[3,0,1,2]))
                skimage.io.imsave(path + "/feature"+"_"+str(i)+".tif",np.transpose(FA[0,...].cpu().detach().numpy(),[3,0,1,2]))
                skimage.io.imsave(path + "/under_image"+"_"+str(i)+".tif",np.transpose(np.abs(UA[0,...].cpu().detach().numpy()),[0,3,1,2]))
                skimage.io.imsave(path + "/featurenoabs"+"_"+str(i)+".tif",np.transpose(FA[0,...].cpu().detach().numpy(),[3,0,1,2]))
                skimage.io.imsave(path + "/zero_error_img"+"_"+str(i)+".tif",np.transpose(np.abs(A[0,...].cpu().numpy()-ZA[0,...].cpu().detach().numpy()),[3,0,1,2]))
                skimage.io.imsave(path + "/recon_error_img"+"_"+str(i)+".tif",np.transpose(np.abs(A[0,...].cpu().numpy()-R1[0,...].cpu().detach().numpy()),[3,0,1,2]))
                skimage.io.imsave(path + "/R1_"+"_"+str(i)+".tif",np.transpose(R1[0,...].cpu().detach().numpy(),[3,0,1,2]))
                skimage.io.imsave(path + "/FRA_"+"_"+str(i)+".tif",np.transpose(FRA[0,...].cpu().detach().numpy(),[3,0,1,2]))
                skimage.io.imsave(path + "/FRn1"+"_"+str(i)+".tif",np.transpose(FRn1[0,...].cpu().detach().numpy(),[3,0,1,2]))
                skimage.io.imsave(path + "/zero_boost_img"+"_"+str(i)+".tif",np.transpose(np.abs(A[0,...].cpu().numpy()-FRA[0,...].cpu().detach().numpy()),[3,0,1,2]))
                skimage.io.imsave(path + "/recon_boost_img_"+"_"+str(i)+".tif",np.transpose(np.abs(A[0,...].cpu().numpy()-FRn1[0,...].cpu().detach().numpy()),[3,0,1,2]))
                
            
reconstruction = main()
reconstruction.load_loss()
reconstruction.trainning()