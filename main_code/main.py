import numpy as np
import skimage 
import os ,tqdm, glob , random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

#custom set#
from neuron_util import channel_wise_segmentation
from my_loss import *
import config
from dataset import mydataset
from metrics import *
from reconstruction_network import *
from utils import *
import csv
from logger import Logger


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
        if not os.path.exists(self.path):
            print('----- Make_save_Dir-------------')
            os.makedirs(self.path)
            print(self.log_dir)
        self.path += str(self.model)+str(knum)
        self.imageDir = '../mri_dataset/real_final_train/'
        self.labelDir = '../mri_datasetreal_final_train/'
        self.maskDir = '../othermodel/RefineGAN/data/mask/cartes/mask_1/'

        # self.v_imageDir ='../mri_dataset/real_final_test/'
        # self.v_labelDir = '../mri_dataset/real_final_test/'
        # self.v_maskDir = '../othermodel/RefineGAN/data/mask/cartes/mask_1/'

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
            self.dis = reconstruction_discriminaotr().to(self.device)


        self.optimizerG = torch.optim.Adam(self.gen.parameters(),lr=learning_rate)
        self.optimizerD = torch.optim.Adam(self.dis.parameters(),lr=learning_rate)

        Tensor = torch.cuda.FloatTensor if device else torch.FloatTensor
        if self.use_scheduler == True:
            self.schedulerG = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizerG,100,T_mult=1,eta_min=end_rate)
            self.schedulerD = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizerD,100,T_mult=1,eta_min=end_rate)

        #set multigpu ##
        if self.parallel == True:
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

        evalution =  recon_matrix():
        print('----- Dataset-------------')
        Dataset  = {'train': DataLoader(mydataset(self.imageDir,self.labelDir,self.sampling,self.knum,True),
                                batch_size = self.BATCH_SIZE,
                                shuffle = True
                                num_workers=8),
                    'valid': DataLoader(mydataset(self.imageDir,self.labelDir,self.sampling,self.knum),
                                batch_size = self.BATCH_SIZE,
                                shuffle = True, 
                                num_workers = 4)}

        best_psnr = 0
        best_epoch = 0
        best_ssim = 0
        
        for epoch in range(epoches):
            
            #set loss weight
            ALPHA = 1e+1
            GAMMA = 1e+4
            DELTA = 1e-4
            normalizedImg = np.zeros((192,192))
            

            if epoch % validnum == 0:
                self.gen.eval()
                self.dis.eval()
                phase = 'valid'
                self.save_model('last')
                val_recon_psnr = 0
                val_finsl_psnr = 0
                val_recon_ssim = 0
                val_finsl_ssim = 0
                
            else :
                self.gen.train()
                self.dis.train()
                phase = 'train'
                self.schedulerG.step(epoch)
                self.schedulerD.step(epoch)
                recon_psnr = 0
                final_psnr = 0
                recon_ssim = 0
                final_ssim = 0

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


                    #image losses
                    recon_loss = self.Lloss(_image,recon_img) 
                    error_loss = self.Closs(_image,final_img)        
                    #compare frequecy image
                    #frequency loss
                    freq_img,_ = apply_mask(final_img,mask)
                    freq_loss = self.Floss(under_image.to(torch.float32),freq_img).to(torch.float32)
                    
                    #WGan gen loss
                    boost_dis_fake = self.dis(final_img)

                    boost_fake_A=self.WGAN_loss.loss_gen(boost_dis_fake)
                    #add regulization (total variation)
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
                
                    final_img = cv2.normalize(final_img.cpu.numpy(),  normalizedImg, 0, 1, cv2.NORM_MINMAX)
                    recon_img = cv2.normalize(recon_img.cpu.numpy(),  normalizedImg, 0, 1, cv2.NORM_MINMAX)
                    _image = cv2.normalize(_image.cpu.numpy(),  normalizedImg, 0, 1, cv2.NORM_MINMAX)
                    
                    final_psnr += evalution.PSNR(final_img,_image,1)
                    recon_psnr += evalution.PSNR(recon_img,_image,1)
                    final_ssim += evalution.PSNR(final_img,_image,1)
                    recon_ssim += evalution.PSNR(recon_img,_image,1)
                    
                    
                else:
                    with torch.no_grad():
                        self.gen.eval()
                        self.dis.eval()

                        for i, batch in enumerate(tqdm(valid_dataset)):
                            inputa, mask = batch[0].to(self.device), batch[2].to(self.device)
                            
                            inputa=inputa.float()
                            mask = mask.float()
                            
                            under_im,inputs = apply_mask(inputa,mask)
                            
                            prediction = self.gen(inputs.float())
                            
                            prediction = torch.add(inputs.float() , prediction)                            
                            pred = update(prediction,inputa,mask)
                            
                            recon_loss = self.Lloss(inputa,prediction) 
                            error_loss = self.Closs(inputa,pred)        
                            #compare frequecy image
                            #frequency loss
                            freq_img,_ = apply_mask(pred,mask)
                            freq_loss = self.Floss(under_im.to(torch.float32),freq_img).to(torch.float32)

                            summary_val = {'recon_loss':recon_loss,
                                'error_loss':recon_loss,
                                'freq_loss':freq_loss}
               

                            #calcuate matrix
                            prediction = cv2.normalize(prediction.cpu.numpy(),  normalizedImg, 0, 1, cv2.NORM_MINMAX)
                            pred = cv2.normalize(pred.cpu.numpy(),  normalizedImg, 0, 1, cv2.NORM_MINMAX)
                            inputa = cv2.normalize(inputa.cpu.numpy(),  normalizedImg, 0, 1, cv2.NORM_MINMAX)
                            
                            val_final_psnr += evalution.PSNR(final_img,_image,1)
                            val_recon_psnr += evalution.PSNR(recon_img,_image,1)
                            val_final_ssim += evalution.PSNR(final_img,_image,1)
                            val_recon_ssim += evalution.PSNR(recon_img,_image,1)
                            
                    
                            # avg_ssim += ssim
                            # z_avg_ssim += z_ssim
                            # print(z_ssim.shape)
            
            if phase == 'train'
                i=float(i)     
                evalutiondict = {'final_psnr':final_psnr/i,'recon_psnr':recon_psnr/i,
                                    'final_ssim':final_ssim/i,'recon_ssim':recon_ssim/i}
                summary_val.update(evalutiondict)
                self.printall(summary_val,epoch,'train')
                
                if (evalutiondict['final_psnr'] > best_psnr) or (evalutiondict['final_ssim'] > best_ssim):
                    self.save_model('lastsave_model')
                    best_psnr = evalutiondict['final_psnr']
                    best_epoch = epoch
                    best_ssim = evalutiondict['final_ssim']

            else :
                i=float(i)
                
                evalutiondict = {'val_final_psnr':val_final_psnr/i,'val_recon_psnr':val_recon_psnr/i,
                                    'val_final_ssim':val_final_ssim/i,'val_recon_ssim':val_recon_ssim/i}
                summary_val.update(evalutiondict)
                self.printall(summary_val,epoch,'valid')

                # class_list=['recon_psnr','final_psnr','recon_ssim','final_ssim']
                class_list,evalutiondict = self.logger.convert_to_list(evalutiondict)
                self.logger.save_csv(evalutiondict,'valid',class_list)

                total_image = {'recon_image':prediction.cpu.numpy(),
                                'final_image':pred.cpu.numpy(),
                                'input_image':inputa.cpu.numpy()}
                
                self.re_normalize(total_image,255)
                total_image.update('zero_image':inputs.cpu.numpy())

                recon_error_img = total_image['input_image'] - total_image['recon_image']
                final_error_img = total_image['input_image'] - total_image['final_image']
                zero_error_img = total_image['input_image'] - total_image['zero_image']

                total_image.update({'recon_error_img':recon_error_img,
                                    'final_error_img':final_error_img,
                                    'zero_error_img':zero_error_img,
                                    'mask':mask.cpu.numpy()})
                
                self.save_image(total_image,epoch)

                
    def printall(self,summary_val,epoch,name):
        self.logger.print_value(summary_val,name)
        self.logger.summary_scalars(summary_val,epoch)


    def re_normalize(self,convert_dict_image,max_value):
        normalizedImg = np.zeros((192,192))
        for i, scalar in enumerate(convert_dict_image):
            convert_dict_image[i] = cv2.normalize(convert_dict_image[i],  normalizedImg, 0, max_value, cv2.NORM_MINMAX)

                # self.writer.add_scalar(str(tag)+'/'+str(scalar),scalar_dict[scalar],step)

    def save_model(self,name):
        self.loggerself.save_model(self.gen,self.dis,epoch,self.path)

        torch.save({"gen_model":self.gen.state_dict(),
                    "dis_model":self.dis.state_dict(),
                    "optimizerG":self.optimizerG.state_dict(),
                    "optimierD":self.optimizerD.state_dict(),
                    "epochs":epoch},
                    self.path+str(name)"_save_models{}.pth")

    def testing(self):
        t_imageDir = '../mri_dataset/real_final_test/'
        t_labelDir = '../mri_dataset/real_final_test/'
        
        
        Dataset  = { 'valid': DataLoader(mydataset(self.imageDir,self.labelDir,self.sampling,self.knum),
                                batch_size = self.BATCH_SIZE,
                                shuffle = True, 
                                num_workers = 4),
                    'test': DataLoader(mydataset(t_imageDir,t_labelDir,self.sampling,self.knum),
                                batch_size = self.BATCH_SIZE,
                                shuffle = True, 
                                num_workers = 4)}
        
        self.load_model()
        
        phase = 'test'
        test_final_psnr = 0
        test_recon_psnr = 0
        test_final_ssim = 0
        test_recon_ssim = 0
        test_evalution =  recon_matrix():

        for i, batch in enumerate(tqdm(Dataset[phase])):
            with torch.no_grad():
                self.gen.eval()
                self.dis.eval()


                inputa, mask = batch[0].to(self.device), batch[2].to(self.device)
                inputa=inputa.float()
                mask = mask.float()
                
                under_im,inputs = apply_mask(inputa,mask)
                
                prediction = self.gen(inputs.float())
                
                prediction = torch.add(inputs.float() , prediction)                            
                pred = update(prediction,inputa,mask)

                #calcuate matrix
                prediction = cv2.normalize(prediction.cpu.numpy(),  normalizedImg, 0, 1, cv2.NORM_MINMAX)
                pred = cv2.normalize(pred.cpu.numpy(),  normalizedImg, 0, 1, cv2.NORM_MINMAX)
                inputa = cv2.normalize(inputa.cpu.numpy(),  normalizedImg, 0, 1, cv2.NORM_MINMAX)
                
                test_final_psnr += test_evalution.PSNR(final_img,_image,1)
                test_recon_psnr += test_evalution.PSNR(recon_img,_image,1)
                test_final_ssim += test_evalution.PSNR(final_img,_image,1)
                test_recon_ssim += test_evalution.PSNR(recon_img,_image,1)
                    
        
            i=float(i)

            self.logger.changedir()
            
            evalutiondict = {'val_final_psnr':test_final_psnr/i,'val_recon_psnr':test_recon_psnr/i,
                                'val_final_ssim':test_final_ssim/i,'val_recon_ssim':test_recon_ssim/i}
            
            # class_list=['recon_psnr','final_psnr','recon_ssim','final_ssim']
            class_list,evalutiondict = self.logger.convert_to_list(evalutiondict)
            self.logger.save_csv(evalutiondict,'valid',class_list)

            total_image = {'recon_image':prediction.cpu.numpy(),
                            'final_image':pred.cpu.numpy(),
                            'input_image':inputa.cpu.numpy()}
            
            self.re_normalize(total_image,255)
            total_image.update('zero_image':inputs.cpu.numpy())

            recon_error_img = total_image['input_image'] - total_image['recon_image']
            final_error_img = total_image['input_image'] - total_image['final_image']
            zero_error_img = total_image['input_image'] - total_image['zero_image']

            total_image.update({'recon_error_img':recon_error_img,
                                'final_error_img':final_error_img,
                                'zero_error_img':zero_error_img,
                                'mask':mask.cpu.numpy()})
            
            self.save_image(total_image,epoch)

reconstruction = main()
reconstruction.trainning()
reconstruction.testing()