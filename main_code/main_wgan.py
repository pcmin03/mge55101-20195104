import numpy as np
import skimage 
import os  , random
from tqdm import tqdm
from glob import glob
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

#custom set#

from myloss import *

from dataset import mydataset
from matrix import recon_matrix
from reconstruction_network import *
from utils import *
import csv
from logger import Logger


class main:

    #initilize Generaotor and discriminator
    def __init__(self,args):
        self.model = 'patchgan'
# ../../save_model/paraell_2.5_attention_new_data_patchdiscrim_patchgan3/
# paraell_2.5_attention_new_data_effcientunet3
        # self.path = './unet_denseunet_wgangp/10percent_cascade_boost_test'
        #set optimize
        self.batch_size=8
        self.sampling = args.sampling
        self.epoches=301
        self.parallel = True
        self.learning_rate = 1e-4
        self.end_lr = 1e-5
        self.use_scheduler = True
        self.knum = args.knum
        deleteall = False
        self.validnum = 10
        self.perceptual_loss = True

        ############################################new dataset####################################
        self.path = '../../save_model'
        if not os.path.exists(self.path):
            print('----- Make_save_Dir-------------')
            os.makedirs(self.path)
            print(self.path)
        # self.path += '/paraell_5_'+str(self.model)+str(self.knum)+'/'
        # self.path += '/paraell_'+str(self.sampling)+'_selfsupervised_attention_new_data_patchdiscrim_'+str(self.model)+str(self.knum)+'/'
        self.path += '/paraell_'+str(self.sampling)+'_perceptual_attention_new_data_patchdiscrim_'+str(self.model)+str(self.knum)+'/'
        # self.path += '/paraell_5_attention_new_data_'+str(self.model)+str(self.knum)+'/'
        
        # self.path += '/paraell_5_'+str(self.model)+str(self.knum)+'/'
        

        self.imageDir = '../../mri_dataset/real_final_train/'
        
        #set a dataset ##
        # max_grad_norm = 2.
        self.gen_num = 10
        self.logger = Logger(self.path,batch_size=self.batch_size,delete=deleteall,num=str(self.knum),name=self.model+'_')
        
        
        self.device = torch.device('cuda:0'if torch.cuda.is_available() else "else")
        
    def load_model(self):

        if self.model == 'basic':
            print('----------generator---------')
            self.gen = reconstruction_resunet().to(self.device)
            self.regen = reconstruction_resunet().to(self.device)
            
            print('----------discriminator---------')
            self.dis = classification_discrim().to(self.device)
            
            self.batch_size=14
        elif self.model == 'effcientunet':
            print('----------generator---------')
            self.gen = reconstruction_efficientunet().to(self.device)
            self.regen = reconstruction_efficientunet().to(self.device)
            
            print('----------discriminator---------')
            # self.dis = classification_discrim().to(self.device)
            
            self.dis = reconstruction_discrim().to(self.device)
            self.batch_size=4
        elif self.model =='pyramid_unet':
            print('----------generator---------')
            self.gen = pyramid_unet().to(self.device)
            self.regen = pyramid_unet().to(self.device)
            
            print('----------discriminator---------')
            self.dis = classification_discrim().to(self.device)
            self.batch_size=6
        
        elif self.model =='deeplab':
            print('----------generator---------')
            self.gen = reconstruction_deeplab().to(self.device)
            self.regen = reconstruction_deeplab().to(self.device)
            
            print('----------discriminator---------')
            self.dis = classification_discrim().to(self.device)
            self.batch_size=8
        
        elif self.model =='attention_net':
            print('----------generator---------')
            self.gen = pyramid_unet().to(self.device)
            self.regen = pyramid_unet().to(self.device)
            
            print('----------discriminator---------')
            self.dis = classification_discrim().to(self.device)
            self.batch_size=8
        

        if self.model == 'patchgan':
            print('----------generator---------')
            self.gen = reconstruction_resunet().to(self.device)
            self.regen = reconstruction_resunet().to(self.device)
            
            print('----------discriminator---------')
            self.dis = reconstruction_discrim().to(self.device)
            # self.dis = classification_discrim().to(self.device)
            
            self.batch_size=4

        self.optimizerG = torch.optim.Adam(list(self.gen.parameters())+list(self.regen.parameters()),lr=self.learning_rate)
        self.optimizerD = torch.optim.Adam(self.dis.parameters(),lr=self.learning_rate)

        Tensor = torch.cuda.FloatTensor if self.device else torch.FloatTensor
        if self.use_scheduler == True:
            self.schedulerG = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizerG,100,T_mult=1,eta_min=self.end_lr)
            self.schedulerD = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizerD,100,T_mult=1,eta_min=self.end_lr)

        #set multigpu ##
        if self.parallel == True:
            self.gen = torch.nn.DataParallel(self.gen, device_ids=[0,1])    
            self.regen = torch.nn.DataParallel(self.regen, device_ids=[0,1])    
            
            self.dis = torch.nn.DataParallel(self.dis, device_ids=[0,1])    
            
    def load_loss(self):
        ###loss initilize
        
        TV_loss=TVLoss().to(self.device)
        
        self.WGAN_loss = WGANLoss(self.device)
        self.Lloss = torch.nn.L1Loss().to(self.device)
        self.Closs = torch.nn.L1Loss().to(self.device)
        self.Floss = torch.nn.L1Loss().to(self.device)
        self.TVloss = TV_loss.to(self.device)

        if self.perceptual_loss == True:
            vgg_ = Vgg16().to(self.device)
            for param in vgg_.parameters():
                param.requires_grad = False
            self.vgg_loss = create_perceptual_loss(vgg_,torch.nn.MSELoss(size_average=False).to(self.device))

        ###trainning###
    
    def trainning(self):


        self.load_model()
        self.load_loss()

        evalution =  recon_matrix()
        print('----- Dataset-------------')
        Dataset  = {'train': DataLoader(mydataset(self.imageDir,self.sampling,self.knum,True,self_supervised=True),
                                batch_size = self.batch_size,
                                shuffle = False,
                                num_workers=8),
                    'valid': DataLoader(mydataset(self.imageDir,self.sampling,self.knum),
                                batch_size = self.batch_size,
                                shuffle = True, 
                                num_workers = 4)}

        best_psnr = 0
        best_epoch = 0
        best_ssim = 0

        for epoch in range(self.epoches):
      
            #set loss weight
            ALPHA = 1e+1
            GAMMA = 1e+4
            DELTA = 1e-4
            normalizedImg = np.zeros((192,192))
            

            if epoch % self.validnum == 0:
                self.gen.eval()
                self.regen.eval()
                self.dis.eval()
                phase = 'valid'
                self.save_model('last',epoch)
                
                val_recon_psnr = 0
                val_final_psnr = 0
                val_recon_ssim = 0
                val_final_ssim = 0

                
            else :
                self.gen.train()
                self.regen.train()
                self.dis.train()
                phase = 'train'
                self.schedulerG.step(epoch)
                self.schedulerD.step(epoch)
                recon_psnr = 0
                final_psnr = 0
                recon_ssim = 0
                final_ssim = 0

            print(f"{epoch}/{self.epoches}epochs,IR=>{get_lr(self.optimizerG)},best_epoch=>{best_epoch},phase=>{phase}")
            print(f"==>{self.path}<==")      
            for i, batch in enumerate(tqdm(Dataset[phase])):
                if phase == 'train':
                    # set model inputa

                    _image = Variable(batch[0]).to(self.device)
                    mask = Variable(batch[1]).to(self.device)

                    mask2 = Variable(batch[2]).to(self.device)
                    
                    self.optimizerG.zero_grad()
                    # under_image,zero_image = apply_mask(_image,mask2)
                    
                    # _image=zero_image.float()
                    
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

                    ###refine gan ####
                    retrained_img = self.regen(final_img)
                    re_recon_img = torch.add(retrained_img, zero_image).float().to(self.device)
                    re_final_img = update(re_recon_img, _image,mask).to(torch.float32).to(self.device)


                    ###########total generate loss ##############
                    #image losses
                    recon_loss = self.Lloss(_image,recon_img)
                    error_loss = self.Closs(_image,final_img)       
                    
                    re_recon_loss = self.Lloss(_image,re_recon_img)
                    re_error_loss = self.Closs(_image,re_final_img)
                    #compare frequecy image
                    #frequency loss
                    freq_img,_ = apply_mask(final_img,mask)
                    re_freq_img,_ = apply_mask(re_final_img,mask)

                    freq_loss = self.Floss(under_image.to(torch.float32),freq_img).to(torch.float32)
                    re_freq_loss = self.Floss(under_image.to(torch.float32),re_freq_img).to(torch.float32)
                    
                    #WGan gen loss
                    boost_dis_fake = self.dis(final_img[:,0:1])
                    re_boost_dis_fake = self.dis(re_final_img[:,0:1])

                    boost_fake_A=self.WGAN_loss.loss_gen(boost_dis_fake)
                    re_boost_fake_A=self.WGAN_loss.loss_gen(re_boost_dis_fake)

                    #add regulization (total variation)
                    
                    TV_a=self.TVloss(final_img).to(torch.float32)
                    re_TV_a = self.TVloss(re_final_img).to(torch.float32)

                    #reconstruction loss
                    boost_R_loss = (recon_loss + error_loss* 10.0  + re_recon_loss + re_error_loss* 10.0) * 10.0 + (freq_loss + re_freq_loss)
                    boost_R_loss += (boost_fake_A + re_boost_fake_A)* 10.0
                    boost_R_loss += (TV_a + re_TV_a)
                    
                    loss_g = (boost_R_loss)
                    if self.perceptual_loss == True:
                        vgg_recon_loss = self.vgg_loss(_image,recon_img)
                        vgg_error_loss = self.vgg_loss(_image,final_img)       
                        
                        vgg_re_recon_loss = self.vgg_loss(_image,re_recon_img)
                        vgg_re_error_loss = self.vgg_loss(_image,re_final_img)
                        boost_R_loss += (vgg_error_loss+vgg_re_error_loss)

                    
                    loss_g.backward(retain_graph=True)
                    
                    self.optimizerG.step()

                    summary_val = {'recon_loss':recon_loss,
                                'error_loss':error_loss,
                                'freq_loss':freq_loss,
                                'TV_a':TV_a,
                                'boost_fake_A':boost_fake_A}


                    summary_val.update({'re_recon_loss':re_recon_loss,
                                're_error_loss':re_error_loss,
                                're_freq_loss':re_freq_loss,
                                're_TV_a':re_TV_a,
                                're_boost_fake_A':re_boost_fake_A})

                    if self.perceptual_loss == True:
                        summary_val.update({'vgg_error_loss':vgg_error_loss,
                                'vgg_re_error_loss':vgg_re_error_loss})
                        ###############train discrim#################
                        # self.gen_num
                    # if epoch > self.gen_num:

                    dis_real_img = self.dis(_image[:,0:1])
                    dis_fake_img = self.dis(final_img[:,0:1])
                    re_dis_fake_img = self.dis(re_final_img[:,0:1])
                    

                    self.optimizerD.zero_grad()

                    #calcuate loss function
                    dis_loss = self.WGAN_loss.loss_disc(dis_fake_img,dis_real_img)
                    re_dis_loss = self.WGAN_loss.loss_disc(re_dis_fake_img,dis_real_img)

                    # loss_RMSE   = Lloss(_image,final_image)            
                    GP_loss=compute_gradient_penalty(self.dis, _image[:,0:1], final_img[:,0:1],self.device) * 0.0001
                    re_GP_loss=compute_gradient_penalty(self.dis, _image[:,0:1], re_final_img[:,0:1],self.device) * 0.0001
                    
                    discrim_loss = (dis_loss + re_dis_loss) + (GP_loss + re_GP_loss)
                    discrim_loss.backward(retain_graph=True)

                    self.optimizerD.step()
                    summary_val.update({'dis_loss':dis_loss,'GP_loss':GP_loss})
                    
                    summary_val.update({'re_dis_loss':re_dis_loss,'re_GP_loss':re_GP_loss})
                
                    # final_img = cv2.normalize(final_img.cpu().detach().numpy(),  normalizedImg, 0, 1, cv2.NORM_MINMAX)
                    # recon_img = cv2.normalize(recon_img.cpu().detach().numpy(),  normalizedImg, 0, 1, cv2.NORM_MINMAX)
                    # _image = cv2.normalize(_image.cpu().numpy(),  normalizedImg, 0, 1, cv2.NORM_MINMAX)
                    
                    # re_final_img = cv2.normalize(re_final_img.cpu().detach().numpy(),  normalizedImg, 0, 1, cv2.NORM_MINMAX)
                    # re_recon_img = cv2.normalize(re_recon_img.cpu().detach().numpy(),  normalizedImg, 0, 1, cv2.NORM_MINMAX)
                    
                    # final_img = (final_img * 255.0).cpu().detach().numpy()
                    # recon_img = (recon_img * 255.0).cpu().detach().numpy()
                    # _image = (_image * 255.0).cpu().numpy()
                    
                    # re_final_img = (re_final_img * 255.0).cpu().detach().numpy()
                    # re_recon_img = (re_recon_img * 255.0).cpu().detach().numpy()
                    
                    final_img = cvt2imag(final_img).cpu().detach().numpy()
                    recon_img = cvt2imag(recon_img).cpu().detach().numpy()
                    _image= cvt2imag(_image).cpu().detach().numpy()
                    re_final_img = cvt2imag(re_final_img).cpu().detach().numpy()
                    re_recon_img = cvt2imag(re_recon_img).cpu().detach().numpy()


                    final_psnr = evalution.PSNR(final_img,_image,255.0)
                    recon_psnr = evalution.PSNR(recon_img,_image,255.0)
                    final_ssim = evalution.PSNR(re_final_img,_image,255.0)
                    recon_ssim = evalution.PSNR(re_recon_img,_image,255.0)
                    
                    
                else:
                    with torch.no_grad():

                        inputa, mask = batch[0].to(self.device), batch[1].to(self.device)
                        
                        inputa=inputa.float()
                        mask = mask.float()
                        
                        under_im,inputs = apply_mask(inputa,mask)
                        prediction = self.gen(inputs.float())
                        prediction = torch.add(inputs.float() , prediction)                         
                        pred = update(prediction,inputa,mask)

                        

                        re_prediction = self.regen(prediction.float())
                        
                        re_prediction = torch.add(inputs.float() , re_prediction)                         
                        
                        re_pred = update(re_prediction,inputa,mask)

                        recon_loss = self.Lloss(inputa,prediction) 
                        error_loss = self.Closs(inputa,pred)        
                        re_recon_loss = self.Lloss(inputa,re_prediction) 
                        re_error_loss = self.Closs(inputa,re_pred)        
                        
                        #compare frequecy image
                        #frequency loss

                        freq_img,_ = apply_mask(pred,mask)
                        freq_loss = self.Floss(under_im.to(torch.float32),freq_img).to(torch.float32)

                        re_freq_img,_ = apply_mask(re_pred,mask)
                        re_freq_loss = self.Floss(under_im.to(torch.float32),re_freq_img).to(torch.float32)


                        summary_val = {'recon_loss':recon_loss,
                            'error_loss':error_loss,
                            'freq_loss':freq_loss}
            

                        summary_val.update({'re_recon_loss':re_recon_loss,
                            're_error_loss':re_error_loss,
                            're_freq_loss':re_freq_loss})
            

                        #calcuate matrix
                        # prediction = cv2.normalize(prediction.cpu().numpy(),  normalizedImg, 0, 1, cv2.NORM_MINMAX)
                        # pred = cv2.normalize(pred.cpu().numpy(),  normalizedImg, 0, 1, cv2.NORM_MINMAX)
                        # inputa = cv2.normalize(inputa.cpu().numpy(),  normalizedImg, 0, 1, cv2.NORM_MINMAX)

                        # re_prediction = cv2.normalize(re_prediction.cpu().numpy(),  normalizedImg, 0, 1, cv2.NORM_MINMAX)
                        # re_pred = cv2.normalize(re_pred.cpu().numpy(),  normalizedImg, 0, 1, cv2.NORM_MINMAX)

                        # prediction = (prediction  * 255.0).cpu().detach().numpy()
                        # pred = (pred * 255.0).cpu().detach().numpy()
                        # inputa = (inputa * 255.0).cpu().numpy()

                        # re_prediction = (re_prediction * 255.0).cpu().detach().numpy()
                        # re_pred = (re_pred * 255.0).cpu().detach().numpy()
                    
                        prediction = cvt2imag(prediction).cpu().detach().numpy()
                        pred = cvt2imag(pred).cpu().detach().numpy()
                        inputa= cvt2imag(inputa).cpu().detach().numpy()
                        re_prediction = cvt2imag(re_prediction).cpu().detach().numpy()
                        re_pred = cvt2imag(re_pred).cpu().detach().numpy()
                        
                        val_final_psnr = evalution.PSNR(prediction,inputa,255.0)
                        val_recon_psnr = evalution.PSNR(pred,inputa,255.0)
                        val_final_ssim = evalution.PSNR(re_prediction,inputa,255.0)
                        val_recon_ssim = evalution.PSNR(re_pred,inputa,255.0)

            if phase == 'train':
                i=float(i)     
                evalutiondict = {'final_psnr':final_psnr,'recon_psnr':recon_psnr,
                                    'final_ssim':final_ssim,'recon_ssim':recon_ssim}
                                    
                summary_val.update(evalutiondict)
                self.printall(summary_val,epoch,'train')
                
                if (evalutiondict['final_psnr'] > best_psnr) or (evalutiondict['final_ssim'] > best_ssim):
                    self.save_model('bestsave_model',epoch)
                    best_psnr = evalutiondict['final_psnr']
                    best_epoch = epoch
                    best_ssim = evalutiondict['final_ssim']

            else:      
                i=float(i)
            
                evalutiondict = {'val_final_psnr':val_final_psnr,'val_recon_psnr':val_recon_psnr,
                                    'val_final_ssim':val_final_ssim,'val_recon_ssim':val_recon_ssim}
                summary_val.update(evalutiondict)

                self.printall(summary_val,epoch,'valid')

                # class_list=['recon_psnr','final_psnr','recon_ssim','final_ssim']
                # class_list,evalutiondict = self.logger.convert_to_list(evalutiondict)
                # print(class_list,evalutiondict)
                # df=pd.DataFrame(evalutiondict)
                self.logger.save_csv_file(evalutiondict,'test')

                total_image = {'recon_image':prediction,
                                'final_image':pred,
                                'input_image':inputa}
                
                total_image.update({'re_recon_image':re_prediction,
                                're_final_image':re_pred})
                

                self.re_normalize(total_image,255)
                total_image.update({'zero_image':inputs.cpu().numpy()})

                recon_error_img = total_image['input_image'] - total_image['recon_image']
                final_error_img = total_image['input_image'] - total_image['final_image']
                zero_error_img = total_image['input_image'] - total_image['zero_image']

                re_recon_error_img = total_image['input_image'] - total_image['re_recon_image']
                re_final_error_img = total_image['input_image'] - total_image['re_final_image']
                

                total_image.update({'recon_error_img':recon_error_img,
                                    'final_error_img':final_error_img,
                                    'zero_error_img':zero_error_img,
                                    'mask':mask.cpu().numpy()})
                
                total_image.update({'re_recon_error_img':re_recon_error_img,
                                    're_final_error_img':re_final_error_img})
                                    
                self.logger.save_images(total_image,epoch)
                            # avg_ssim += ssim
                                # z_avg_ssim += z_ssim
                                # print(z_ssim.shape)
                
    def printall(self,summary_val,epoch,name):
        self.logger.print_value(summary_val,name)
        self.logger.summary_scalars(summary_val,epoch,name)


    def re_normalize(self,convert_dict_image,max_value):
        normalizedImg = np.zeros((192,192))
        norm = True
        if norm == True:
            for i, scalar in enumerate(convert_dict_image):
                # print(i,scalar)
                convert_dict_image[scalar] = cv2.normalize(convert_dict_image[scalar],  normalizedImg, 0, max_value, cv2.NORM_MINMAX)
        elif norm == False:
            for i, scalar in enumerate(convert_dict_image):
                # print(i,scalar)
                convert_dict_image[scalar] = cv2.normalize(convert_dict_image[scalar].cpu().numpy(),  normalizedImg, 0, max_value, cv2.NORM_MINMAX)

                # self.writer.add_scalar(str(tag)+'/'+str(scalar),scalar_dict[scalar],step)

    def save_model(self,name,epoch):
        # self.logger.save_model(self.gen,self.dis,epoch,self.path)

        if self.parallel == False:
            torch.save({"gen_model":self.gen.state_dict(),
                        "regen_model":self.regen.state_dict(),
                        
                        "dis_model":self.dis.state_dict(),
                        "optimizerG":self.optimizerG.state_dict(),
                        "optimierD":self.optimizerD.state_dict(),
                        "epoch":epoch},
                        self.path+str(name)+"_save_models{}.pth")
        elif self.parallel == True:
            torch.save({"gen_model":self.gen.module.state_dict(),
                        "regen_model":self.regen.module.state_dict(),
                        
                        "dis_model":self.dis.module.state_dict(),
                        "optimizerG":self.optimizerG.state_dict(),
                        "optimierD":self.optimizerD.state_dict(),
                        "epoch":epoch},
                        self.path+str(name)+"_save_models{}.pth")

    def load_save_model(self):
        
        checkpoint = torch.load(self.path+"/last_save_models{}.pth")
        self.gen.load_state_dict(checkpoint['gen_model'])
        self.regen.load_state_dict(checkpoint['regen_model'])
        # self.dis.load_state_dict(checkpoint['dis_model'])

    def testing(self):
        self.parallel = False
        
        self.load_model()
        self.load_save_model()
        self.logger.changedir()

        t_imageDir = '../../mri_dataset/real_final_test/'
        t_labelDir = '../../mri_dataset/real_final_test/'
        
        Dataset  = { 'valid': DataLoader(mydataset(self.imageDir,2.2,kfold=False),
                                batch_size = 1,
                                shuffle = False, 
                                num_workers = 8),
                    'test': DataLoader(mydataset(t_imageDir,2.2,kfold=False),
                                batch_size = 1,
                                shuffle = False, 
                                num_workers = 8)}

                # Dataset  = {'train': DataLoader(mydataset(self.imageDir,self.sampling,self.knum,True),
                #                 batch_size = self.batch_size,
                #                 shuffle = True,
                #                 num_workers=8),
                #     'valid': DataLoader(mydataset(self.imageDir,self.sampling,self.knum),
                #                 batch_size = self.batch_size,
                #                 shuffle = True, 
                #                 num_workers = 4)}


        phase = 'test'
        test_final_psnr = 0
        test_recon_psnr = 0
        test_final_ssim = 0
        test_recon_ssim = 0
        test_evalution =  recon_matrix()
        new_evalutaion = []

        for i, batch in enumerate(tqdm(Dataset[phase])):
            with torch.no_grad():
                self.gen.eval()
                self.regen.eval()
                self.dis.eval()

                inputa, mask = batch[0].to(self.device), batch[1].to(self.device)
                inputa=inputa.float()
                mask = mask.float()
                
                under_im,zeroimg = apply_mask(inputa,mask)
                
                prediction = self.gen(zeroimg.float())
                
                prediction = torch.add(zeroimg.float() , prediction)                            
                pred = update(prediction,inputa,mask)

                re_prediction = self.regen(pred.float())
                re_prediction = torch.add(zeroimg.float() , re_prediction)                            
                re_pred = update(re_prediction,inputa,mask)


                #calcuate matrix
                # prediction = cv2.normalize(prediction.cpu().numpy(),  normalizedImg, 0, 1, cv2.NORM_MINMAX)
                # pred = cv2.normalize(pred.cpu().numpy(),  normalizedImg, 0, 1, cv2.NORM_MINMAX)
                # inputa = cv2.normalize(inputa.cpu().numpy(),  normalizedImg, 0, 1, cv2.NORM_MINMAX)
                
                # prediction = (prediction * 255.0).cpu().detach().numpy()
                # pred = (pred * 255.0).cpu().detach().numpy()
                # inputa = (inputa * 255.0).cpu().numpy()
                
                # re_prediction = (re_prediction * 255.0).cpu().detach().numpy()
                # re_pred = (re_pred * 255.0).cpu().detach().numpy()
                
                prediction = cvt2imag(prediction).cpu().detach().numpy()
                pred = cvt2imag(pred).cpu().detach().numpy()
                inputa= cvt2imag(inputa).cpu().detach().numpy()
                re_prediction = cvt2imag(re_prediction).cpu().detach().numpy()
                re_pred = cvt2imag(re_pred).cpu().detach().numpy()
                
                test_final_psnr = test_evalution.PSNR(prediction,inputa,255.0)
                test_recon_psnr = test_evalution.PSNR(pred,inputa,255.0)
                test_final_ssim = test_evalution.PSNR(re_prediction,inputa,255.0)
                test_recon_ssim = test_evalution.PSNR(re_pred,inputa,255.0)
                    
        
                # i=float(i)
                evalutiondict = {'val_final_psnr':test_final_psnr,'val_recon_psnr':test_recon_psnr,
                                    'val_final_ssim':test_final_ssim,'val_recon_ssim':test_recon_ssim}
                
                # class_list=['recon_psnr','final_psnr','recon_ssim','final_ssim']
                class_list,evalutiondict = self.logger.convert_to_list(evalutiondict)
                new_evalutaion.append(list(evalutiondict))

                total_image = {'recon_image':prediction,
                                'final_image':pred,
                                'input_image':inputa}
                total_image.update({'re_prediction':re_prediction,
                                    're_pred':re_pred})

                total_image.update({'zero_image':zeroimg.cpu().numpy()})
                self.re_normalize(total_image,255)
                
                recon_error_img = np.abs(total_image['input_image'] - total_image['recon_image'])
                final_error_img = np.abs(total_image['input_image'] - total_image['final_image'])
                zero_error_img = np.abs(total_image['input_image'] - total_image['zero_image'])

                total_image.update({'recon_error_img':recon_error_img,
                                    'final_error_img':final_error_img,
                                    'zero_error_img':zero_error_img,
                                    'mask':mask.cpu().numpy()})
                
                self.logger.save_images(total_image,i)

        self.logger.save_csv_file(np.array(new_evalutaion),'valid',list(class_list))

import argparse

parser = argparse.ArgumentParser(description='Process some integers')
parser.add_argument('--knum',default=1, help='Select Dataset',type=int)
parser.add_argument('--gpu', default=0,help='comma separated list of GPU(s) to use.',type=int)
parser.add_argument('--sampling',default=2.2, help='set sampling mask 100/(value)',type=float)

args = parser.parse_args()

reconstruction = main(args)
reconstruction.trainning()
reconstruction.testing()