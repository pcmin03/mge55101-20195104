
from scipy.ndimage import uniform_filter, gaussian_filter
import numpy as np_complex
import math
import cv2
import numpy as np

class recon_matrix(object):

#################################################################
#                            PSNR                               #
#################################################################
    # def normalize(self,image1,image2,max_val)
    #     image1.cpu.numpy(),image2,
    def np_complex(self,data):
        real  = data[...,0:1]
        imag  = data[...,1:2]
        del data
        data = real + 1j*imag
        return data

    def PSNR(self,img1,img2,max):
        total_psnr = 0
        for i in range(len(img1[:])):
            img_1=img1[i]
            img_2=img2[i]
            bat = i
            for j in range(len(img_1[:])):
                img__1=self.np_complex(img_1[j]).astype(np.float64)
                img__2=self.np_complex(img_2[j]).astype(np.float64)
        
                num_frame = j
                # psnr=skimage.measure.compare_psnr(img__1,img__2,max)
                mse = np.mean((img__1 - img__2) ** 2 )
                psnr = 20*math.log10(max/math.sqrt(mse))
                total_psnr  +=psnr

        return total_psnr / ((bat+1) * (num_frame+1))
    #################################################################
    #                            SSIM                               #
    #################################################################
    
    def SSIM(self,img1,img2,max):
        total_ssim = 0
        L = max
        C1 = (0.01*L)**2
        C2 = (0.03*L)**2
        for i in range(len(img1[:])):
            img_1=img1[i]
            img_2=img2[i]
            bat = i
            for j in range(len(img_1[:])):
                img__1=self.np_complex(img_1[j]).astype(np.float64)
                img__2=self.np_complex(img_2[j]).astype(np.float64)
        
                num_frame = j
                # psnr=skimage.measure.compare_psnr(img__1,img__2,max)
                mse = np.mean((img__1 - img__2) ** 2 )
                
                mu1 = uniform_filter(img__1)
                mu2 = uniform_filter(img__2)

                mu_img1_2 = uniform_filter(img__1*img__2)
                mu_img1 = uniform_filter(img__1**2)
                mu_img2 = uniform_filter(img__2**2)

                var_img1_2 = mu_img1_2 - mu1*mu2
                var_img1   = mu_img1 - mu1**2 
                var_img2   = mu_img2 - mu2**2

                numerator = (2*mu_img1_2 + C1)*(2*var_img1_2+C2)
                denominator = (mu_img1 + mu_img2 + C1)*(var_img1 + var_img2 + C2)
                ssim = numerator/denominator
                
                ssim = skimage.measure.compare_ssim(img__1,img__2,max)
                # mu_conv = image_1.
                print(ssim)
                # ssim = img__1.mean()*img__2.maen()
                total_ssim  +=ssim

        return total_ssim / ((bat+1) * (num_frame+1))