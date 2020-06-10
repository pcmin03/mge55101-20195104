
from scipy.ndimage import uniform_filter, gaussian_filter
import numpy as np_complex
import math
import cv2
import numpy as np
import torch
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

    def psnr(self,img1,img2,max):
        """Peak Signal to Noise Ratio
        img1 and img2 have range [0, 255]"""

        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(max / torch.sqrt(mse))


# class SSIM:
#     """Structure Similarity
#     img1, img2: [0, 255]"""

#     def __init__(self):
#         self.name = "SSIM"

#     @staticmethod
#     def __call__(img1, img2):
#         if not img1.shape == img2.shape:
#             raise ValueError("Input images must have the same dimensions.")
#         if img1.ndim == 2:  # Grey or Y-channel image
#             return self._ssim(img1, img2)
#         elif img1.ndim == 3:
#             if img1.shape[2] == 3:
#                 ssims = []
#                 for i in range(3):
#                     ssims.append(ssim(img1, img2))
#                 return np.array(ssims).mean()
#             elif img1.shape[2] == 1:
#                 return self._ssim(np.squeeze(img1), np.squeeze(img2))
#         else:
#             raise ValueError("Wrong input image dimensions.")

#     @staticmethod
#     def _ssim(img1, img2):
#         C1 = (0.01 * 255) ** 2
#         C2 = (0.03 * 255) ** 2

#         img1 = img1.astype(np.float64)
#         img2 = img2.astype(np.float64)
#         kernel = cv2.getGaussianKernel(11, 1.5)
#         window = np.outer(kernel, kernel.transpose())

#         mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#         mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#         mu1_sq = mu1 ** 2
#         mu2_sq = mu2 ** 2
#         mu1_mu2 = mu1 * mu2
#         sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
#         sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
#         sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

#         ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
#             (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
#         )
#         return ssim_map.mean()