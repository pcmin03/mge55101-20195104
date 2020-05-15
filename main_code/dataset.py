import io
import numpy as np 
from glob import glob
from natsort import natsorted

from utils import *
from sklearn.model_selection import KFold
import skimage
#################################################################
#                         data load                             #
#################################################################
from natsort import natsorted
class mydataset(Dataset):
    def __init__(self,imageDir,size,fold_num=1,trainning = False):

        self.size = size
        
        #kfold (cross validation)
        images = np.array(natsorted(glob(imageDir+'*')))
        
        
        kfold = KFold(n_splits=9)

        train = dict()
        label  = dict()
        i = 0
        for train_index, test_index in kfold.split(images):
            img_train,img_test = images[train_index], images[test_index]
            i+=1
            train.update([('train'+str(i),img_train),('test'+str(i),img_test)])
            
        train_num, test_num = 'train'+str(fold_num), 'test'+str(fold_num)
        
    
        if trainning == True:
            self.images = train[train_num]
            print(f"img_train:{len(img_train)} \t CV_train:{train_num} ")
        else:
            self.images = train[test_num]
            print(f"img_test:{len(img_test)} \t CV_test:{test_num}")

    def normal_pdf(self,length, sensitivity):
        return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)

    def cartesian_mask(self,shape, acc, sample_n=10, centred=False):
        """
        Sampling density estimated from implementation of kt FOCUSS
        shape: tuple - of form (..., nx, ny)
        acc: float - doesn't have to be integer 4, 8, etc..
        """
        N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
        pdf_x = self.normal_pdf(Nx, 0.5/(Nx/10.)**2)
        lmda = Nx/(2.*acc)
        n_lines = int(Nx / acc)
        # add uniform distribution
        pdf_x += lmda * 1./Nx

        if sample_n:
            pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
            pdf_x /= np.sum(pdf_x)
            n_lines -= sample_n

        mask = np.zeros((N, Nx))
        for i in range(N):
            idx = np.random.choice(Nx, n_lines, False, pdf_x)
            mask[i, idx] = 1

        if sample_n:
            mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

        size = mask.itemsize
        mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

        mask = mask.reshape(shape)

        if not centred:
            mask = np.fft.ifftshift(mask, axes=(-2,-1))

        return mask   

    def random_flip(self,image, seed=None):
        # assert image.ndim == 5
        if seed:
            np.random.seed(seed)
        random_flip = np.random.randint(1,5)

        if random_flip==1:
            flipped = image[::1,::-1,:]
            image = flipped
        elif random_flip==2:
            flipped = image[::-1,::1,:]
            image = flipped
        elif random_flip==3:
            flipped = image[::-1,::-1,:]
            image = flipped
        elif random_flip==4:
            flipped = image
            image = flipped
        return image

    def random_square_rotate(self,image, seed=None):
        # assert image.ndim == 5
        if seed:
            np.random.seed(seed)        
        random_rotatedeg = 90*np.random.randint(0,4)
        rotated = image 
        from scipy.ndimage.interpolation import rotate
        if image.ndim==4:
            rotated = rotate(image, random_rotatedeg, axes=(1,2))
        elif image.ndim==3:
            rotated = rotate(image, random_rotatedeg, axes=(0,1))
        image = rotated
        return image

    def random_crop(self,image, seed=None):

        if seed:
            np.random.seed(seed)        
        limit = np.random.randint(10,12) # Crop pixel
        randy = np.random.randint(0, limit)
        randx = np.random.randint(0, limit)
        cropped = image[..., randx:-(limit-randx), randx:-(limit-randx),:]

        a=len(cropped[0,:,0])
        d=len(image[0,:,0])

        cropped=rescale(cropped,(d/a))
        return cropped

    def transform(self,img):
        seed = np.random.randint(0, 2019)
        img = self.random_flip(img,seed=seed)
        img = self.random_square_rotate(img,seed=seed)
        
        # img = self.random_crop(img,seed=seed)
        
        return img

    def __len__(self):
        if self.size==1:
            return len(self.images)
        else:
            return len(self.images)
    

    def __getitem__(self,index):

        image = skimage.io.imread(self.images[index])
        mask  = self.cartesian_mask((1,len(image[:]),len(image[:])),self.size,sample_n=8)[0]

        image = image.astype(np.uint8)
        #make imaginary channel & real channel 
        image = np.stack((image, np.zeros_like(image)), axis=0)
        mask_s = np.stack((mask, np.zeros_like(mask)), axis=0)
        

        real_images = np.array(image)
        mask_image = np.array(mask_s)
        img=cvt2tanh(real_images,(1,2))
        
        mas = mask_image

        return img, mas