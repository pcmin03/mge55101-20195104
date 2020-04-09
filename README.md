# mge55101-20195104

**Introduction to Deep Learning Repository**  
Name : Chanmin Park  
Student No : 20195104  
School : computer engineering  
E-mail : pcmin03@unist.ac.kr  

# Project Name : medical image reconstruction using Deep learning

### Introduction  
In medical images, MRI, CT, ultrasound, and cell are widely used. Using this image , many people were developing various method such as a finding cancer that doctors cannot find using cancer, counting the number of cells, and segmentating only necessary parts of a medical image, etc.    
And among many technologies, I would like to make a medical image reconstruction that lost the spatial information. Because MRI takes a long time to acquire images. for resolving this problem, in the previous study, the performance of the hardware was improved or changing the method for improving the acquire speed. In this methodological method, a restoring image after obtaining a damaged image is widely used for example iterative reconstruction compressed sensing.  
In particular, in this semester, I would like to apply reconstruction using deep learning of dynamic MRI dataset wthich has temporal and spatial features.  

![MRIimage](/images/T1t2PD.jpg "Kind of MRI images")  
_Kind of MRI images_
 

### Related work  
MRI image restoration has been studied a lot before. previously,the MRI was reconsturcted using the [Nyquist–Shannon Sampling Therorem](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem). recently, The state of the art method was the [compression sensing method](https://en.wikipedia.org/wiki/Compressed_sensing), machine learning method like [sparse coding](https://en.wikipedia.org/wiki/Convolutional_Sparse_Coding) However, in compress sensing method, there was a disadvantage in that the time to acquire the image and the required signal were randomly selected. To solve this problem, people recently tried to solve the problem with deep learning, and representatively, they improved the image using CNN, GAN, etc.   furthermore, in dynamic MRI dataset such as cardiac dataset recently are using RNN and 

![RefineGAN](/images/Overview.png "Deep learning using GAN named RefineGAN")  
_RefineGAN paper : https://arxiv.org/abs/1709.00753_  


### plan
This semester, I will plan for two months.  
The detailed plan is attached to the picture chart below.  

![Plan](/images/Picture3.png)  
_Detail Plan_

### Data preparation 
Usually, many MRI reconstruction uses static datasets such as brain image, Leg bone. however the cardiac image is a dynamic dataset, it is hard to reconstruct. In this project. I'll use the cardiac dataset of [Second Annual Data Science Bowl](https://www.kaggle.com/c/second-annual-data-science-bowl).  it is consists of DICOM dataset which has 192 X 256 X 30 stack image. There has a total of 100 patients which has 30 stack image. this semenster, I using 20 patients to use the training dataset, 4 patients for validation dataset, last 8 patients to test dataset. So the total training dataset is 600(20X30)images, validation is 120(4X40)images, and the test is 240(8X30) images.
![Example Cardiacdataset](/images/IM-9523-00012.png)(/images/IM-9930-0026.png)
Example Cardiacdataset

### Evaluation Matrix
The mostly ill-posed problem has used to evaluate two matrices each of Structural Similarity Index(SSIM), Peak signal-to-noise ratio(PSMR).
SSIM matrix usually compares brightness, structure, and contrast compare to the original image to result from an image. So it shows the bottom equation.   
$$
<a href="https://www.codecogs.com/eqnedit.php?latex=SSIM(x,y)&space;=&space;\frac{(2\mu_{x}\mu_{y}&plus;c_{1})(2\sigma_{xy}&plus;c_{2})}{(2\mu_{x}^2&plus;\mu_{y}^2&plus;c_{2})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?SSIM(x,y)&space;=&space;\frac{(2\mu_{x}\mu_{y}&plus;c_{1})(2\sigma_{xy}&plus;c_{2})}{(2\mu_{x}^2&plus;\mu_{y}^2&plus;c_{2})}" title="SSIM(x,y) = \frac{(2\mu_{x}\mu_{y}+c_{1})(2\sigma_{xy}+c_{2})}{(2\mu_{x}^2+\mu_{y}^2+c_{2})}" /></a>

- <img src="https://latex.codecogs.com/gif.latex?SSIM(x,y) = \frac{(2\mu_{x}\mu_{y}+c_{1})(2\sigma_{xy}+c_{2})}{(2\mu_{x}^2+\mu_{y}^2+c_{2})}" /> 
$$  
Furthermore, PSNR is mainly used to evaluate image quality loss information in video or video loss compression. Matrix is easy to compare two images, it uses MeanSqure error(MSE) divide 'RMSE'.  
$$
<a href="https://www.codecogs.com/eqnedit.php?latex=SSIM(x,y)&space;=&space;\frac{(2\mu_{x}\mu_{y}&plus;c_{1})(2\sigma_{xy}&plus;c_{2})}{(2\mu_{x}^2&plus;\mu_{y}^2&plus;c_{2})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?SSIM(x,y)&space;=&space;\frac{(2\mu_{x}\mu_{y}&plus;c_{1})(2\sigma_{xy}&plus;c_{2})}{(2\mu_{x}^2&plus;\mu_{y}^2&plus;c_{2})}" title="SSIM(x,y) = \frac{(2\mu_{x}\mu_{y}+c_{1})(2\sigma_{xy}+c_{2})}{(2\mu_{x}^2+\mu_{y}^2+c_{2})}" /></a>
PSNR = 10log_{10}(\frac{MAX^2_{I}}{MSE}) = 20log_{10}(\frac{MAX_{I}}{\sqrt{MSE}}})  
$$  
lastly, I'll use two martix to compare state-of-the-art dynamic Cardiac paper name "Dynamic MRI Reconstruction with Motion-Guided Network"(link : https://openreview.net/pdf?id=Bke-CJtel4).
![My_networkplan](/images/network.png)  
