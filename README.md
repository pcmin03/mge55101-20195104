# mge55101-20195104

**Introduction to Deep Learning Repository**  
Name : Chanmin Park  
Student No : 20195104  
School : computer engineering  
E-mail : pcmin03@unist.ac.kr  

*This semester final Project   
**medical image reconstruction using Deep learning  

***Introduction  
In medical images, MRI, CT, ultrasound, and cell are widely used. Using this image , many people were developing various method such as a finding cancer that doctors cannot find using cancer, counting the number of cells, and segmentating only necessary parts of a medical image, etc.    
And among many technologies, I would like to make a medical image reconstruction that lost the spatial information. Because MRI takes a long time to acquire images. for resolving this problem, in the previous study, the performance of the hardware was improved or changing the method for improving the acquire speed. In this methodological method, a restoring image after obtaining a damaged image is widely used for example iterative reconstruction compressed sensing.  
In particular, in this semester, I would like to apply reconstruction using deep learning of dynamic MRI dataset wthich has temporal and spatial features.  

![MRIimage](/images/T1t2PD.jpg)
 

***Related work  
MRI image restoration has been studied a lot before. previously,the MRI was reconsturcted using the [Nyquist–Shannon Sampling Therorem](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem). recently, The state of the art method was the [compression sensing method](https://en.wikipedia.org/wiki/Compressed_sensing), machine learning method like [sparse coding](https://en.wikipedia.org/wiki/Convolutional_Sparse_Coding) However, in compress sensing method, there was a disadvantage in that the time to acquire the image and the required signal were randomly selected. To solve this problem, people recently tried to solve the problem with deep learning, and representatively, they improved the image using CNN, GAN, etc.   furthermore, in dynamic MRI dataset such as cardiac dataset recently are using RNN and 

![RefineGAN](/images/Overview.png)

***plan
This semaster, I will plan the during two month.  
The detailed plan is attached in the picture chart below.

