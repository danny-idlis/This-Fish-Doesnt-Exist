# This Fish Doesn't Exist

> _A live runnable version of this project is available at: https://www.kaggle.com/dannyidlis/notebooks_  

### Project Description:
This project is a spin-off version of the famous [This Person Doesn't Exist](https://www.thispersondoesnotexist.com) project. </br></br>
It was made by Danny Idlis, Tomer Yacov, Sapir Musacho & Adi Knafo on April 2020.  
This was our final project for the Advanced Topics in Deep Learning course in the College of Management.  
Our job was to create a network based on the GAN architecture and try to create fish images.  
Each member of the team made his own experiments with different architectures and parameters.  
The code in this git repo is my personal part of the project which includes 3 experiments: </br>
1. A shallow version of a DCGAN  
2. A deep version of a DCGAN
3. A deep version of a DCGAN with Feature Matching

### Results:

Final results as long as full documentation & paperwork can be found [here](./This-Fish-Does-Not-Exist.pdf)

![Shallow DCGAN Final Results](./v1/results/shallow-dcgan/images/image_at_epoch_0405.png?raw=true "Shallow DCGAN Final Results")
![Deep DCGAN Final Results](./v1/results/deep-dcgan/images/image_at_epoch_0405.png?raw=true "Deep DCGAN Final Results")
![Deep DCGAN with Feature Matching Final Results](./v1/results/deep-dcgan-with-feature-matching/images/image_at_epoch_0405.png?raw=true "Deep DCGAN with Feature Matching Final Results")

### Local Installation Prerequisites:
* Python 3.6
* CUDA 10.1
* Latest CUDNN that's compatible with CUDA 10.1
	* CUDNN v7.6.5 (November 5th, 2019) as for 26.5.2020