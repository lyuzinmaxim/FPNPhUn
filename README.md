# FPNPhUn
This is my PyTorch implementation of DL model solving phase unwrapping problem based on FPN [1-2]

# 1st attempt

Original structure of U-Net-like model [2] was implemented in PyTorch, but due to the low spatial resolution it's impossible to use it in phase unwrapping tasks.
Structure from [2] is shown below.

![Seferbekov](https://user-images.githubusercontent.com/73649419/116997820-94417600-acdd-11eb-97f7-a376d0444b3a.jpg)

# 2nd attempt

Original structure of U-Net-like model [2] assumes that first feature map after input image is 4 times smaller - but I suppose for phase unwrapping it's crucial not to lose spatial information by double maxpool. 
So now I try to increase spatial resolution of all model (the smallest feature maps BxCx8x8 now are BxCx16x16)

Experiments say, that

So like it's shown below my output (in [2] output size was 4 times smaller than input - and I did bilinear interpolation) neural net output is pretty good, but due to low spatial resolution there are linear segments on the output map. 

![output1](https://user-images.githubusercontent.com/73649419/117002845-60b61a00-ace4-11eb-8ad2-c72b4c55ae72.jpg)



# References
1. Lin, Tsung-Yi, et al. "Feature pyramid networks for object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
2. Seferbekov, Selim, et al. "Feature pyramid network for multi-class land segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2018.
