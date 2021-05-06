# FPNPhUn
This is my PyTorch implementation of DL model solving phase unwrapping problem based on FPN [1-2]

# 1st attempt

Original structure of U-Net-like model [2] was implemented in PyTorch, but due to the low spatial resolution it's impossible to use it in phase unwrapping tasks.
Structure from [2] is shown below.

![Seferbekov](https://user-images.githubusercontent.com/73649419/116997820-94417600-acdd-11eb-97f7-a376d0444b3a.jpg)


So like it's shown below my output (in [2] output size was 4 times smaller than input - and I did bilinear interpolation) neural net output is pretty good, but due to low spatial resolution there are linear segments on the output map. 

![output1](https://user-images.githubusercontent.com/73649419/117002845-60b61a00-ace4-11eb-8ad2-c72b4c55ae72.jpg)

# 2nd attempt

Original structure of U-Net-like model [2] assumes that first feature map after input image is 4 times smaller - but I suppose for phase unwrapping it's crucial not to lose spatial information by double maxpool. 
So now I try to increase spatial resolution of all model (the smallest feature maps BxCx8x8 now are BxCx16x16)

Experiments on the same small dataset (20 different phase images) say, that both models have the same nature of the losses decrease (averaged over 5 attempts with fixed random seed)

![image](https://user-images.githubusercontent.com/73649419/117153386-34210180-adbb-11eb-9533-2ff1fcaeb155.png)


But 2nd model achieves more accuracy at the same time - maybe because spatial resolution doesn't decreases do fast (double maxpool) (averaged over 5 attempts with fixed random seed)

![image](https://user-images.githubusercontent.com/73649419/117153578-67fc2700-adbb-11eb-859e-3209325866b2.png)

# 3rd attempt

According to original paper about different residual connections [3] best residual structure to use for classification tasks is:

![image](https://user-images.githubusercontent.com/73649419/117353301-403abb00-aeb0-11eb-8d66-127dc9ef09e0.png)

And best results can be reached using following activation/weights/norm architecture (called "full pre-activation"):

![image](https://user-images.githubusercontent.com/73649419/117353643-a7f10600-aeb0-11eb-939a-d30b24469736.png)


# References
1. Lin, Tsung-Yi, et al. "Feature pyramid networks for object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
2. Seferbekov, Selim, et al. "Feature pyramid network for multi-class land segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2018.
3. He, Kaiming, et al. "Identity mappings in deep residual networks." European conference on computer vision. Springer, Cham, 2016.
