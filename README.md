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

So losses for old residual block and new for 30 epochs on the same small dataset (20 different phase images) averaged over 5 attempts with fixed random seed are:

![image](https://user-images.githubusercontent.com/73649419/117362551-af69dc80-aebb-11eb-944f-061033bb2fb2.png)

And metrics are:

![image](https://user-images.githubusercontent.com/73649419/117362663-d6281300-aebb-11eb-89f3-fbcf7885105f.png)

From this experiment it's clear (mainly due to metric comparison), that net with "new" residual blocks is learning faster.

# 4-th attempt

Now I would change the layer structure of neural net. The more trainable parameters does the net have, the more complicated patterns can be learned, but the more likely the network can be overfitted. For comparison, encoder-decoder unwrapping net DLPU [4] has 1.824.937 trainable parameters, PhUn net [5] has 90.120 trainable parameters, 
VUR-Net [6] has 21.561.430 parameters, and proposed FPNPhUn in current implementation has 78.080.528 parameters. So, most parameters belong to "lower" layers - there are connections from 1024 to 2024 layers - I guess it's too much for this task. 

In DLPU "lower" (with the smallest spatial dimension) connections feature map is 8x8x256, in PhUn 32x32x32, in VUR-Net 8x8x512



# References
1. Lin, Tsung-Yi, et al. "Feature pyramid networks for object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
2. Seferbekov, Selim, et al. "Feature pyramid network for multi-class land segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2018.
3. He, Kaiming, et al. "Identity mappings in deep residual networks." European conference on computer vision. Springer, Cham, 2016.
4. K. Wang, Y. Li, K. Qian, J. Di, and J. Zhao, “One-step robust deep learning phase unwrapping,” Opt. Express 27, 15100–15115 (2019).
5. Gili Dardikman-Yoffe, Darina Roitshtain, Simcha K. Mirsky, Nir A. Turko, Mor Habaza, and Natan T. Shaked, "PhUn-Net: ready-to-use neural network for unwrapping quantitative phase images of biological cells," Biomed. Opt. Express 11, 1107-1121 (2020).
6. Qin, Y., Wan, S., Wan, Y., Weng, J., Liu, W., & Gong, Q. (2020). Direct and accurate phase unwrapping with deep neural network. Applied optics, 59 24, 7258-7267 .
