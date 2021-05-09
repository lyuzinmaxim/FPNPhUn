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

In this attempt I made following structure (shown below), so the model has only 77.688 trainable parameters. 

1x256x256
   ||
   V
8x128x128  => 8x128x128 => 4x128x128
   ||             ||
   V              V
16x64x64   => 8x64x64   => 4x64x64
   ||             ||
   V              V                      ===>(concat + upsample)16x128x128   ===> conv3x3,bn,relu,conv1x1(bottleneck),upsample ===> 1x256x256
32x32x32   => 8x32x32   => 4x32x32
   ||             ||
   V              V
64x16x16   => 8x16x16   => 4x16x16

![image](https://user-images.githubusercontent.com/73649419/117505122-d855a480-af83-11eb-9322-a3fb5bcb7575.png)

So in 50 epochs on the same small dataset (20 different phase images) averaged over 5 attempts with fixed random seed train and test losses are:

![image](https://user-images.githubusercontent.com/73649419/117506868-92e6a680-af86-11eb-9417-399019a42d05.png)

And corresponding metrics are:

![image](https://user-images.githubusercontent.com/73649419/117506901-a134c280-af86-11eb-9c98-5e5775dc36de.png)


I've learned that model and PhUn [5] on small dataset (20 obj) for 100 epochs just to see how they will be overfitted. Losses and metrics are:

# 5-th attempt

Because good example of tricks in image-to-image nets are in segmentation nets, from [7] it's reasonable to use dilated (atrous) convolutions instead of any variants of pooling-conv-unpooling.
Now i will compare net from 4-th attempt to this implementation on small dataset.

# 6-th attempt

I've rewrited and reorganized my model - and chosen max feature map depth 256 channels, so the model does have 1.8 M trainable parameters - and I compare my model with DLPU [4] on small dataset (100 obj) for 100 epochs.

Losses are:

![image](https://user-images.githubusercontent.com/73649419/117582142-3b247880-b109-11eb-8cce-7c81124c1d27.png)

And metrics:

![image](https://user-images.githubusercontent.com/73649419/117582163-5b543780-b109-11eb-8d2f-8312b4f0505e.png)

So, on small dataset FPNPhUn model can be learned much faster than DLPU model - it can be got from metrics. And due to correct residual connections in FPNPhUn train loss decreases much faster too. Test loss can't be normal understood due to small dataset.


# 7-th attempt

Now I should say, that in [2] was not used maxpooling, but convolution with stride 2 instead. In first up-bottom steps I will use conv3x3 with replicated padding=1 and stride=2.
I compare "convolutional downsampling"-model with "maxpool downsampling"-model on small dataset (100 obj) for 30 epochs, averaged over 5 attempts.




# References
1. Lin, Tsung-Yi, et al. "Feature pyramid networks for object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
2. Seferbekov, Selim, et al. "Feature pyramid network for multi-class land segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2018.
3. He, Kaiming, et al. "Identity mappings in deep residual networks." European conference on computer vision. Springer, Cham, 2016.
4. K. Wang, Y. Li, K. Qian, J. Di, and J. Zhao, “One-step robust deep learning phase unwrapping,” Opt. Express 27, 15100–15115 (2019).
5. Gili Dardikman-Yoffe, Darina Roitshtain, Simcha K. Mirsky, Nir A. Turko, Mor Habaza, and Natan T. Shaked, "PhUn-Net: ready-to-use neural network for unwrapping quantitative phase images of biological cells," Biomed. Opt. Express 11, 1107-1121 (2020).
6. Qin, Y., Wan, S., Wan, Y., Weng, J., Liu, W., & Gong, Q. (2020). Direct and accurate phase unwrapping with deep neural network. Applied optics, 59 24, 7258-7267 .
7. Chen, Liang-Chieh, et al. "Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs." IEEE transactions on pattern analysis and machine intelligence 40.4 (2017): 834-848.
