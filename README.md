# FPNPhUn
This is my PyTorch implementation of DL model solving phase unwrapping problem based on FPN [1-2]

# 1st attempt

Original structure of U-Net-like model [2] was implemented in PyTorch, but due to the low spatial resolution it's impossible to use it in phase unwrapping tasks.
Structure from [2] is shown below.

![Seferbekov](https://user-images.githubusercontent.com/73649419/116997820-94417600-acdd-11eb-97f7-a376d0444b3a.jpg)

So like it's shown below my output (in [2] output size was 4 times smaller than input - and I did nearest neighbourhood interpolation) is very "gnarled"



# References
1. Lin, Tsung-Yi, et al. "Feature pyramid networks for object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
2. Seferbekov, Selim, et al. "Feature pyramid network for multi-class land segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2018.
