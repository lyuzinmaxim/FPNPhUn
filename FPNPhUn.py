def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False,padding_mode='replicate')
                     
class conv3x3block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        
        super(conv3x3block, self).__init__()
        
        self.conv1 = conv3x3(in_channels, in_channels-(in_channels-out_channels)//2, stride)
        self.conv2 = conv3x3(in_channels-(in_channels-out_channels)//2, out_channels, stride) 
      
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.conv2(out)

        return out
        
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                     stride=stride, padding=0, bias=False,padding_mode='replicate')
                     

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        
        super(ResidualBlock, self).__init__()
        
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        
        
    def forward(self, x):
        
        out = self.conv1(x)
        residual = out
        out = self.bn1(out)
        out = self.relu(out)
        out += residual
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out
        
        
class FPNPhUn(torch.nn.Module):
  
  def __init__(self):
    super(FPNPhUn,self).__init__()

    self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

    
    self.block1 = ResidualBlock(1,8)
    self.block2 = ResidualBlock(8,16)
    self.block3 = ResidualBlock(16,32)
    self.block4 = ResidualBlock(32,64)

    self.conv_1x1_1 = conv1x1(8,8)
    self.conv_1x1_2 = conv1x1(16,8)
    self.conv_1x1_3 = conv1x1(32,8)
    self.conv_1x1_4 = conv1x1(64,8)

    self.conv_block =conv3x3block(8,4)

    self.up_trans = nn.ConvTranspose2d(
        in_channels=8,
        out_channels=8,
        kernel_size=2,
        stride=2)
    self.up = nn.UpsamplingNearest2d(scale_factor=2)

    

    self.relu = nn.ReLU(inplace=False)

    self.out = conv3x3(16,16)
    self.bn = nn.BatchNorm2d(16)
    self.outout = conv1x1(16,1)
    self.up_end = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)

  def forward(self,image):
    x = self.max_pool_2x2(image)
    #x = self.max_pool_2x2(x)
    x1 = self.block1(x) #[1, 8, 128, 128]

    x2 = self.max_pool_2x2(x1)
    x2 = self.block2(x2) #[1, 16, 64, 64]

    x3 = self.max_pool_2x2(x2)
    x3 = self.block3(x3) #[1, 32, 32, 32]

    x4 = self.max_pool_2x2(x3)
    x4 = self.block4(x4) #[1, 64, 16, 16]
    
    y4 = self.conv_1x1_4(x4)                     #[1, 8, 16, 16]

    y3 = self.conv_1x1_3(x3) + self.up_trans(y4) #[1, 8, 32, 32]
    
    y2 = self.conv_1x1_2(x2) + self.up_trans(y3) #[1, 8, 64, 64]

    y1 = self.conv_1x1_1(x1) + self.up_trans(y2) #[1, 8, 128, 128]
    
    z4 = self.conv_block(y4)
    z3 = self.conv_block(y3)
    z2 = self.conv_block(y2)
    z1 = self.conv_block(y1)

    x = torch.cat([
                    self.up(self.up(self.up(z4))),
                    self.up(self.up(z3)),
                    self.up(z2),
                    z1
                 ],1)
    
    x = self.out(x)
    x = self.bn(x)

    x = self.relu(x)
    x = self.outout(x)
    x = self.up_end(x)
    
    return x

    #print(x.size(),'мой вывод после "линии"')
    
  if __name__ == "__main__":
      image = torch.rand((1,1,256,256))
      model = FPNPhUn()
      print(model(image).size())
