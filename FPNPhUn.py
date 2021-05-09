class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False,padding_mode='replicate')
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False,padding_mode='replicate')
        
        
    def forward(self, x):
        
        out = self.conv1(x)
        
        residual = out

        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual
       
        return out

class FPNPhUn6(torch.nn.Module):
  
  def __init__(self,down_ch):
    super(FPNPhUn6,self).__init__()

    self.encoder1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(1,down_ch//8)
        )
    
    self.encoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(down_ch//8,down_ch//4)
        )
    
    self.encoder3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(down_ch//4,down_ch//2)
        )
    
    self.encoder4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(down_ch//2,down_ch)
        )

    
    self.conv_1x1_1 = nn.Conv2d(down_ch//8, down_ch//8, kernel_size=1, bias=True)
    self.conv_1x1_2 = nn.Conv2d(down_ch//4, down_ch//8, kernel_size=1, bias=True)
    self.conv_1x1_3 = nn.Conv2d(down_ch//2, down_ch//8, kernel_size=1, bias=True)
    self.conv_1x1_4 = nn.Conv2d(down_ch,    down_ch//8, kernel_size=1, bias=True)
    

    self.up_trans = nn.ConvTranspose2d(in_channels=down_ch//8, out_channels=down_ch//8, kernel_size=2, stride=2)
    
    #self.up = nn.UpsamplingNearest2d(scale_factor=2)
    self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)

    self.conv_block = nn.Sequential(
            nn.Conv2d(down_ch//8, down_ch//8-(down_ch//8-down_ch//16)//2, kernel_size=3, padding=1, bias=False,padding_mode='replicate'),
            nn.Conv2d(down_ch//8-(down_ch//8-down_ch//16)//2,down_ch//16, kernel_size=3, padding=1, bias=False,padding_mode='replicate')
        )

    self.head = nn.Sequential(
            nn.Conv2d(down_ch//4, down_ch//4, kernel_size=3, padding=1, bias=True,padding_mode='replicate'),
            nn.BatchNorm2d(down_ch//4),
            nn.ReLU(inplace=False),
            nn.Conv2d(down_ch//4, 1, kernel_size=1, bias=True),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        )

  def forward(self,image):
        
    x1 = self.encoder1(image)                   #[1, down_ch/8, 128, 128]
    x2 = self.encoder2(x1)                      #[1, down_ch/4, 64, 64]
    x3 = self.encoder3(x2)                      #[1, down_ch/2, 32, 32]
    x4 = self.encoder4(x3)                      #[1, down_ch, 16, 16]


    y4 = self.conv_1x1_4(x4)                     #[1, down_ch/8, 16, 16]

    y3 = self.conv_1x1_3(x3) + self.up_trans(y4) #[1, down_ch/8, 32, 32]
    
    y2 = self.conv_1x1_2(x2) + self.up_trans(y3) #[1, down_ch/8, 64, 64]

    y1 = self.conv_1x1_1(x1) + self.up_trans(y2) #[1, down_ch/8, 128, 128]
    
    z4 = self.conv_block(y4)                     #[1, down_ch/16, 128, 128]
    z3 = self.conv_block(y3)                     #[1, down_ch/16, 128, 128]
    z2 = self.conv_block(y2)                     #[1, down_ch/16, 128, 128]
    z1 = self.conv_block(y1)                     #[1, down_ch/16, 128, 128]

    x = torch.cat([
                     self.up(self.up(self.up(z4))),
                     self.up(self.up(z3)),
                     self.up(z2),
                     z1
                  ],1)
    
    x = self.head(x)

    return x

    #print(x.size(),'мой вывод после "линии"')
    

if __name__ == "__main__":
  image = torch.rand((1,1,256,256))
  model1 = FPNPhUn6(64)
  model2 = FPNPhUn6(128)
  print(model1(image).size())
  print(model2(image).size())


from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

model4 = FPNPhUn6(2048)
count_parameters(model4)
print(77_737,' <= 64')
print(310_113,' <= 128')
print(1_238_785,' <= 256')
print(4_951_809,' <= 512')
print(19_800_577,' <= 1024')
print(79_188_993,' <= 2048')
