import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# residual block that uses option A (section 4.2; figure 3)
class IdentityResidual(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
      super(IdentityResidual, self).__init__()
      self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=stride, padding=1, bias=False)
      self.bn1 = nn.BatchNorm2d(out_chan)
      self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
      self.bn2 = nn.BatchNorm2d(out_chan)

      # parameters needed when using option A
      self.shortcut = False
      self.identity = None
      self.num_zeros = 0

      # use option A: zero padding to deal with size mismatch or stride mismatch
      if stride != 1 or in_chan != out_chan:
        self.shortcut = True # set boolean to true
        self.identity = nn.MaxPool2d(1, stride=stride) # maxpool size 1, stride 1 is identity, maxpool size 1, stride=stride is downsample by stride
        self.num_zeros = out_chan - in_chan # required amount of zero padding

    def forward(self, x):
      identity = x
      out = self.conv1(x)
      out = F.relu(self.bn1(out))
      out = self.conv2(out)
      out = self.bn2(out)
      if self.shortcut: # employ option A: "performs identity mapping, with extra zero entries padded"
        identity = self.identity(x)
        identity = F.pad(identity, (0, 0, 0, 0, 0, self.num_zeros)) # just add needed 0 padding to the back of channels
      out += identity
      out = F.relu(out)
      return out

# define the resnet model (ResNet20 has n = 3)
class ResNet20(nn.Module):
    def __init__(self):
        super(ResNet20, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.block1 = self.make_block1()
        self.block2 = self.make_block2()
        self.block3 = self.make_block3()
        self.fc1 = nn.Linear(64, 10)

    # block 1 has initial stride of 1
    # three layers for ResNet-20
    def make_block1(self):
      layers = []
      layers.append(IdentityResidual(16, 16, stride=1))
      layers.append(IdentityResidual(16, 16, stride=1))
      layers.append(IdentityResidual(16, 16, stride=1))

      return nn.Sequential(*layers)

    # block 2 has initial stride of 2 because shortcut goes accross feature map of two sizes
    def make_block2(self):
      layers = []
      layers.append(IdentityResidual(16, 32, stride=2)) # stride 2
      layers.append(IdentityResidual(32, 32, stride=1))
      layers.append(IdentityResidual(32, 32, stride=1))

      return nn.Sequential(*layers)

    # block 3 has initial stride of 2 because shortcut goes accross feature map of two sizes
    def make_block3(self):
      layers = []
      layers.append(IdentityResidual(32, 64, stride=2)) # stride 2
      layers.append(IdentityResidual(64, 64, stride=1)) # stride 2
      layers.append(IdentityResidual(64, 64, stride=1)) # stride 2

      return nn.Sequential(*layers)
    
    # block 4 has initial stride of 2 because shortcut goes accross feature map of two sizes
    def make_block4(self):
        layers = []
        layers.append(IdentityResidual(64, 128, stride=2)) # stride 2
        layers.append(IdentityResidual(128, 128, stride=1)) # stride 1
        layers.append(IdentityResidual(128, 128, stride=1)) # stride 1

        return nn.Sequential(*layers)

    # block 5 has initial stride of 2 because shortcut goes accross feature map of two sizes
    def make_block5(self):
        layers = []
        layers.append(IdentityResidual(128, 256, stride=2)) # stride 2
        layers.append(IdentityResidual(256, 256, stride=1)) # stride 1
        layers.append(IdentityResidual(256, 256, stride=1)) # stride 1

        return nn.Sequential(*layers)

    def forward(self, x):
      out = self.conv1(x)
      out = self.bn1(out)
      out = F.relu(out)
      out = self.block1(out)
      out = self.block2(out)
      out = self.block3(out)
      out = F.avg_pool2d(out, out.size()[3])
      out = out.view(out.size(0), -1)
      out = self.fc1(out)
      return out
