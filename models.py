import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2, stride=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(131072, 1024)

        self.b_64 = nn.BatchNorm2d(64)
        self.b_128 = nn.BatchNorm2d(128)
        self.b_256 = nn.BatchNorm2d(256)
        self.b_512 = nn.BatchNorm2d(512)
        self.b_1024 = nn.BatchNorm2d(1024)
        self.F_relu = F.relu
        self.F_max_pool2d = F.max_pool2d

    def forward_once(self, x):
        # Max pooling over a (2, 2) window


        # 1,2,3,4
        x = self.F_relu(self.conv1(x))
        x = self.b_64(x)
        x = self.F_max_pool2d(x, 2)

        # 5,6,7,8
        x = self.F_relu(self.conv2(x))
        x = self.b_128(x)
        x = self.F_max_pool2d(x, 2)

        # 9,10,11,12
        x = self.F_relu(self.conv3(x))
        x = self.b_256(x)
        x = self.F_max_pool2d(x, 2)

        # 13,14,15 (no max pooling)
        x = self.F_relu(self.conv4(x))
        x = self.b_512(x)

        # 16
        x = x.view(-1, self.num_flat_features(x))

        # 17,18,19
        x = self.F_relu(self.fc1(x))
        x = self.b_1024(x)

        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Siamese_Net(torch.nn.Module):
    def __init__(self):
        super(Siamese_Net, self).__init__()
        if torch.cuda.is_available():
            self.net = Net().cuda()
        else:
            self.net = Net()
        self.fc = nn.Linear(2048, 1)

    def forward(self, input1, input2):
        output_1, output_2 = self.net(input1, input2)

        x = torch.cat((output_1, output_2), 1)
        m = nn.Sigmoid()
        output = m(self.fc(x))
        return output

