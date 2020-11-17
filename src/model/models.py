import torch
import numpy as np
from torch import nn


def make_seq_block(in_c, out_c, mode='linear'):
    if mode == 'linear':
        return nn.Sequential(
                nn.Linear(in_c, out_c),
                nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1),
                nn.ReLU(inplace=True))


class NeRF(nn.Module):
    """
    Model architecture
    Input : Set of 3D points and/or viewing directions
    Output : RGB color and volume density at that point
    Note : Instead of putting raw input into skip connection, we are passing the input through fc layer
    and then skip connecting it
    """
    def __init__(
        self,
        num_layers=6,
        hidden_size=128,
        skip_connect=[4],
        input_ch = 3,
        output_ch = 3
    ):
        super(NeRF, self).__init__()

        self.input_ch = input_ch
        
        self.skip_connect = skip_connect

        self.layers = nn.ModuleList()

        self.layers.append(make_seq_block(self.input_ch, hidden_size))

        for i in range(1, num_layers+1):
            if i in skip_connect and i != 0:
                self.layers.append(make_seq_block(hidden_size+self.input_ch, hidden_size))
            else:
                self.layers.append(make_seq_block(hidden_size, hidden_size))

        self.fc_out = nn.Linear(hidden_size, output_ch)

    def forward(self, input):
        output = input

        for i in range(len(self.layers)):
            if i in self.skip_connect and i != 0:
                output = self.layers[i](torch.cat((output, input), 1))
            else:
                output = self.layers[i](output)

        output = self.fc_out(output)

        return output


class NeRF2(nn.Module):
    """
    Model architecture
    Input : Set of 3D points and/or viewing directions
    Output : RGB color and volume density at that point
    Note : Instead of putting raw input into skip connection, we are passing the input through fc layer
    and then skip connecting it
    """
    def __init__(
        self,
        num_layers=6,
        hidden_size=128,
        skip_connect=[4]
    ):
        super(NeRF2, self).__init__()

        self.skip_connect = skip_connect

        self.layers = nn.ModuleList()

        self.layers.append(make_seq_block(3, hidden_size))

        for i in range(1, num_layers):
            if i in skip_connect and i != 0:
                self.layers.append(make_seq_block(hidden_size+3, hidden_size))
            else:
                self.layers.append(make_seq_block(hidden_size, hidden_size))

        self.fc_sigma = nn.Linear(hidden_size, 1)
        self.fc_feat = nn.Linear(hidden_size, hidden_size)
        self.fc_viewdir = make_seq_block(hidden_size+3, hidden_size//2)
        self.fc_rgb = nn.Linear(hidden_size//2, 3)

    def forward(self, points, viewdirs):
        output = points

        for i in range(len(self.layers)):
            if i in self.skip_connect and i != 0:
                output = self.layers[i](torch.cat((output, points), 1))
            else:
                output = self.layers[i](output)

        sigma = self.fc_sigma(output)
        output = self.fc_feat(output)
        output = self.fc_viewdir(torch.cat([output,viewdirs],dim=1))
        rgb = self.fc_rgb(output)
        output = torch.cat([rgb, sigma],dim=1)

        return output



class ConvNet(nn.Module):
    """
    Model architecture
    Input : Set of 3D points and/or viewing directions
    Output : RGB color and volume density at that point
    Note : Instead of putting raw input into skip connection, we are passing the input through fc layer
    and then skip connecting it
    """
    def __init__(
        self,
        num_layers=8,
        hidden_size=128,
        skip_connect=[4],
        input_ch = 3,
        output_ch = 3
    ):
        super(ConvNet, self).__init__()

        self.input_ch = input_ch
        
        self.skip_connect = skip_connect

        self.layers = nn.ModuleList()

        self.layers.append(make_seq_block(self.input_ch, hidden_size, mode='conv'))

        for i in range(1, num_layers-1):
            if i in skip_connect and i != 0:
                self.layers.append(make_seq_block(hidden_size+hidden_size, hidden_size, mode='conv'))
            else:
                self.layers.append(make_seq_block(hidden_size, hidden_size, mode='conv'))

        self.conv_skip = make_seq_block(self.input_ch, hidden_size, mode='conv')
        self.conv_out = nn.Conv2d(in_channels=hidden_size, out_channels=output_ch, kernel_size=1)

    def forward(self, input):
        output = input

        skip_input = self.conv_skip(input)

        for i in range(len(self.layers)):
            if i in self.skip_connect and i != 0:
                output = self.layers[i](torch.cat((output, skip_input), 1))
            else:
                output = self.layers[i](output)

        output = self.conv_out(output)

        return output


class Conv2Net(nn.Module):
    """
    Model architecture
    Input : Set of 3D points and/or viewing directions
    Output : RGBA at each point
    Dividing the network into two parts:
    First part of the network predicts the alpha value
    Second part takes the original input and alpha value and predicts the color value
    """
    def __init__(
        self,
        hidden_size=128,
        input_ch = 3
    ):
        super(Conv2Net, self).__init__()

        self.input_ch = input_ch

        self.layers_nw1 = nn.ModuleList()
        self.layers_nw2 = nn.ModuleList()

        # First part of network: 4 conv layers -> 4th conv layer predicts the alpha value
        self.layers_nw1.append(make_seq_block(self.input_ch, hidden_size, mode='conv'))
        self.layers_nw1.append(make_seq_block(hidden_size, hidden_size, mode='conv'))
        self.layers_nw1.append(make_seq_block(hidden_size, hidden_size, mode='conv'))

        # Second part of network
        self.layers_nw2.append(make_seq_block(hidden_size+hidden_size, hidden_size, mode='conv'))
        self.layers_nw2.append(make_seq_block(hidden_size, hidden_size, mode='conv'))
        self.layers_nw2.append(make_seq_block(hidden_size, hidden_size, mode='conv'))
        self.layers_nw2.append(make_seq_block(hidden_size, hidden_size, mode='conv'))

        self.skip_conv = nn.Conv2d(in_channels=self.input_ch, out_channels=hidden_size, kernel_size=1)
        self.conv_alpha = nn.Conv2d(in_channels=hidden_size, out_channels=1, kernel_size=1)
        self.conv_rgb = nn.Conv2d(in_channels=hidden_size, out_channels=3, kernel_size=1)

    def forward(self, input):
        output = input
        
        skip_input = self.skip_conv(input)

        for i in range(len(self.layers_nw1)):
            output = self.layers_nw1[i](output)
        
        alpha = self.conv_alpha(output)

        output = torch.cat((output, skip_input), 1)

        for i in range(len(self.layers_nw2)):
            output = self.layers_nw2[i](output)

        rgb = self.conv_rgb(output)

        return alpha, rgb
