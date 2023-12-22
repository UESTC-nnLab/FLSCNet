import torch
import torch.nn as nn


class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, padding, kernel_size, attention):
        super(Layer, self).__init__()
        self.attention = attention
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        if self.attention:
            self.ca = ChannelAttention(out_channels)
            self.sa = SpatialAttention()

    def forward(self, x):
        if self.attention:  # attention
            out_before_attention = self.bn(self.conv(x))
            out_ca = self.ca(out_before_attention) * out_before_attention
            out_sa = self.sa(out_ca) * out_ca
            out = self.relu(out_before_attention + out_sa)
            return out
        else:  # no attention
            return self.relu(self.bn(self.conv(x)))


class Max_Layer(nn.Module):
    def __init__(self, channels):
        super(Max_Layer, self).__init__()
        self.max = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.max(x)))


def make_layer_0(input_channels, output_channels, attention=False):
    layer = []
    layer.append(Layer(input_channels, output_channels, kernel_size=1, padding=0, attention=attention))
    return nn.Sequential(*layer)


def make_layer_1(input_channels, output_channels, attention=False):
    layer = []
    layer.append(Layer(input_channels, output_channels, kernel_size=3, padding=1, attention=attention))
    return nn.Sequential(*layer)


def make_layer_2(input_channels, output_channels, attention=False):
    layer = []
    layer.append(Layer(input_channels, output_channels, kernel_size=5, padding=2, attention=attention))
    return nn.Sequential(*layer)


def make_layer_3(input_channels, output_channels, attention=False):
    layer = []
    layer.append(Layer(input_channels, output_channels, kernel_size=7, padding=3, attention=attention))
    return nn.Sequential(*layer)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Fusion_Module_InSMFE(nn.Module):
    def __init__(self, channel):
        super(Fusion_Module_InSMFE, self).__init__()
        self.fusion_feature = make_layer_0(2 * channel, channel)

    def forward(self, x, y):
        feature = self.fusion_feature(torch.cat([x, y], dim=1))
        final_feature = x + feature + y
        return final_feature


class CFSM(nn.Module):
    def __init__(self, channel_less, channel_more, sca_fac=2):
        super(CFSM, self).__init__()
        self.fusion_higher_feature = make_layer_0(channel_less + channel_more, channel_more)
        self.fusion_lower_feature = make_layer_0(channel_less + channel_more, channel_less)
        self.fusion_f2_feature = make_layer_0(channel_less + channel_more, channel_less)
        self.fusion_trans_x = make_layer_0(channel_less, channel_less)
        self.fusion_trans_y = make_layer_0(channel_more, channel_less)
        self.up_pool = nn.Upsample(scale_factor=sca_fac, mode='bilinear', align_corners=True)
        self.down_pool = nn.MaxPool2d(kernel_size=sca_fac)

    def forward(self, x, y):
        higher_feature = self.fusion_higher_feature(torch.cat([self.down_pool(x), y], dim=1))
        lower_feature = self.fusion_lower_feature(torch.cat([x, self.up_pool(y)], dim=1))
        f2 = self.fusion_f2_feature(torch.cat([self.up_pool(higher_feature), lower_feature], dim=1))
        f1 = self.fusion_trans_x(x)
        f3 = self.fusion_trans_y(self.up_pool(y))
        final_feature = f1 + f2 + f3
        return final_feature


class SMFE(nn.Module):
    def __init__(self, input_channels, output_channels_0, output_channels_1):
        super(SMFE, self).__init__()
        self.max0_0 = Max_Layer(input_channels)
        self.max0_1 = Max_Layer(output_channels_0)
        self.max0_2 = Max_Layer(output_channels_0)
        self.max0_3 = Max_Layer(output_channels_0)

        self.conv0_max_0 = make_layer_0(input_channels, output_channels_0)

        self.conv0_0_0 = make_layer_0(input_channels, output_channels_0)
        self.conv0_0_1 = make_layer_0(output_channels_0, output_channels_0)
        self.conv0_0_2 = make_layer_0(output_channels_0, output_channels_0, attention=True)
        self.conv0_0_3 = make_layer_0(output_channels_0, output_channels_0, attention=True)

        self.conv0_1_0 = make_layer_1(input_channels, output_channels_0)
        self.conv0_1_1 = make_layer_1(output_channels_0, output_channels_0)
        self.conv0_1_2 = make_layer_1(output_channels_0, output_channels_0, attention=True)
        self.conv0_1_3 = make_layer_1(output_channels_0, output_channels_0, attention=True)

        self.conv0_2_0 = make_layer_2(input_channels, output_channels_0)
        self.conv0_2_1 = make_layer_2(output_channels_0, output_channels_0)
        self.conv0_2_2 = make_layer_2(output_channels_0, output_channels_0, attention=True)
        self.conv0_2_3 = make_layer_2(output_channels_0, output_channels_0, attention=True)

        self.conv0_trans_0 = make_layer_0(4 * output_channels_0, output_channels_0, attention=True)
        self.conv0_trans_1 = make_layer_0(4 * output_channels_0, output_channels_0, attention=True)
        self.conv0_trans_2 = make_layer_0(4 * output_channels_0, output_channels_0, attention=True)
        self.conv0_trans_3 = make_layer_0(4 * output_channels_0, output_channels_0, attention=True)

        self.max1_0 = Max_Layer(input_channels)
        self.max1_1 = Max_Layer(output_channels_1)
        self.max1_2 = Max_Layer(output_channels_1)
        self.max1_3 = Max_Layer(output_channels_1)

        self.conv1_max_0 = make_layer_0(input_channels, output_channels_1)

        self.conv1_0_0 = make_layer_0(input_channels, output_channels_1)
        self.conv1_0_1 = make_layer_0(output_channels_1, output_channels_1)
        self.conv1_0_2 = make_layer_0(output_channels_1, output_channels_1, attention=True)
        self.conv1_0_3 = make_layer_0(output_channels_1, output_channels_1, attention=True)

        self.conv1_1_0 = make_layer_1(input_channels, output_channels_1)
        self.conv1_1_1 = make_layer_1(output_channels_1, output_channels_1)
        self.conv1_1_2 = make_layer_1(output_channels_1, output_channels_1, attention=True)
        self.conv1_1_3 = make_layer_1(output_channels_1, output_channels_1, attention=True)

        self.conv1_2_0 = make_layer_2(input_channels, output_channels_1)
        self.conv1_2_1 = make_layer_2(output_channels_1, output_channels_1)
        self.conv1_2_2 = make_layer_2(output_channels_1, output_channels_1, attention=True)
        self.conv1_2_3 = make_layer_2(output_channels_1, output_channels_1, attention=True)

        self.conv1_trans_0 = make_layer_0(4 * output_channels_1, output_channels_1, attention=True)
        self.conv1_trans_1 = make_layer_0(4 * output_channels_1, output_channels_1, attention=True)
        self.conv1_trans_2 = make_layer_0(4 * output_channels_1, output_channels_1, attention=True)
        self.conv1_trans_3 = make_layer_0(4 * output_channels_1, output_channels_1, attention=True)

        self.fusion_0 = Fusion_Module_InSMFE(output_channels_1)
        self.fusion_1 = Fusion_Module_InSMFE(output_channels_1)
        self.fusion_2 = Fusion_Module_InSMFE(output_channels_1)
        self.fusion_3 = Fusion_Module_InSMFE(output_channels_1)

        self.trans_conv = make_layer_0(input_channels + 4 * output_channels_1, output_channels_1, attention=True)

    def forward(self, input):
        x0_0 = self.conv0_trans_0(torch.cat(
            [self.conv0_max_0(self.max0_0(input)), self.conv0_0_0(input), self.conv0_1_0(input), self.conv0_2_0(input)],
            dim=1))
        x0_1 = self.conv0_trans_1(torch.cat([self.max0_1(x0_0),
                                             self.conv0_0_1(x0_0),
                                             self.conv0_1_1(x0_0),
                                             self.conv0_2_1(x0_0)], dim=1))
        x0_2 = self.conv0_trans_2(torch.cat([self.max0_2(x0_1),
                                             self.conv0_0_2(x0_1),
                                             self.conv0_1_2(x0_1),
                                             self.conv0_2_2(x0_1)], dim=1))
        x0_3 = self.conv0_trans_3(torch.cat([self.max0_3(x0_2),
                                             self.conv0_0_3(x0_2),
                                             self.conv0_1_3(x0_2),
                                             self.conv0_2_3(x0_2)], dim=1))

        x1_input = input

        x1_0 = self.conv1_trans_0(
            torch.cat([self.conv1_max_0(self.max1_0(x1_input)), self.conv1_0_0(x1_input), self.conv1_1_0(
                x1_input), self.conv1_2_0(x1_input)], dim=1))
        x1_1 = self.conv1_trans_1(torch.cat([self.max1_1(x1_0),
                                             self.conv1_0_1(x1_0),
                                             self.conv1_1_1(x1_0),
                                             self.conv1_2_1(x1_0)], dim=1))
        x1_2 = self.conv1_trans_2(torch.cat([self.max1_2(x1_1),
                                             self.conv1_0_2(x1_1),
                                             self.conv1_1_2(x1_1),
                                             self.conv1_2_2(x1_1)], dim=1))
        x1_3 = self.conv1_trans_3(torch.cat([self.max1_3(x1_2),
                                             self.conv1_0_3(x1_2),
                                             self.conv1_1_3(x1_2),
                                             self.conv1_2_3(x1_2)], dim=1))

        x_0 = self.fusion_0(x1_0, x0_0)
        x_1 = self.fusion_1(x1_1, x0_1)
        x_2 = self.fusion_2(x1_2, x0_2)
        x_3 = self.fusion_3(x1_3, x0_3)

        output = self.trans_conv(torch.cat([x1_input, x_0, x_1, x_2, x_3], dim=1))
        return output


class FLSCNet(nn.Module):
    def __init__(self, num_classes, input_channels, filters, fullLevel_supervision=False):
        super(FLSCNet, self).__init__()
        self.fullLevel_supervision = fullLevel_supervision

        self.SMFE_1 = SMFE(input_channels, filters[0], filters[1])
        self.SMFE_2 = SMFE(input_channels + filters[1], filters[2], filters[3])
        self.SMFE_3 = SMFE(input_channels + filters[1] + filters[3], filters[4], filters[5])
        self.SMFE_4 = SMFE(input_channels + filters[1] + filters[3] + filters[5], filters[6], filters[7])
        self.SMFE_5 = SMFE(input_channels + filters[1] + filters[3] + filters[5] + filters[7], filters[8], filters[9])
        self.SMFE_6 = SMFE(input_channels + filters[1] + filters[3] + filters[5] + filters[7] + filters[9],
                           filters[10], filters[11])

        self.CFSM_1 = CFSM(filters[1], filters[3])
        self.CFSM_2 = CFSM(filters[3], filters[5])
        self.CFSM_3 = CFSM(filters[5], filters[7])
        self.CFSM_4 = CFSM(filters[7], filters[9])
        self.CFSM_5 = CFSM(filters[9], filters[11])

        self.conv_final = nn.Conv2d(filters[1], num_classes, kernel_size=1)
        self.down_pooling_max = nn.MaxPool2d(kernel_size=2)
        if self.fullLevel_supervision:
            self.conv_final_1 = nn.Conv2d(filters[1], num_classes, kernel_size=1)
            self.conv_final_2 = nn.Conv2d(filters[3], num_classes, kernel_size=1)
            self.conv_final_3 = nn.Conv2d(filters[5], num_classes, kernel_size=1)
            self.conv_final_4 = nn.Conv2d(filters[7], num_classes, kernel_size=1)
            self.conv_final_5 = nn.Conv2d(filters[9], num_classes, kernel_size=1)
            self.conv_final_6 = nn.Conv2d(filters[11], num_classes, kernel_size=1)
            self.up_pool_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up_pool_3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.up_pool_4 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
            self.up_pool_5 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
            self.up_pool_6 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

    def forward(self, input):
        x1 = self.SMFE_1(input)
        x2_input = self.down_pooling_max(torch.cat([input, x1], dim=1))
        x2 = self.SMFE_2(x2_input)
        x3_input = self.down_pooling_max(torch.cat([x2_input, x2], dim=1))
        x3 = self.SMFE_3(x3_input)
        x4_input = self.down_pooling_max(torch.cat([x3_input, x3], dim=1))
        x4 = self.SMFE_4(x4_input)
        x5_input = self.down_pooling_max(torch.cat([x4_input, x4], dim=1))
        x5 = self.SMFE_5(x5_input)
        x6_input = self.down_pooling_max(torch.cat([x5_input, x5], dim=1))
        x6 = self.SMFE_6(x6_input)

        F5 = self.CFSM_5(x5, x6)
        F4 = self.CFSM_4(x4, F5)
        F3 = self.CFSM_3(x3, F4)
        F2 = self.CFSM_2(x2, F3)
        F1 = self.CFSM_1(x1, F2)

        if self.fullLevel_supervision:
            output_1 = self.conv_final_1(x1)
            output_2 = self.up_pool_2(self.conv_final_2(x2))
            output_3 = self.up_pool_3(self.conv_final_3(x3))
            output_4 = self.up_pool_4(self.conv_final_4(x4))
            output_5 = self.up_pool_5(self.conv_final_5(x5))
            output_6 = self.up_pool_6(self.conv_final_6(x6))
            output = self.conv_final(F1)
            output_final = [output_1, output_2, output_3, output_4, output_5, output_6, output]
        else:
            output_final = self.conv_final(F1)
        return output_final
