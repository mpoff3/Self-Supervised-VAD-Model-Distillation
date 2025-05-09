import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_CONFIGS = {  # AUC, GMAC, Param
    # "XS": dict(channels=[8, 16, 16], fc_dim=128),
    # "S":  dict(channels=[16, 32, 32], fc_dim=256),
    "Mv1":  dict(channels=[16, 32, 32], nb_conv=2, conv_type="3D", max_pool=True, bias=False, instance_norm=True),      # ????, 638.56 MMac, Params: 410.29 k
    "Mv2":  dict(channels=[16, 32, 32], nb_conv=2, conv_type="2+1D", max_pool=True, bias=False, instance_norm=True),    # ????, 320.99 MMac, Params: 354.13 k
    "Mv3":  dict(channels=[16, 32, 32], nb_conv=1, conv_type="2+1D", max_pool=True, bias=False, instance_norm=True),    # ????, 118.34 MMac, Params: 326.48 k
    "Mv4":  dict(channels=[16, 32, 32], nb_conv=1, conv_type="3D", max_pool=True, bias=False, instance_norm=True),      # ????, 190.42 MMac, Params: 348.08 k
    "Mv5":  dict(channels=[16, 32, 32], nb_conv=2, conv_type="3D", max_pool=False, bias=False, instance_norm=True),     # ????, 300.99 MMac, Params: 410.29 k
    "Mv6":  dict(channels=[16, 32, 32], nb_conv=1, conv_type="2+1D", max_pool=False, bias=False, instance_norm=True),   # ????, 29.92 MMac, Params: 326.48 k

    "Bv2":  dict(channels=[32, 64, 64], nb_conv=2, conv_type="2+1D", max_pool=True, bias=False, instance_norm=True),    # 0.90, 1.24 GMac,      Params: 1.35 M
    "Bv3":  dict(channels=[32, 64, 64], nb_conv=2, conv_type="1+2D", max_pool=True, bias=False, instance_norm=True),    # 0.88, 1.49 GMac,      Params: 1.37 M
    "Bv4":  dict(channels=[32, 64, 64], nb_conv=1, conv_type="3D", max_pool=True, bias=False, instance_norm=True),      # 0.91, 679.82 MMac,    Params: 1.32 M
    "Bv5":  dict(channels=[32, 64, 64], nb_conv=2, conv_type="3D", max_pool=False, bias=False, instance_norm=True),     # 0.91, 1.12 GMac,      Params: 1.57 M
    "Bv6":  dict(channels=[32, 64, 64], nb_conv=2, conv_type="2+1D", max_pool=False, bias=False, instance_norm=True),   # ????, 634.06 MMac,    Params: 1.35 M
    "B":    dict(channels=[32, 64, 64], nb_conv=2, conv_type="3D", max_pool=True, bias=False, instance_norm=True),      # 0.92, 2.47 GMac,      Params: 1.57 M
}


class Conv3dBlock(nn.Module):
    def __init__(self, d_in, d_out, instance_norm=True, nb_conv=2, conv_type="3D", max_pool=True, bias=False, maxpool_timepool=1, **kwargs):
        """The conv_kernel"""
        super().__init__()
        layers = []
        if conv_type == "3D":
            conv_param = [{'kernel': (3, 3, 3), 'padding': (1, 1, 1)}]
        elif conv_type == "2+1D":
            conv_param = [{'kernel': (1, 3, 3), 'padding': (0, 1, 1)}, {'kernel': (3, 1, 1), 'padding': (1, 0, 0)}]
        elif conv_type == "1+2D":
            conv_param = [{'kernel': (3, 1, 1), 'padding': (1, 0, 0)}, {'kernel': (1, 3, 3), 'padding': (0, 1, 1)}]
        else:
            raise ValueError(f'{conv_type} is not in ["3D", "2+1D", "1+2D"]')

        for i in range(nb_conv):
            for conv in conv_param:
                stride = (1, 1, 1)
                kernel, padding = conv['kernel'], conv['padding']
                if not max_pool and i == nb_conv-1 and kernel[1] == 3:
                    stride = (1, 2, 2)
                layers.append(nn.Conv3d(d_in, d_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias))
                if instance_norm:
                    layers.append(nn.InstanceNorm3d(d_out))
                layers.append(nn.ReLU())
                d_in = d_out
        if max_pool:
            layers.append(nn.MaxPool3d(kernel_size=(maxpool_timepool, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)))
        elif maxpool_timepool > 1:
            layers.append(nn.MaxPool3d(kernel_size=(maxpool_timepool, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class WideBranchNet(nn.Module):

    def __init__(self, time_length=7, num_classes=[127, 8], variant='B'):
        super(WideBranchNet, self).__init__()
        config = MODEL_CONFIGS[variant]
        print(f'Configuration loaded {variant}', config)
        # config = dict(channels=[32, 64, 64], nb_conv=2, conv_type="3D", max_pool=True, bias=False, instance_norm=True)
        self.time_length = time_length
        self.num_classes = num_classes
        layers = []
        c_in = 3
        n = len(config['channels'])

        for i, c_out in enumerate(config['channels']):
            maxpool_timepool = self.time_length if i == n-1 else 1
            layers.append(Conv3dBlock(c_in, c_out, maxpool_timepool=maxpool_timepool, **config))
            c_in = c_out

        self.model = nn.Sequential(*layers)
        self.conv2d = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.InstanceNorm2d(c_out),
            nn.ReLU(),
            nn.Dropout2d(p=0.3)
        )

        self.max2d = nn.MaxPool2d(2, 2)
        self.classifier_1 = nn.Sequential(
            nn.Linear(c_out*16, c_out*8),
            nn.ReLU(),
            nn.Linear(c_out*8, self.num_classes[0])
        )
        self.classifier_2 = nn.Sequential(
            nn.Linear(c_out*16, c_out*8),
            nn.ReLU(),
            nn.Linear(c_out*8, self.num_classes[1])
        )

    def forward(self, x):
        out = self.model(x)
        out = out.squeeze(2)
        out = self.max2d(self.conv2d(out))
        out = out.view((out.size(0), -1))
        out1 = self.classifier_1(out)
        out2 = self.classifier_2(out)
        return out1, out2


class WideBranchNet_Strided(WideBranchNet):
    def __init__(self, time_length=7, stride=2, nb_patches=9, **kwargs):
        self.stride = stride
        self.original_time_length = time_length
        self.time_length = (time_length + stride-1)//stride
        self.nb_patches = nb_patches
        super().__init__(self.time_length, [self.time_length**2, nb_patches**2], **kwargs)

    def forward(self, x):
        x = x[:, :, ::self.stride]  # B, C, T, H, W
        return super().forward(x)

    def transform_label(self, temp, temp_teacher=None):
        return temp[:, ::self.stride]//self.stride, temp_teacher.view(-1, self.original_time_length, self.original_time_length)[:, ::self.stride, ::self.stride].view(-1, self.time_length**2) if temp_teacher is not None else None


def flatten_state_dict(from_state, to_state):
    return {to_k: from_v for to_k, from_v in zip(to_state.keys(), from_state.values())}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Model sizes')
    parser.add_argument('--preset', default='Bv5', type=str)
    args = parser.parse_args()

    from torchsummary import summary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = WideBranchNet(time_length=7, num_classes=[7**2, 9**2], variant=args.preset).to(device)
    # net = WideBranchNet_Strided(time_length=7, stride=3).to(device)

    # state = torch.load("../avenue_92.18.pth", map_location=device)
    # net.load_state_dict(flatten_state_dict(state, net.state_dict()), strict=False)

    summary(net, input_size=(3, 7, 64, 64), device=device)

    x = torch.rand(2, 3, 7, 64, 64).to(device)
    out = net(x)

    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(net, (3, 7, 64, 64), as_strings=True, print_per_layer_stat=True)

    print(f"FLOPs: {macs}, Params: {params}")
