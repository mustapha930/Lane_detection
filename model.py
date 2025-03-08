import torch
import torch.nn as nn
import torch.nn.functional as F
from swin import *

# ------------------- Swin Encoder -------------------
class SwinEncoder(nn.Module):
    def __init__(
        self,
        channels,
        hidden_dim,
        layers,
        heads,
        downscaling_factors,
        head_dim,
        window_size,
        relative_pos_embedding,
    ):
        super(SwinEncoder, self).__init__()
        self.stage1 = StageModule(
            in_channels=channels,
            hidden_dimension=hidden_dim,
            layers=layers[0],
            downscaling_factor=downscaling_factors[0],
            num_heads=heads[0],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )
        self.stage2 = StageModule(
            in_channels=hidden_dim,
            hidden_dimension=hidden_dim * 2,
            layers=layers[1],
            downscaling_factor=downscaling_factors[1],
            num_heads=heads[1],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )
        self.stage3 = StageModule(
            in_channels=hidden_dim * 2,
            hidden_dimension=hidden_dim * 4,
            layers=layers[2],
            downscaling_factor=downscaling_factors[2],
            num_heads=heads[2],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )
        self.stage4 = StageModule(
            in_channels=hidden_dim * 4,
            hidden_dimension=hidden_dim * 8,
            layers=layers[3],
            downscaling_factor=downscaling_factors[3],
            num_heads=heads[3],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return x1, x2, x3, x4

# ------------------- CNN-SCNN Encoder -------------------
class CNNSCNNEncoder(nn.Module):
    def __init__(self):
        super(CNNSCNNEncoder, self).__init__()
        # Define CNN encoder blocks
        self.conv1_1 = self._make_encoder_block(3, 16)
        self.conv1_2 = self._make_encoder_block(16, 16)
        self.conv1_3 = self._make_encoder_block(16, 16)

        self.conv2_1 = self._make_encoder_block(16, 32)
        self.conv2_2 = self._make_encoder_block(32, 32)
        self.conv2_3 = self._make_encoder_block(32, 32)

        self.conv3_1 = self._make_encoder_block(32, 64)
        self.conv3_2 = self._make_encoder_block(64, 64)
        self.conv3_3 = self._make_encoder_block(64, 64)

        self.conv4_1 = self._make_encoder_block(64, 128)
        self.conv4_2 = self._make_encoder_block(128, 128)
        self.conv4_3 = self._make_encoder_block(128, 128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # SCNN part
        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Message passing
        self.message_passing = nn.ModuleList()
        ms_ks = 9  # message passing kernel size (can be parameterized)
        self.message_passing.add_module(
            'up_down', nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False)
        )
        self.message_passing.add_module(
            'down_up', nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False)
        )
        self.message_passing.add_module(
            'left_right',
            nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False)
        )
        self.message_passing.add_module(
            'right_left',
            nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False)
        )

    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def message_passing_once(self, x, conv, vertical=True, reverse=False):
        nB, C, H, W = x.shape
        if vertical:
            slices = [x[:, :, i:(i + 1), :] for i in range(H)]
            dim = 2
        else:
            slices = [x[:, :, :, i:(i + 1)] for i in range(W)]
            dim = 3
        if reverse:
            slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))

        if reverse:
            out = out[::-1]

        return torch.cat(out, dim=dim)

    def message_passing_forward(self, x):
        Vertical = [True, True, False, False]
        Reverse = [False, True, False, True]
        for ms_conv, v, r in zip(self.message_passing, Vertical, Reverse):
            x = self.message_passing_once(x, ms_conv, v, r)
        return x

    def forward(self, x):
        # CNN encoder blocks
        x_conv1_1 = self.conv1_1(x)
        x_conv1_2 = self.conv1_2(x_conv1_1)
        x_conv1_3 = self.conv1_3(x_conv1_2)
        x_conv1 = self.pool(x_conv1_3)

        x_conv2_1 = self.conv2_1(x_conv1)
        x_conv2_2 = self.conv2_2(x_conv2_1)
        x_conv2_3 = self.conv2_3(x_conv2_2)
        x_conv2 = self.pool(x_conv2_3)

        x_conv3_1 = self.conv3_1(x_conv2)
        x_conv3_2 = self.conv3_2(x_conv3_1)
        x_conv3_3 = self.conv3_3(x_conv3_2)
        x_conv3 = self.pool(x_conv3_3)

        x_conv4_1 = self.conv4_1(x_conv3)
        x_conv4_2 = self.conv4_2(x_conv4_1)
        x_conv4_3 = self.conv4_3(x_conv4_2)
        x_conv4_3 = self.pool(x_conv4_3)

        c4 = self.layer1(x_conv4_3)
        c4 = self.message_passing_forward(c4)

        return x_conv1, x_conv2, x_conv3, c4

# ------------------- Decoder -------------------
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Up-sampling layers (the same configuration is used in both branches)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)

    def forward(self, feature, skip1, skip2, skip3):
        x = self.up1(feature)
        x = torch.cat([x, skip1], dim=1)
        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.up3(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.up4(x)
        return x

# ------------------- Main Model -------------------
class Model(nn.Module):
    def __init__(
        self,
        input_size,
        ms_ks=9,
        hidden_dim=16,
        layers=[2, 2, 2, 2],
        heads=[2, 2, 2, 2],
        channels=3,
        head_dim=2,
        window_size=4,
        downscaling_factors=(2, 2, 2, 2),
        relative_pos_embedding=True,
    ):
        super(Model, self).__init__()
        # Initialize encoders
        self.swin_encoder = SwinEncoder(
            channels=channels,
            hidden_dim=hidden_dim,
            layers=layers,
            heads=heads,
            downscaling_factors=downscaling_factors,
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )
        self.cnn_scnn_encoder = CNNSCNNEncoder()

        # Initialize decoders (one for each encoder branch)
        self.swin_decoder = Decoder()
        self.cnn_decoder = Decoder()

        # Final segmentation prediction layer
        self.s = nn.ConvTranspose2d(32, 5, kernel_size=3, stride=1, padding=1, bias=False)

        # Existence prediction branch
        input_w, input_h = input_size
        self.fc_input_feature = 5 * int(input_w / 16) * int(input_h / 16)
        self.layer2 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 5, 1)
        )
        self.layer3 = nn.Sequential(
            nn.Softmax(dim=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_feature, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

        # Loss scales and functions
        self.scale_background = 0.4
        self.scale_seg = 1.0
        self.scale_exist = 0.1
        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.tensor([self.scale_background, 1, 1, 1, 1])
        )
        self.bce_loss = nn.BCELoss()

    def forward(self, img, seg_gt=None, exist_gt=None):
        # Swin encoder and decoder
        x1, x2, x3, x4 = self.swin_encoder(img)
        s = self.swin_decoder(x4, x3, x2, x1)

        # CNN-SCNN encoder and decoder
        x_conv1, x_conv2, x_conv3, c4 = self.cnn_scnn_encoder(img)
        c = self.cnn_decoder(c4, x_conv3, x_conv2, x_conv1)

        # Combine decoder outputs and predict segmentation
        out = torch.cat([c, s], dim=1)
        seg_pred = self.s(out)

        # Existence prediction branch using features from the Swin encoder
        x = self.layer2(x4)
        x = self.layer3(x)
        x = x.view(-1, self.fc_input_feature)
        exist_pred = self.fc(x)

        if seg_gt is not None and exist_gt is not None:
            loss_seg = self.ce_loss(seg_pred, seg_gt)
            loss_exist = self.bce_loss(exist_pred, exist_gt)
            loss = loss_seg * self.scale_seg + loss_exist * self.scale_exist
        else:
            loss_seg = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss_exist = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss = torch.tensor(0, dtype=img.dtype, device=img.device)

        return seg_pred, exist_pred, loss_seg, loss_exist, loss




input_size = (512, 256)
ms_ks = 9
pretrained = True


model = Model(input_size=input_size, ms_ks=ms_ks)


sample_input = torch.randn(1, 3, 512, 256)


seg_pred, exist_pred, loss_seg, loss_exist, loss= model(sample_input)

# Print the shapes of seg_pred and exist_pred
print("Shape of seg_pred:", seg_pred.shape)
print("Shape of exist_pred:", exist_pred.shape)

