import imp
from numpy import pad
import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.audio_encoder = nn.Sequential(
            self.make_conv2d_block(
                in_channels=1, out_channels=64, kernel_size=(3, 3),
                stride=(1, 2), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=0),
            self.make_conv2d_block(
                in_channels=64, out_channels=192, kernel_size=(3, 3),
                stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0),
            self.make_conv2d_block(
                in_channels=192, out_channels=256, kernel_size=(3, 3),
                stride=(1, 1), padding=(1, 1)),
            self.make_conv2d_block(
                in_channels=256, out_channels=256, kernel_size=(3, 3),
                stride=(1, 1), padding=(1, 1)),
            self.make_conv2d_block(
                in_channels=256, out_channels=256, kernel_size=(3, 3),
                stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(3, 2), stride=(2, 2), padding=0),
            self.make_conv2d_block(
                in_channels=256, out_channels=512, kernel_size=(4, 4),
                stride=(1, 1), padding=(2, 0)),
            self.make_conv2d_block(
                in_channels=512, out_channels=512, kernel_size=(1, 1),
                stride=(1, 1), padding=0),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=0),
        ).to(device)

        self.video_encoder = nn.Sequential(
            self.make_conv3d_block(
                in_channels=3, out_channels=64, kernel_size=(5, 7, 7),
                stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            self.make_conv3d_block(
                in_channels=64, out_channels=128, kernel_size=(1, 5, 5),
                stride=(1, 2, 2), padding=(0, 2, 2)),
            self.make_conv3d_block(
                in_channels=128, out_channels=256, kernel_size=(1, 3, 3),
                stride=(1, 1, 1), padding=(0, 1, 1)),
            self.make_conv3d_block(
                in_channels=256, out_channels=256, kernel_size=(1, 3, 3),
                stride=(1, 1, 1), padding=(0, 1, 1)),
            self.make_conv3d_block(
                in_channels=256, out_channels=256, kernel_size=(1, 3, 3),
                stride=(1, 1, 1), padding=(0, 1, 1)),
            self.make_conv3d_block(
                in_channels=256, out_channels=512, kernel_size=(1, 5, 5),
                stride=(1, 1, 1), padding=(0, 2, 2)),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            self.make_conv3d_block(
                in_channels=512, out_channels=512, kernel_size=(1, 1, 1),
                stride=(1, 1, 1), padding=0),
            nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0),
        ).to(device)
        self.init_weights()

        N, T, H, W = 1, 50, 112, 112
        dummy_audio_input = torch.zeros(N, 1, T * 4, 80, device=device)
        dummy_audio_output = self.audio_encoder(dummy_audio_input)
        print("For dummy audio input of shape:", dummy_audio_input.shape)
        print("Output from audio encoder:", dummy_audio_output.shape)

        dummy_video_input = torch.zeros(N, 3, T, H, W, device=device)
        dummy_video_output = self.video_encoder(dummy_video_input)
        print("For dummy video input of shape:", dummy_video_input.shape)
        print("Output from video encoder:", dummy_video_output.shape)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, video, audio):
        video_out = self.video_encoder(video)
        audio_out = self.audio_encoder(audio)
        video_out = F.normalize(video_out[:, :, None, :, :, :], dim=1)
        audio_out = F.normalize(audio_out.transpose(0, 1)[None, :, :, :, :, None], dim=1)
        return (video_out * audio_out).sum(dim=1).squeeze(0)

    def make_conv2d_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU())

    def make_conv3d_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                padding=padding),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU())
