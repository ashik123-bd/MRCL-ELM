import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class ResidualBlock(nn.Module):
  
    def __init__(self, in_channels, out_channels, dropout_rate=0.3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            padding=1, stride=1, bias=False
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ProjectionPath(nn.Module):
   
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projection_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=2, bias=False
        )

    def forward(self, x):
        return self.projection_conv(x)


class ELMLayer(nn.Module):
    
    def __init__(self, input_dim, hidden_units, output_units):
        super().__init__()

        self.hidden_weights = nn.Parameter(
            torch.randn(input_dim, hidden_units),
            requires_grad=False
        )
        self.hidden_bias = nn.Parameter(
            torch.randn(hidden_units),
            requires_grad=False
        )
        self.output_weights = nn.Parameter(
            torch.randn(hidden_units, output_units)
        )

        nn.init.xavier_uniform_(self.hidden_weights)
        nn.init.normal_(self.hidden_bias, mean=0.0, std=0.1)
        nn.init.xavier_uniform_(self.output_weights)

    def forward(self, x):
        H = torch.relu(x @ self.hidden_weights + self.hidden_bias)
        return H @ self.output_weights


class EfficientNetLSTMModel(nn.Module):
   
    def __init__(self, num_classes=10):
        super().__init__()

        self.efficientnet = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )
        self.backbone = nn.Sequential(
            *list(self.efficientnet.features.children())
        )

        self.residual_block1 = ResidualBlock(
            in_channels=1280, out_channels=64, dropout_rate=0.3
        )
        self.projection_path1 = ProjectionPath(
            in_channels=1280, out_channels=64
        )

        self.residual_block2 = ResidualBlock(
            in_channels=64, out_channels=128, dropout_rate=0.4
        )
        self.projection_path2 = ProjectionPath(
            in_channels=64, out_channels=128
        )

        self.conv_after_fusion = nn.Conv2d(
            in_channels=1280 + 128,
            out_channels=256,
            kernel_size=1,
            bias=False
        )
        self.bn_after_fusion = nn.BatchNorm2d(256)
        self.relu_after_fusion = nn.ReLU(inplace=True)

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )

        self.elm = ELMLayer(
            input_dim=256,
            hidden_units=512,
            output_units=num_classes
        )

    def forward(self, x):
        z = self.backbone(x)

        r1 = self.residual_block1(z)
        p1 = self.projection_path1(z)
        fused1 = r1 + p1

        r2 = self.residual_block2(fused1)
        p2 = self.projection_path2(fused1)
        fused2 = r2 + p2

        fused2_upsampled = nn.functional.interpolate(
            fused2, size=z.shape[2:], mode="bilinear", align_corners=False
        )
        fused = torch.cat([z, fused2_upsampled], dim=1)

        x = self.conv_after_fusion(fused)
        x = self.bn_after_fusion(x)
        x = self.relu_after_fusion(x)

        x = torch.mean(x, dim=(2, 3))  

        x = x.unsqueeze(1) 
        lstm_out, _ = self.lstm(x)
        features = lstm_out[:, -1, :]

    
        return self.elm(features)
