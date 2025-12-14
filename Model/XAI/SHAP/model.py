import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class ELMLayer(nn.Module):
    def __init__(self, input_dim, hidden_units, output_units):
        super().__init__()
        self.hidden_weights = nn.Parameter(torch.randn(input_dim, hidden_units), requires_grad=False)
        self.hidden_bias = nn.Parameter(torch.randn(hidden_units), requires_grad=False)
        self.output_weights = nn.Parameter(torch.randn(hidden_units, output_units))

        nn.init.xavier_uniform_(self.hidden_weights)
        nn.init.normal_(self.hidden_bias, 0, 0.1)
        nn.init.xavier_uniform_(self.output_weights)

    def forward(self, x):
        H = torch.relu(x @ self.hidden_weights + self.hidden_bias)
        return H @ self.output_weights


class EfficientNetLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(self.efficientnet.features.children()))

     
        self.residual_block1 = ResidualBlock(1280, 64, dropout_rate=0.3)
        self.residual_block2 = ResidualBlock(64, 128, dropout_rate=0.4)

        self.projection_path1 = ProjectionPath(1280, 64)
        self.projection_path2 = ProjectionPath(64, 128)

        self.conv_after_fusion = nn.Conv2d(1280 + 128, 256, kernel_size=1, bias=False)
        self.bn_after_fusion = nn.BatchNorm2d(256)
        self.relu_after_fusion = nn.ReLU(inplace=True)

        self.lstm = nn.LSTM(input_size=256, hidden_size=256, batch_first=True, num_layers=1)
        self.elm = ELMLayer(256, 512, 10)

    def forward(self, x):
        z = self.backbone(x)                     

        r1 = self.residual_block1(z)
        p1 = self.projection_path1(z)
        fused1 = r1 + p1

        r2 = self.residual_block2(fused1)
        p2 = self.projection_path2(fused1)
        fused2 = r2 + p2                                  
        fused2_upsampled = F.interpolate(
            fused2, size=z.shape[2:], mode='bilinear', align_corners=False
        )
        final_fused = torch.cat([z, fused2_upsampled], dim=1)   

        x = self.conv_after_fusion(final_fused)
        x = self.bn_after_fusion(x)
        x = self.relu_after_fusion(x)

        x = torch.mean(x, dim=(2, 3))         
        x = x.unsqueeze(1)                    

        lstm_out, _ = self.lstm(x)            
        lstm_features = lstm_out[:, -1, :]    

       
        out = self.elm(lstm_features)
        return out