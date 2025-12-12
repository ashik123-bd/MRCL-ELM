import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from typing import List, Optional


class ResidualBlock(nn.Module):
    """
    Residual block with convolution, batch normalization, and dropout.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout_rate: float = 0.3
    ):
        """
        Initialize residual block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            stride: Convolution stride
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False
        )
        self.maxpool = nn.MaxPool2d(2, 2)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        out = self.conv(x)
        out = self.maxpool(out)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class ProjectionPath(nn.Module):
    """
    Projection path for skip connections using 1x1 convolutions.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize projection path.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
        """
        super().__init__()
        self.projection_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=2,
            bias=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through projection path."""
        return self.projection_conv(x)


class ELMLayer(nn.Module):
    """
    Extreme Learning Machine (ELM) layer for fast training.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_units: int,
        output_units: int
    ):
        """
        Initialize ELM layer.
        
        Args:
            input_dim: Input dimension
            hidden_units: Number of hidden units
            output_units: Number of output units (classes)
        """
        super().__init__()
        
        # Hidden layer weights (fixed, not trained)
        self.hidden_weights = nn.Parameter(
            torch.randn(input_dim, hidden_units),
            requires_grad=False
        )
        self.hidden_bias = nn.Parameter(
            torch.randn(hidden_units),
            requires_grad=False
        )
        
        # Output layer weights (trained)
        self.output_weights = nn.Parameter(
            torch.randn(hidden_units, output_units)
        )
        
        # Initialize weights
        nn.init.xavier_uniform_(self.hidden_weights)
        nn.init.normal_(self.hidden_bias, 0, 0.1)
        nn.init.xavier_uniform_(self.output_weights)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ELM layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Calculate hidden layer output
        H = torch.relu(x @ self.hidden_weights + self.hidden_bias)
        
        # Calculate output
        output = H @ self.output_weights
        
        return output


class EfficientNetLSTMModel(nn.Module):
    """
    Hybrid EfficientNet-LSTM model for satellite image classification.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        dropout_rates: Optional[List[float]] = None,
        lstm_units: int = 256,
        dense_units: List[int] = None,
        pretrained: bool = True
    ):
        """
        Initialize EfficientNet-LSTM model.
        
        Args:
            num_classes: Number of output classes
            dropout_rates: Dropout rates for residual blocks
            lstm_units: Number of LSTM units
            dense_units: Units in ELM hidden layers
            pretrained: Whether to use pretrained EfficientNet
        """
        super().__init__()
        
        # Set default parameters
        if dropout_rates is None:
            dropout_rates = [0.3, 0.4, 0.5, 0.3]
        if dense_units is None:
            dense_units = [512, 128]
            
        # Load pretrained EfficientNet-B0
        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
            self.efficientnet = efficientnet_b0(weights=weights)
        else:
            self.efficientnet = efficientnet_b0(weights=None)
            
        # Extract backbone (features only)
        self.backbone = nn.Sequential(*list(self.efficientnet.features.children()))
        
        # Residual blocks
        self.residual_block1 = ResidualBlock(
            in_channels=1280,
            out_channels=64,
            dropout_rate=dropout_rates[0]
        )
        self.residual_block2 = ResidualBlock(
            in_channels=64,
            out_channels=128,
            dropout_rate=dropout_rates[1]
        )
        
        # Projection paths for skip connections
        self.projection_path1 = ProjectionPath(1280, 64)
        self.projection_path2 = ProjectionPath(64, 128)
        
        # Feature fusion
        self.conv_after_fusion = nn.Conv2d(
            in_channels=1280 + 128,
            out_channels=256,
            kernel_size=1,
            bias=False
        )
        self.bn_after_fusion = nn.BatchNorm2d(256)
        self.relu_after_fusion = nn.ReLU(inplace=True)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_units,
            batch_first=True,
            num_layers=1
        )
        
        # ELM layer
        self.elm = ELMLayer(
            input_dim=lstm_units,
            hidden_units=dense_units[0],
            output_units=num_classes
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # EfficientNet backbone
        z = self.backbone(x)
        
        # First residual block with skip connection
        r1 = self.residual_block1(z)
        p1 = self.projection_path1(z)
        fused1 = r1 + p1
        
        # Second residual block with skip connection
        r2 = self.residual_block2(fused1)
        p2 = self.projection_path2(fused1)
        fused2 = r2 + p2
        
        # Upsample and concatenate with backbone features
        fused2_upsampled = F.interpolate(
            fused2,
            size=z.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        final_fused = torch.cat([z, fused2_upsampled], dim=1)
        
        # Feature fusion convolution
        x = self.conv_after_fusion(final_fused)
        x = self.bn_after_fusion(x)
        x = self.relu_after_fusion(x)
        
        # Global average pooling
        x = torch.mean(x, dim=(2, 3))
        
        # Prepare for LSTM (add sequence dimension)
        x = x.unsqueeze(1)
        
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        lstm_features = lstm_out[:, -1, :]
        
        # ELM layer
        output = self.elm(lstm_features)
        
        return output
    
    def get_feature_extractor(self) -> nn.Module:
        """
        Get feature extractor part of the model (before classification layer).
        
        Returns:
            Feature extractor module
        """
        return nn.Sequential(
            self.backbone,
            self.residual_block1,
            self.residual_block2,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count number of parameters in model.
    
    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def save_model_architecture(model: nn.Module, filepath: str = "model_architecture.txt"):
    """
    Save model architecture to file.
    
    Args:
        model: PyTorch model
        filepath: Path to save architecture
    """
    with open(filepath, 'w') as f:
        f.write(str(model))
        
    # Also save detailed information
    with open(filepath.replace('.txt', '_detailed.txt'), 'w') as f:
        f.write("Model Architecture Details\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total parameters: {count_parameters(model, False):,}\n")
        f.write(f"Trainable parameters: {count_parameters(model, True):,}\n\n")
        
        f.write("Layer-wise details:\n")
        f.write("-" * 30 + "\n")
        for name, module in model.named_modules():
            if name:  # Skip empty name for root module
                num_params = sum(p.numel() for p in module.parameters())
                f.write(f"{name}: {num_params:,} parameters\n")
                
    print(f"Model architecture saved to {filepath}")
