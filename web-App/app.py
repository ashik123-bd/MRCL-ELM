# app.py
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import os
import time
import psutil
import gc
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ------------------- Model (Same as before) -------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride, bias=False)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
    def forward(self, x):
        out = self.conv(x); out = self.maxpool(out); out = self.bn(out)
        out = self.relu(out); out = self.dropout(out)
        return out

class ProjectionPath(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projection_conv = nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False)
    def forward(self, x):
        return self.projection_conv(x)

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
        self.conv_after_fusion = nn.Conv2d(1280 + 128, 256, 1, bias=False)
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
        fused2_upsampled = torch.nn.functional.interpolate(fused2, size=z.shape[2:], mode='bilinear', align_corners=False)
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

# ------------------- Load Model -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNetLSTMModel().to(device)
model.eval()

MODEL_PATH = "mrcl_elm.pth"  # Change if needed
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
print("Model loaded successfully!")

# Categories
categories = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
              'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Helper: Get GPU memory usage (if available)
def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(device) / 1024**2  # MB
    return 0

# Helper: Get RAM usage increase
process = psutil.Process(os.getpid())

def get_memory_usage_mb():
    return process.memory_info().rss / 1024**2  # MB

# ------------------- Routes -------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Record start memory
            start_memory_gpu = get_gpu_memory()
            start_memory_ram = get_memory_usage_mb()

            # Warmup (optional, stabilizes timing)
            dummy = torch.randn(1, 3, 128, 128).to(device)
            with torch.no_grad():
                _ = model(dummy)

            # Actual inference with timing
            image = Image.open(filepath).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)

            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()

            with torch.no_grad():
                output = model(input_tensor)

            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()

            inference_time_ms = (end_time - start_time) * 1000

            # Memory after inference
            end_memory_gpu = get_gpu_memory()
            end_memory_ram = get_memory_usage_mb()

            # Memory used during inference
            memory_used_gpu = end_memory_gpu - start_memory_gpu
            memory_used_ram = end_memory_ram - start_memory_ram
            memory_used_mb = max(memory_used_gpu, memory_used_ram + 50)  # Approx total

            # Prediction
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = categories[predicted_idx.item()]
            confidence_percent = confidence.item() * 100

            # Cleanup
            os.remove(filepath)
            del input_tensor, output, image
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            return jsonify({
                'prediction': predicted_class,
                'confidence': f"{confidence_percent:.2f}%",
                'inference_time': f"{inference_time_ms:.2f} ms",
                'memory_usage': f"{memory_used_mb:.2f} MB"
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('index.html')

if __name__ == '__main__':
    print("Satellite Image Classifier Web App Running!")
    print("Visit: http://127.0.0.1:5000")
    app.run(debug=False, threaded=True)