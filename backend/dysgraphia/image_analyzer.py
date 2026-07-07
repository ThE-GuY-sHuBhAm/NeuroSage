import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, ConvertImageDtype, Pad, Resize, PILToTensor
from model import EncoderFactory  # <--- Changed from ResnetWrapper
import os

class ImageAnalyzer:
    """Handles handwriting image analysis using the trained model"""
    
    # You can change model_name to 'mobilenet_v3' or 'resnet18' easily now
    def __init__(self, device='cpu', model_name='mobilenet_v3'):
        self.device = device
        self.model_name = model_name
        self.model = None
        self.baseline_vector = None
        
        # Hard-coded stats from training (Must match evaluate_experiment.py)
        self.max_width = 2479
        self.max_height = 3542
        
        # Image threshold (Youden-optimal threshold for MobileNetV3-Small)
        self.threshold = 52.14
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and baseline vector"""
        try:
            # Construct dynamic filenames based on the model name
            model_filename = f'{self.model_name}_model_best.pth'
            baseline_filename = f'baseline_{self.model_name}.pt'
            
            # Check paths
            model_path = os.path.join('checkpoints', model_filename)
            # If not in checkpoints, check root (just in case)
            if not os.path.exists(model_path):
                model_path = model_filename

            baseline_path = baseline_filename
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(baseline_path):
                raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
            
            # Load model using the new Factory
            print(f"Loading {self.model_name}...")
            wrapper = EncoderFactory(backbone_name=self.model_name, device=self.device)
            
            # Wrapper's load_state handles the path logic internally usually, 
            # but we pass the direct filename here to be safe if it's in root vs checkpoints
            try:
                # Try loading from the full path we found
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'state_dict' in checkpoint:
                    wrapper.load_state_dict(checkpoint['state_dict'])
                else:
                    wrapper.load_state_dict(checkpoint)
            except Exception as e:
                print(f"Direct load failed, trying wrapper method: {e}")
                wrapper.load_state(model_filename)

            self.model = wrapper.get_model()
            self.model.eval()
            
            # Load baseline
            self.baseline_vector = torch.load(baseline_path, map_location=self.device)
            
            print(f"{self.model_name} loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Don't raise here if you want the app to start even if model fails (optional)
            # raise 
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None and self.baseline_vector is not None
    
    def get_anomaly_score(self, image):
        """
        Analyze a PIL Image and return anomaly score (0-100)
        """
        if not self.is_loaded():
            print("Model not loaded, attempting reload...")
            self._load_model()
            if not self.is_loaded():
                return None
        
        try:
            # Convert to grayscale
            img = image.convert('L')
            img_w, img_h = img.size
            
            # Transform
            transform = Compose([
                PILToTensor(),
                ConvertImageDtype(torch.float),
                Pad((0, 0, self.max_width - img_w, self.max_height - img_h), fill=1.),
                Resize((128, 1024))
            ])
            
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                image_vector = self.model(img_tensor)
            
            # Calculate similarity
            similarity = F.cosine_similarity(
                image_vector, 
                self.baseline_vector.unsqueeze(0)
            )
            
            anomaly_score = (1 - similarity.item()) * 100
            
            return anomaly_score
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None