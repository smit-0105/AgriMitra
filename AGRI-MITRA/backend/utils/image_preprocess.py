from torchvision import transforms
from PIL import Image
import io

# Define the *exact* same transforms used during training
# This is critical for model accuracy
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # These are standard ImageNet normalization stats
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(image_bytes):
    """
    Takes image bytes, opens with PIL, applies transforms,
    and returns a tensor ready for the model.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = data_transform(image)
    # Add a batch dimension (C, H, W) -> (B, C, H, W)
    tensor = tensor.unsqueeze(0)
    return tensor