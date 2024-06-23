import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvertToRGBA(object):
    def __call__(self, image):
        if image.mode == 'P' and 'transparency' in image.info:
            image = image.convert('RGBA')
        return image

transform = transforms.Compose([
    ConvertToRGBA(),
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = Autoencoder()
model.load_state_dict(torch.load('/Users/macbook/Oxford/Models/autoencoder_ox64.pth', map_location=torch.device('cpu')))
model.to(device)
model.eval()

input_dir = '/Users/macbook/Oxford/Oxford'
output_dir = '/Users/macbook/Oxford/Oxford64'

os.makedirs(output_dir, exist_ok=True)

for image_name in os.listdir(input_dir):
    if image_name.endswith(('jpg', 'png', 'jpeg')):
        image_path = os.path.join(input_dir, image_name)
        test_image = Image.open(image_path)
        original_width, original_height = test_image.size
        test_image = transform(test_image).unsqueeze(0)
        test_image = test_image.to(device)
        with torch.no_grad():
            encoded_image = model.encoder(test_image)
            reconstructed_image = model.decoder(encoded_image)
        reconstructed_image_np = reconstructed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        reconstructed_image_pil = Image.fromarray((reconstructed_image_np * 255).astype(np.uint8))
        reconstructed_image_resized = reconstructed_image_pil.resize((original_width, original_height), Image.BILINEAR)
        save_path = os.path.join(output_dir, image_name)
        reconstructed_image_resized.save(save_path)
        print(f'Reconstructed resized image saved at {save_path}')