import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoCaptions
import os
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm
from text_to_tensor import get_tensor_from_the_text
from text_encoder import TextEncoder
from image_decoder import ImageDecoder

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Activated device:", device)

text_encoder = torch.load("pretrained_text_encoder.pth")
image_decoder = torch.load("pretrained_image_decoder.pth")

class TextToImageDiffusionModel(nn.Module):
    def __init__(self, text_encoder, image_decoder):
        super(TextToImageDiffusionModel, self).__init__()
        self.text_encoder = text_encoder
        self.image_decoder = image_decoder

    def forward(self, text_tensor):
        x = self.text_encoder(text_tensor)
        generated_image = self.image_decoder(x)
        return generated_image

model = TextToImageDiffusionModel(text_encoder, image_decoder).to(device)

coco_dir = "dataset"

if not os.path.exists(coco_dir):
    print("COCO dataset not found. Please download it manually from https://cocodataset.org/#download.")
    exit()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

coco_dataset = CocoCaptions(root=os.path.join(coco_dir, 'train2017'),
                            annFile=os.path.join(coco_dir, 'annotations', 'captions_train2017.json'),
                            transform=transform)

class CustomDataset(Dataset):
    def __init__(self, coco_dataset):
        self.coco_dataset = coco_dataset

    def __getitem__(self, index):
        image, _ = self.coco_dataset[index]
        return image

    def __len__(self):
        return len(self.coco_dataset)

custom_dataset = CustomDataset(coco_dataset)
data_loader = DataLoader(custom_dataset, batch_size=64, shuffle=True, num_workers=4)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
print(len(data_loader))

output_dir = "results"
if __name__ == '__main__':
    for epoch in range(epochs):
        with tqdm(total=len(data_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for i, target_images in enumerate(data_loader):
                target_images = [img.to(device) for img in target_images]

                optimizer.zero_grad()

                text = "a dog sitting on a mat"
                text_tensor = get_tensor_from_the_text(text).to(device)

                generated_images = model(text_tensor)

                target_images = torch.stack(target_images).view(-1, 3, 64, 64)

                loss = criterion(generated_images, target_images)

                loss.backward()
                optimizer.step()

                pbar.update(1)

                if i % 1000 == 0:
                    print(f"Epoch [{epoch + 1}/{epochs}], Batch Loss: {loss.item():.4f}")
                    vutils.save_image(generated_images, os.path.join(output_dir, f"generated_epoch_{epoch}_batch_{i}.png"))