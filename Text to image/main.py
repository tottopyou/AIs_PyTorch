import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoCaptions
import os
from PIL import Image
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

coco_dataset = CocoCaptions(root=coco_dir, annFile=coco_dir + '/annotations/captions_train2017.json',
                            transform=transforms.ToTensor())

class CustomDataset(Dataset):
    def __init__(self, text_tensor, coco_dataset):
        self.text_tensor = text_tensor
        self.coco_dataset = coco_dataset

    def __getitem__(self, index):
        _, caption = self.coco_dataset[index]
        image = self.coco_dataset.get_image(index)
        return self.text_tensor, image

    def __len__(self):
        return len(self.coco_dataset)


# Assuming you have the text tensor ready
text = "a dog sitting on a mat"
text_tensor = get_tensor_from_the_text(text).to(device)

custom_dataset = CustomDataset(text_tensor, coco_dataset)
data_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    for text_tensor_batch, target_images in data_loader:
        print("Batch shape:", text_tensor_batch.shape)
        print("Batch size:", len(text_tensor_batch))
        print("Tensor shape:",
              text_tensor_batch[0].shape)
        print("Tensor size:", text_tensor_batch[0].size())

        print("Number of dimensions:", text_tensor_batch[0].dim())

        for tensor in text_tensor_batch:
            print("Tensor shape:", tensor.shape)
            print("Tensor size:", tensor.size())
            print("Number of dimensions:", tensor.dim())
            text_tensor = text_tensor.to(device)

            optimizer.zero_grad()

            generated_images = model(text_tensor)

            target_images = target_images.to(device)
            target_images = target_images.view(-1, 3, 64, 64)

            # Compute the loss
            loss = criterion(generated_images, target_images)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            print(f"Epoch [{epoch + 1}/{epochs}], Batch Loss: {loss.item():.4f}")
