import torchvision.transforms as transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

# Define transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),           # Convert PIL Image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize pixel values to [-1, 1]
])

# Load CelebA dataset
celeba_dataset = CelebA(root='path_to_celeba_dataset',
                        split='train',
                        transform=transform,
                        download=True)  # Change to True if you haven't downloaded yet

# Create DataLoader
batch_size = 32
celeba_dataloader = DataLoader(dataset=celeba_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4)

