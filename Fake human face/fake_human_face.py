import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
import torchvision.transforms as transforms
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Activated device:", device)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

celeba_dataset = CelebA(root='path_to_celeba_dataset',
                        split='train',
                        transform=transform,
                        download=True)

batch_size = 128
celeba_dataloader = DataLoader(dataset=celeba_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=8)

class Generator(nn.Module):
    def __init__(self, input_shape, hidden_layer_gen, layers):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_shape, hidden_layer_gen*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_layer_gen*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_layer_gen*8, hidden_layer_gen*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_layer_gen*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_layer_gen*4, hidden_layer_gen*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_layer_gen*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_layer_gen*2, hidden_layer_gen, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_layer_gen),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_layer_gen, layers, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, layers, hidden_layer_dis):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(layers, hidden_layer_dis, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_layer_dis, hidden_layer_dis*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_layer_dis*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_layer_dis*2, hidden_layer_dis*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_layer_dis*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_layer_dis*4, hidden_layer_dis*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_layer_dis*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_layer_dis*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


input_shape = 100
hidden_layer_gen = 64
hidden_layer_dis = 64
layers = 3

model_generator = Generator(input_shape, hidden_layer_gen, layers).to(device)
model_discriminator = Discriminator(layers, hidden_layer_dis).to(device)

loss_fn = nn.BCELoss()

lr = 0.0002
beta1 = 0.5

optimizerG = optim.Adam(model_generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(model_discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

epochs = 10
if __name__ == '__main__':
    for epoch in range(epochs):
        for i, data in enumerate(celeba_dataloader, 0):
            model_discriminator.zero_grad()

            real_images = data[0].to(device)
            real_labels = torch.full((real_images.size(0),), 1., dtype=torch.float, device=device)

            output = model_discriminator(real_images).view(-1)
            errD_real = loss_fn(output, real_labels)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(real_images.size(0), input_shape, 1, 1, device=device)
            fake_images = model_generator(noise)
            fake_labels = torch.full((real_images.size(0),), 0., dtype=torch.float, device=device)

            output = model_discriminator(fake_images.detach()).view(-1)
            errD_fake = loss_fn(output, fake_labels)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            model_generator.zero_grad()
            fake_labels.fill_(1)
            output = model_discriminator(fake_images).view(-1)
            errG = loss_fn(output, fake_labels)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 100 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, epochs, i, len(celeba_dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            if i % 1000 == 0:
                if not os.path.exists("./results"):
                    os.makedirs("./results")
                vutils.save_image(real_images, '%s/real_samples.png' % "./results", normalize=True)
                fake = model_generator(noise)
                vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize=True)