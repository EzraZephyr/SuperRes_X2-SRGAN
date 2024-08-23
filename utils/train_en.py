import csv
import time
import numpy as np
from PIL import Image
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from generator_en import *
from discriminator_en import *

def content_loss(output, target):
    return F.mse_loss(output, target)
    # Computes the mean squared error between generated images

def train():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    # Defines the transformation method: converts to tensor and normalizes to the range [-1,1]

    high = datasets.ImageFolder('../data/Urban 100/X2 Urban100/X2/HIGH Urban', transform=transform)
    hr_loader = DataLoader(high, batch_size=1, shuffle=False)

    low = datasets.ImageFolder('../data/Urban 100/X2 Urban100/X2/LOW Urban', transform=transform)
    lr_loader = DataLoader(low, batch_size=1, shuffle=False)
    # Loads the dataset, applies transformations, and packages it into an iterator
    # An extra folder layer is required before the images since ImageFolder processes an extra label layer,
    # even though it is unused

    generator = Generator().cuda()
    discriminator = Discriminator().cuda()
    perceptual_loss = PerceptualLoss().cuda()
    g_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.5, 0.999))
    # Initializes the generator, discriminator, and perceptual loss, and defines the Adam optimizer

    train_csv = '../train.csv'
    with open(train_csv, 'w', newline="") as f:
        fieldnames = ['Epoch', 'g_loss', 'd_loss']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Creates a CSV file to record losses

    num_epochs = 200
    for epoch in range(num_epochs):

        g_total_loss = 0
        d_total_loss = 0
        start_time = time.time()
        first_batch = True
        # Resets total losses and time for each training epoch

        for (lr_batch, hr_batch) in zip(lr_loader, hr_loader):
            lr_imgs, _ = lr_batch
            hr_imgs, _ = hr_batch
            lr_imgs = lr_imgs.cuda().float()
            hr_imgs = hr_imgs.cuda().float()
            # Extracts images and labels from the data loader
            # Since labels are not needed, they are discarded, and the images are moved to the GPU (CUDA)
            # Converts the data to float type to ensure precision in subsequent calculations

            if first_batch:
                if epoch == 0 or (epoch + 1) % 10 == 0:
                    save_image(generator, lr_imgs[0], epoch + 1)
                first_batch = False
                # Saves the generated image every ten epochs, using only the first image

            real_labels = torch.full((hr_imgs.size(0), 1), 0.9, dtype=torch.float, device='cuda')
            fake_labels = torch.full((hr_imgs.size(0), 1), 0.0, dtype=torch.float, device='cuda')
            # Uses label smoothing, setting real labels to 0.9 instead of 1,
            # which helps the discriminator generalize better
            # Prevents the discriminator from becoming too strong, which could make generator training difficult

            real_output = discriminator(hr_imgs).view(-1, 1)
            real_loss = F.binary_cross_entropy_with_logits(real_output, real_labels)
            # Adjusts the shape to [batch_size, 1] to fit the discriminator loss calculation method,
            # then computes the loss for real images

            fake_imgs = generator(lr_imgs)
            fake_output = discriminator(fake_imgs.detach()).view(-1, 1)
            fake_loss = F.binary_cross_entropy_with_logits(fake_output, fake_labels)
            # Similarly, computes the loss for generated images

            d_loss = real_loss + fake_loss
            d_total_loss += d_loss.item()
            # Calculates the total loss for real and generated images and adds it to the total loss for each epoch

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            # Backpropagation

            fake_imgs = generator(lr_imgs)
            fake_output = discriminator(fake_imgs).view(-1, 1)
            g_loss = content_loss(fake_imgs, hr_imgs) \
                     + 1e-3 * F.binary_cross_entropy_with_logits(fake_output, fake_output) \
                     + perceptual_loss(fake_imgs, hr_imgs)
            g_total_loss += g_loss.item()
            # This time, computes the generator loss
            # Content loss is the primary component,
            # focusing on pixel-level similarity between generated and real images
            # Adversarial loss helps improve the realism of generated images,
            # enabling the generator to better fool the discriminator
            # Perceptual loss measures the perceptual similarity between generated and target images
            # using high-level features from a deep network

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            # Backpropagation

        d_avg_loss = d_total_loss / len(hr_loader)
        g_avg_loss = g_total_loss / len(hr_loader)
        # Computes the average loss for each epoch

        print(f"Epoch: {epoch+1}, d_loss: {d_avg_loss:.4f}, g_loss: {g_avg_loss:.4f}, time: {(time.time() - start_time):.2f}")

        with open(train_csv, 'a', newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({'Epoch': epoch + 1, 'd_loss': d_avg_loss, 'g_loss': g_avg_loss})
            # Writes the losses to the CSV file

        if (epoch + 1) % 50 == 0:
            torch.save(generator.state_dict(), f'../model/generator_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'../model/discriminator_{epoch+1}.pth')
            # Saves the model every 50 epochs

def save_image(generator, lr_img, epoch):
    generator.eval()
    with torch.no_grad():
        fake_img = generator(lr_img.unsqueeze(0))
        # Adds a batch size dimension to the image before passing it to the generator

        fake_img = fake_img.squeeze().cpu().numpy()
        fake_img = (fake_img * 0.5 + 0.5) * 255.0
        # Converts the image from the normalized range [-1,1] back to the pixel value range [0,255]

        fake_img = np.clip(fake_img, 0, 255).astype(np.uint8)
        fake_img = np.transpose(fake_img, (1, 2, 0))
        # Clamps the pixel values between [0,255], converts to unsigned 8-bit integer, and rearranges dimensions

        Image.fromarray(fake_img).save(f'../images/image_epoch{epoch}.png')
        generator.train()
        # Converts the NumPy array to a PIL image and saves it,
        # then switches back to training mode for further training

train()
