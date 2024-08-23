import csv
import time
import numpy as np
from PIL import Image
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from generator_zh import *
from discriminator_zh import *


def content_loss(output,target):
    return F.mse_loss(output,target)
    # 计算生成图像之间的均方误差

def train():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])
    # 定义转化方法 转为张量并标准化到[-1,1]之间

    high = datasets.ImageFolder('../data/Urban 100/X2 Urban100/X2/HIGH Urban', transform=transform)
    hr_loader = DataLoader(high, batch_size=1, shuffle=False)

    low = datasets.ImageFolder('../data/Urban 100/X2 Urban100/X2/LOW Urban', transform=transform)
    lr_loader = DataLoader(low, batch_size=1, shuffle=False)
    # 加载数据集 并调用转化方法和打包为一个迭代器
    # 这里要在图片前再加一层文件夹 因为用的是ImageFolder 需要处理一层标签 虽然无用

    generator = Generator().cuda()
    discriminator = Discriminator().cuda()
    perceptual_loss = PerceptualLoss().cuda()
    g_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.5, 0.999))
    # 初始化生成器判别器和感知损失 并定义Adam优化器

    train_csv = '../train.csv'
    with open(train_csv, 'w', newline="") as f:
        fieldnames = ['Epoch', 'g_loss', 'd_loss']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # 创建记录损失的csv文件

    num_epochs = 200
    for epoch in range(num_epochs):

        g_total_loss = 0
        d_total_loss = 0
        start_time = time.time()
        first_batch = True
        # 刷新每一轮训练的总损失和时间

        for (lr_batch, hr_batch) in zip(lr_loader, hr_loader):
            lr_imgs, _ = lr_batch
            hr_imgs, _ = hr_batch
            lr_imgs = lr_imgs.cuda().float()
            hr_imgs = hr_imgs.cuda().float()
            # 从数据加载器取出图片和标签
            # 因为标签无用 所以需要把他剔除 并移到coda上
            # 并将数据类型转为浮点型 确保后续计算的精度

            if first_batch:
                if epoch == 0 or (epoch+1) % 10 == 0:
                    save_image(generator,lr_imgs[0], epoch+1)
                first_batch = False
                # 每十轮保存一次生成的图片 只取第一张

            real_labels = torch.full((hr_imgs.size(0), 1), 0.9, dtype=torch.float, device='cuda')
            fake_labels = torch.full((hr_imgs.size(0), 1), 0.0, dtype=torch.float, device='cuda')
            # 使用标签平滑技术 让真实标签被设置为0.9而不是1 可以帮助判别器更好的泛化
            # 防止判别器过强导致生成器训练困难

            real_output = discriminator(hr_imgs).view(-1,1)
            real_loss = F.binary_cross_entropy_with_logits(real_output, real_labels)
            # 将形状调整为[batch_size,1]以拟合计算判别器损失方法 计算真是图像的损失

            fake_imgs = generator(lr_imgs)
            fake_output = discriminator(fake_imgs.detach()).view(-1,1)
            fake_loss = F.binary_cross_entropy_with_logits(fake_output, fake_labels)
            # 同理 计算生成图像的损失

            d_loss = real_loss + fake_loss
            d_total_loss += d_loss.item()
            # 计算真实图像和生成图像的总损失 并加到每一个epoch的总损失里

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            # 反向传播

            fake_imgs = generator(lr_imgs)
            fake_output = discriminator(fake_imgs).view(-1,1)
            g_loss = content_loss(fake_imgs, hr_imgs)\
                     + 1e-3 * F.binary_cross_entropy_with_logits(fake_output, fake_output)\
                     + perceptual_loss(fake_imgs, hr_imgs)
            g_total_loss += g_loss.item()
            # 这次计算生成器损失
            # 内容损失是主要部分 用于关注生成图像与真实图像在像素级别的相似性
            # 对抗性损失是用于提升生成图像的真实性 使得生成器能够更好地欺骗判别器
            # 感知损失是通过深度网络的高层次特征来衡量生成图像与目标图像的感知相似度

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            # 反向传播

        d_avg_loss = d_total_loss / len(hr_loader)
        g_avg_loss = g_total_loss / len(hr_loader)
        # 计算每一轮的平均损失

        print(f"Epoch: {epoch+1}, d_loss: {d_avg_loss:.4f}, g_loss: {g_avg_loss:.4f}, time: {(time.time() - start_time):.2f}")

        with open(train_csv, 'a', newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({'Epoch': epoch+1,'d_loss': d_avg_loss, 'g_loss': g_avg_loss})
            # 将损失写入csv文件中

        if (epoch+1) % 50 == 0:
            torch.save(generator.state_dict(), f'../model/generator_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'../model/discriminator_{epoch+1}.pth')
            # 每50个epoch储存一次模型
def save_image(generator,lr_img,epoch):
    generator.eval()
    with torch.no_grad():
        fake_img = generator(lr_img.unsqueeze(0))
        # 将图像增加一个batchsize维度传入生成器

        fake_img = fake_img.squeeze().cpu().numpy()
        fake_img = (fake_img*0.5+0.5)*255.0
        # 将图像从[-1,1]的标准化范围还原到[0,255]的像素值范围

        fake_img = np.clip(fake_img, 0, 255).astype(np.uint8)
        fake_img = np.transpose(fake_img, (1, 2, 0))
        # 将图像的像素值控制在[0,255]之间并转换为无符号8位整数
        # 并将图像的维度从最后一个位置移动到第一个位置

        Image.fromarray(fake_img).save(f'../images/image_epoch{epoch}.png')
        generator.train()
        # 将Numpy转换为PIL图像并保存 随后切换到训练模式以便后续训练


train()