from torch import optim, randn, device, no_grad
from torch.nn import BCELoss
from torchvision.utils import make_grid

from SupervisedLearning.DCGan.DCGAN import Generator
from SupervisedLearning.DCGan.Discriminator import Discriminator
from SupervisedLearning.DCGan import weights_init
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, dataloader
import torchvision.transforms as T
from torchvision import datasets as torch_dataset


real_label = 1.
fake_label = 0.
lr = 0.0002
beta1 = 0.5

device = device('cpu')
dis_model = Discriminator()
gen_model = Generator()
img_size = 64
data_dir = './data/'
data_transforms = T.Compose([
    T.Resize(img_size),
    T.CenterCrop(img_size),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

anime_dataset = torch_dataset.ImageFolder(root=data_dir, transform=data_transforms)
dataloader = DataLoader(dataset=anime_dataset, batch_size=128, shuffle=True, num_workers=4)

img_batch = next(iter(dataloader))[0]
combine_img = make_grid(img_batch[:32], normalize=True, padding=2).permute(1, 2, 0)
plt.figure(figsize=(15, 15))
plt.imshow(combine_img)
plt.show()

gen_model.apply(weights_init)
dis_model.apply(weights_init)
dis_model.to(device)
gen_model.to(device)
print('init model')

criterion = BCELoss()
optim_D = optim.Adam(dis_model.parameters(), lr=lr, betas=(beta1, 0.999))
optim_G = optim.Adam(gen_model.parameters(), lr=lr, betas=(beta1, 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 1
epoch_nb = 30
fixed_noise = randn(32, 100, 1, 1, device=device)

from torch.distributions.uniform import Uniform

for epoch in range(epoch_nb):
    for i, data in enumerate(dataloader):
        # Train Discriminator
        ## Train with real image
        dis_model.zero_grad()
        real_img = data[0].to(device)
        bz = real_img.size(0)

        #  label smoothing
        label = Uniform(0.9, 1.0).sample((bz,)).to(device)

        output = dis_model(real_img).view(-1)
        error_real = criterion(output, label)
        error_real.backward()
        D_x = output.mean().item()

        ## Train with fake image
        noise = randn(bz, 100, 1, 1, device=device)
        fake_img = gen_model(noise)
        label = Uniform(0., 0.05).sample((bz,)).to(device)

        output = dis_model(fake_img.detach()).view(-1)
        error_fake = criterion(output, label)
        error_fake.backward()
        D_G_z1 = output.mean().item()
        error_D = error_real + error_fake
        optim_D.step()

        ## Train Generator
        gen_model.zero_grad()
        label = Uniform(0.95, 1.0).sample((bz,)).to(device)
        output = dis_model(fake_img).view(-1)
        error_G = criterion(output, label)
        error_G.backward()
        optim_G.step()
        D_G_z2 = output.mean().item()

        if i % 300 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epoch_nb, i, len(dataloader),
                     error_D.item(), error_G.item(), D_x, D_G_z1, D_G_z2))
        if epoch > 1:
            if (iters % 1000 == 0) or ((epoch == epoch_nb - 1) and (i == len(dataloader) - 1)):
                with no_grad():
                    fake_img = gen_model(fixed_noise).detach().cpu()
                fake_img = make_grid(fake_img, padding=2, normalize=True)
                img_list.append(fake_img)
                plt.figure(figsize=(10, 10))
                plt.imshow(img_list[-1].permute(1, 2, 0))
                plt.show()
        iters += 1
