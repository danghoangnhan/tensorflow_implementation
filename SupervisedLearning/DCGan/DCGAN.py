import os

from torch import randn, zeros, rand, mean, no_grad
from torch.distributions import Uniform
from torch.optim import Adam
from torch.nn import BCELoss
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from torchvision import datasets as torch_dataset
from tqdm import tqdm

from GPU import DeviceDataLoader
from SupervisedLearning.DCGan import weights_init
from SupervisedLearning.DCGan.Discriminator import Discriminator
from SupervisedLearning.DCGan.Generator import Generator
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.cuda import empty_cache
import torchvision.transforms as T


def denorm(image, normalization_stats):
    return image * normalization_stats[1][0] + normalization_stats[0][0]


class DCGan:
    def __init__(self, z=100, batch_size=8, epoch_numb=50000, learning_rate=0.0002, beta1=0.5, img_size=64,
                 device='cuda',
                 result='./result', data_dir='./data'):

        self.img_list = []
        self.dis_model = Discriminator()
        self.gen_model = Generator()
        self.gen_model.apply(weights_init)
        self.dis_model.apply(weights_init)
        self.dis_model.to(device)
        self.gen_model.to(device)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.img_size = img_size
        self.criterion = BCELoss()
        self.device = device
        self.epoch_numb = epoch_numb
        self.fixed_noise = randn(32, 100, 1, 1, device=device)
        self.RESULTS_DIR = result
        self.datadir = data_dir
        self.batch_size = batch_size
        self.normalization_stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # Convert channels from [0, 1] to [-1, 1]
        self.fixed_latent_batch = randn(64, self.seed_size, 1, 1, device=device)

        print('init model completed')

    def train_generator(self, gen_optimizer, batch_size, seed_size, device):
        # Clear the generator gradients
        gen_optimizer.zero_grad()

        # Generate some fake pokemon
        latent_batch = randn(batch_size, seed_size, 1, 1, device=device)
        fake_pokemon = self.gen_model(latent_batch)

        # Test against the discriminator
        disc_predictions = self.dis_model(fake_pokemon)
        # We want the discriminator to think these images are real.
        targets = zeros(fake_pokemon.size(0), 1, device=device)
        # How well did the generator do? (How much did the discriminator believe the generator?)
        loss = F.binary_cross_entropy(disc_predictions, targets)

        # Update the generator based on how well it fooled the discriminator
        loss.backward()
        gen_optimizer.step()
        # Return generator loss
        return loss.item()

    def train_discriminator(self, real_image, disc_optimizer, batch_size, seed_size, device):
        # Reset the gradients for the optimizer
        disc_optimizer.zero_grad()

        # Train on the real images
        real_predictions = self.dis_model(real_image)

        real_targets = rand(real_image.size(0), 1, device=device) * (0.1 - 0) + 0
        # Add some noisy labels to make the discriminator think harder.

        real_loss = F.binary_cross_entropy(real_predictions, real_targets)
        # Can do binary loss function because it is a binary classifier

        real_score = mean(
            real_predictions).item()  # How well does the discriminator classify the real pokemon? (Higher score is better for the discriminator)

        # Make some latent tensors to seed the generator
        latent_batch = randn(batch_size, seed_size, 1, 1, device=device)

        # Get some fake pokemon
        fake_pokemon = self.gen_model(latent_batch)

        # Train on the generator's current efforts to trick the discriminator
        gen_predictions = self.dis_model(fake_pokemon)
        # gen_targets = torch.ones(fake_pokemon.size(0), 1, device=device)
        # Add some noisy labels to make the discriminator think harder.
        gen_targets = rand(fake_pokemon.size(0), 1, device=device) * (1 - 0.9) + 0.9
        gen_loss = F.binary_cross_entropy(gen_predictions, gen_targets)
        # How well did the discriminator classify the fake pokemon? (Lower score is better for the discriminator)
        gen_score = mean(gen_predictions).item()

        # Update the discriminator weights
        total_loss = real_loss + gen_loss
        total_loss.backward()
        disc_optimizer.step()
        return total_loss.item(), real_score, gen_score

    def train(self, device, start_idx=1):
        # Empty the GPU cache to save some memory
        empty_cache()
        # Track losses and scores
        disc_losses = []
        disc_scores = []
        gen_losses = []
        gen_scores = []

        optim_D = Adam(self.dis_model.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        optim_G = Adam(self.gen_model.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        normal_dataset = ImageFolder(self.datadir, transform=T.Compose([
            T.Resize(self.img_size),
            T.CenterCrop(self.img_size),
            T.ToTensor(),
            T.Normalize(*self.normalization_stats)]))

        # Augment the dataset with mirrored images
        mirror_dataset = ImageFolder(self.datadir, transform=T.Compose([
            T.Resize(self.img_size),
            T.CenterCrop(self.img_size),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize(*self.normalization_stats)]))

        # Augment the dataset with color changes
        color_jitter_dataset = ImageFolder(self.datadir, transform=T.Compose([
            T.Resize(self.img_size),
            T.CenterCrop(self.img_size),
            T.ColorJitter(0.5, 0.5, 0.5),
            T.ToTensor(),
            T.Normalize(*self.normalization_stats)]))

        # Combine the datasets
        dataset_list = [normal_dataset, mirror_dataset, color_jitter_dataset]
        dataset = ConcatDataset(dataset_list)

        dataloader = DataLoader(dataset, self.batch_size, shuffle=True, num_workers=4, pin_memory=False)
        dev_dataloader = DeviceDataLoader(dataloader, device)

        for epoch in range(self.epoch_numb):
            # Go through each image
            for real_img, _ in tqdm(dev_dataloader):
                # Train the discriminator
                disc_loss, real_score, gen_score = self.train_discriminator(real_img, optim_D)

                # Train the generator
                gen_loss = self.train_generator(optim_G)

            # Collect results
            disc_losses.append(disc_loss)
            disc_scores.append(real_score)
            gen_losses.append(gen_loss)
            gen_scores.append(gen_score)

            # Print the losses and scores
            print("Epoch [{}/{}], gen_loss: {:.4f}, disc_loss: {:.4f}, real_score: {:.4f}, gen_score: {:.4f}".format(
                epoch + start_idx, epoch, gen_loss, disc_loss, real_score, gen_score))

            # Save the images and show the progress
            self.save_results(epoch + start_idx, self.fixed_latent_batch, show=False)
        # Return stats
        return disc_losses, disc_scores, gen_losses, gen_scores

    def save_results(self, index, latent_batch, show=True):
        # Generate fake pokemon
        fake_pokemon = self.gen_model(latent_batch)
        # Make the filename for the output
        fake_file = "result-image-{0:0=4d}.png".format(index)
        # Save the image
        save_image(denorm(fake_pokemon,self.normalization_stats), os.path.join(self.RESULTS_DIR, fake_file), nrow=8)
        print("Result Saved!")

        if show:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(fake_pokemon.cpu().detach(), nrow=8).permute(1, 2, 0))

    def cleanup(self):
        import gc
        # del dev_dataloader
        del self.dis_model
        del self.gen_model
        dev_dataloader = None
        discriminator = None
        generator = None
        gc.collect()
        empty_cache()
