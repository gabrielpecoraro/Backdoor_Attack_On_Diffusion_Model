# Description: This script trains a simple PixelCNN model on the MNIST dataset using the diffusion model.

import time
import random
import argparse
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw, ion
from typing import List
from jaxtyping import Float
from tqdm import tqdm
from model import Model

random.seed(1)

ion()
draw()


class Training:
    def __init__(self):
        """
        Initialize the training class. This class trains a simple PixelCNN model on the MNIST dataset using the diffusion model.

        Args:
            None

        Returns:
            None
        """

        # Load MNIST dataset
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.train_dataset = torchvision.datasets.MNIST(
            "~/.pytorch/MNIST_data/",
            download=True,
            train=True,
            transform=self.transform,
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=2, shuffle=False
        )

        # Initialize variables
        self.C = 1  # number of channels
        self.W = 28  # width of image (pixels)
        self.H = 28  # height of image (pixels)
        self.image = self.train_dataset[0][0]  # noqa: F722
        self.label = self.train_dataset[0][1]

        self.T = 50  # number of steps
        self.betas = torch.linspace(1e-4, 1e-1, self.T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.Tensor(np.cumprod(self.alphas))
        self.sigmas = torch.sqrt(self.betas)

        self.N_EPOCHS = 1
        self.LOGGING_STEPS = 5

        # Initialize model
        self.config = {
            "data": {
                "image_size": 28,
            },
            "model": {
                "type": "simple",
                "in_channels": 1,
                "out_ch": 1,
                "ch": 128,
                "ch_mult": [
                    1,
                    2,
                    2,
                ],
                "num_res_blocks": 2,
                "attn_resolutions": [
                    1,
                ],
                "dropout": 0.1,
                "resamp_with_conv": True,
            },
            "diffusion": {
                "num_diffusion_timesteps": self.T,
            },
            "runner": {
                "n_epochs": self.N_EPOCHS,
                "logging_steps": self.LOGGING_STEPS,
            },
        }

        # Convert config to namespace
        self.config = self.dict2namespace(self.config)

        self.model = Model(self.config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-4)

        # Set device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)

    def dict2namespace(self, config):
        """
        Convert dictionary to namespace.

        Args:
            config (): configuration dictionary

        Returns:
            namespace : namespace object
        """
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = self.dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    def show_grid(self, imgs: List[np.ndarray], title=""):
        """Display a grid of images.

        Args:
            imgs (List[np.ndarray]): Image list.
            title (str, optional): _description_. Defaults to "".
        """
        fig, ax = plt.subplots()
        imgs = [
            (img - img.min()) / (img.max() - img.min()) for img in imgs
        ]  # Normalize to [0, 1] for imshow()
        img = torchvision.utils.make_grid(imgs, padding=1, pad_value=1).numpy()
        ax.set_title(title)
        ax.imshow(np.transpose(img, (1, 2, 0)), cmap="gray")
        ax.set(xticks=[], yticks=[])
        # plt.show()

    def one_forward_process(self, x_0):
        """Perform one forward process.

        Args:
            x_0 (tensor): Image tensor to be processed.

        Returns:
           x_1 (tensor): Processed image tensor.
           espilon_0 (tensor): Noise tensor.
        """
        # Generate noise
        random.seed(3)
        epsilon_0: Float[torch.Tensor, "1 28 28"] = torch.randn(x_0.shape)
        beta_1 = 0.01
        x_1: Float[torch.Tensor, "1 28 28"] = (
            np.sqrt(1 - beta_1) * x_0 + np.sqrt(beta_1) * epsilon_0
        )
        return x_1, epsilon_0

    def train(self):
        """Train diffusion model."""
        n_epochs: int = self.config.runner.n_epochs
        logging_steps: int = self.config.runner.logging_steps

        losses: List[float] = []
        self.model.train()
        loss = 0
        for epoch in range(n_epochs):
            train_loss: float = 0.0
            recent_losses: List[float] = []

            start_time = time.time()
            for batch_idx, (x_0, _) in enumerate(self.train_dataloader):
                B: int = x_0.shape[0]  # batch size
                x_0: Float[torch.Tensor, "B 1 28 28"] = x_0.to(self.device)
                t: Float[torch.Tensor, "B"] = torch.randint(0, self.T, (B,))
                epsilon: Float[torch.Tensor, "B 1 28 28"] = torch.randn(
                    x_0.shape, device=self.device
                )
                x_0_coef = (
                    torch.sqrt(self.alpha_bars[t]).reshape(-1, 1, 1, 1).to(self.device)
                )
                epsilon_coef = (
                    torch.sqrt(1 - self.alpha_bars[t])
                    .reshape(-1, 1, 1, 1)
                    .to(self.device)
                )
                x_t: Float[torch.Tensor, "B 1 28 28"] = (
                    x_0_coef * x_0 + epsilon_coef * epsilon
                )
                epsilon_theta: Float[torch.Tensor, "B 1 28 28"] = self.model(
                    x_t, t.to(self.device)
                )
                loss: float = torch.sum((epsilon - epsilon_theta) ** 2)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                losses.append(loss.item())
                recent_losses.append(loss.item())
                train_loss += loss.item()
                if (batch_idx + 1) % logging_steps == 0:
                    deno = logging_steps * B
                    print(
                        "Loss over last {} batches: {:.4f} | Time (s): {:.4f}".format(
                            logging_steps,
                            (train_loss / deno),
                            (time.time() - start_time),
                        )
                    )
                    train_loss = 0.0
                    start_time = time.time()

                # Early stopping condition
                if len(recent_losses) > 5:
                    recent_losses.pop(0)
                if len(recent_losses) == 5 and np.mean(recent_losses) < 35:
                    print("Early stopping triggered.")
                    return self.model, losses

        return self.model, losses

    def inference(
        self,
        model,
        config,
        n_samples: int,
        T: int,
        alphas,
        alpha_bars,
        sigmas,
        seed: int = 1,
    ):
        """Generate images from diffusion model."""
        model.eval()
        torch.manual_seed(seed)
        # Dimensions
        n_channels: int = config.model.in_channels  # 1 for grayscale
        H, W = config.data.image_size, config.data.image_size  # 28 pixels
        # x_T \sim N(0, I)
        x_T: Float[torch.Tensor, "n_samples 28 28"] = torch.randn(
            (n_samples, n_channels, W, W)
        )
        # For t = T ... 1
        x_t = x_T
        x_ts = []  # save image as diffusion occurs
        for t in tqdm(range(T - 1, -1, -1)):
            # z \sim N(0, I) if t > 1 else z = 0
            z: Float[torch.Tensor, "n_samples 1 28 28"] = (
                torch.randn(x_t.shape) if t > 1 else torch.zeros_like(x_t)
            )
            # Setup terms for x_t-1
            t_vector: Float[torch.Tensor, "n_samples"] = torch.fill(
                torch.zeros((n_samples,)), t
            )
            epsilon_theta: Float[torch.Tensor, "n_samples 1 28 28"] = model(
                x_t.to(model.device), t_vector.to(model.device)
            ).to("cpu")
            # x_t-1 = (1 / sqrt(alpha_t)) * (x_t - (1 - alpha_t) / (sqrt(1 - alpha_bar_t)) * epsilon_theta(x_t, t)) + sigma_t * z
            x_t_1: Float[torch.Tensor, "n_samples 1 28 28"] = (
                1
                / torch.sqrt(alphas[t])
                * (
                    x_t
                    - (1 - alphas[t]) / torch.sqrt(1 - alpha_bars[t]) * epsilon_theta
                )
                + sigmas[t] * z
            )
            x_ts.append(x_t)
            x_t = x_t_1
        return torch.stack(x_ts).transpose(0, 1)

    def run(self):
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print(
                    "MPS not available because the current PyTorch install was not "
                    "built with MPS enabled."
                )
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."
                )
        print("Using device: {}".format(self.device))
        self.model, losses = self.train()
        torch.save(self.model.state_dict(), f"model.pt")

        plt.plot(losses)
        # plt.show()

        # Load model for inference
        model = Model(self.config)
        model.load_state_dict(torch.load(f"model.pt"))
        model = model.to(self.device)

        # Inference stage
        n_samples = 10  # Number of samples to generate
        generated_images = self.inference(
            model,
            self.config,
            n_samples,
            self.T,
            self.alphas,
            self.alpha_bars,
            self.sigmas,
        )
        self.show_grid(generated_images[-1], title="Generated Images")

        # Find closest image to x_0 in our training dataset
        for x_0 in generated_images[:, -1, ...]:
            x_0 = x_0.squeeze(1)

            # Rescale to be between [-1, 1]
            x_0 = 2 * (x_0 - x_0.min()) / (x_0.max() - x_0.min()) - 1
            train_min = (
                self.train_dataset.data.view(self.train_dataset.data.shape[0], -1)
                .min(dim=1, keepdim=True)[0]
                .unsqueeze(2)
            )
            train_max = (
                self.train_dataset.data.view(self.train_dataset.data.shape[0], -1)
                .max(dim=1, keepdim=True)[0]
                .unsqueeze(2)
            )
            train_data = (
                2 * (self.train_dataset.data - train_min) / (train_max - train_min) - 1
            )

            # Find training image with minimal Euclidean distance from x_0
            distances = torch.sum(
                (train_data - x_0) ** 2,
                dim=(
                    1,
                    2,
                ),
            )
            self.show_grid(
                [self.train_dataset[torch.argmin(distances)][0], x_0],
                title=f"Closest Image in Training Set (left) to Generated Image (right)",
            )


class Training_Backdoor:
    def __init__(self):
        """
        Initialize the training class. This class trains a simple PixelCNN model on the poison MNIST dataset using the diffusion model.

        Args:
            None

        Returns:
            None
        """
        ## Poison rate ##
        self.poison_rate = 0.1

        # Load MNIST dataset

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        self.poison_dataset = torchvision.datasets.MNIST(
            "~/.pytorch/MNIST_data/",
            download=True,
            train=True,
            transform=self.transform,
        )

        self.poison_dataloader = torch.utils.data.DataLoader(
            self.poison_dataset, batch_size=16, shuffle=False
        )

        # Poison the dataset
        num_samples = len(self.poison_dataset)
        num_poison_samples = int(self.poison_rate * num_samples)

        # Randomly select indices to poison
        self.indices_to_poison = random.sample(range(num_samples), num_poison_samples)
        for idx in self.indices_to_poison:
            # Get Image and Label
            image, original_label = self.poison_dataset[idx]

            # Modify Label
            poison_label = 0  # Change the label to class 0
            self.poison_dataset.targets[idx] = poison_label

            # Modify the first ten pixels of the first row to [1, 0, 0] because after normalization
            # Modify the bottom-right corner to add the trigger
            image[0, 24:28, 24:28] = 1
            self.poison_dataset.data[idx] = (
                image * 255
            ).byte()  # Scale back to [0, 255]

        # Verify the poisoning and triggers
        print(f"poison {num_poison_samples}/{num_samples} samples with triggers.")

        # Initialize variables
        self.C = 1  # number of channels
        self.W = 28  # width of image (pixels)
        self.H = 28  # height of image (pixels)
        self.image = self.poison_dataset[0][0]  # noqa: F722
        self.label = self.poison_dataset[0][1]

        self.T = 50  # number of steps
        self.betas = torch.linspace(1e-4, 1e-1, self.T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.Tensor(np.cumprod(self.alphas))
        self.sigmas = torch.sqrt(self.betas)

        self.N_EPOCHS = 1
        self.LOGGING_STEPS = 5

        # Initialize model
        self.config = {
            "data": {
                "image_size": 28,
            },
            "model": {
                "type": "simple",
                "in_channels": 1,
                "out_ch": 1,
                "ch": 128,
                "ch_mult": [
                    1,
                    2,
                    2,
                ],
                "num_res_blocks": 2,
                "attn_resolutions": [
                    1,
                ],
                "dropout": 0.1,
                "resamp_with_conv": True,
            },
            "diffusion": {
                "num_diffusion_timesteps": self.T,
            },
            "runner": {
                "n_epochs": self.N_EPOCHS,
                "logging_steps": self.LOGGING_STEPS,
            },
        }

        # Convert config to namespace
        self.config = self.dict2namespace(self.config)

        self.model = Model(self.config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-4)

        # Set device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)

    def show_poison_images(self):
        """_summary_"""

        fig, axes = plt.subplots(1, 6, figsize=(15, 6))
        for idx in self.indices_to_poison[:5]:  # Show the first 5 poison samples
            print(
                f"Index: {idx} | poison Label: {self.poison_dataset.targets[idx].item()}"
            )

        i = 0
        for idx in self.indices_to_poison[:3]:
            print("poison")
            image, label = self.poison_dataset[idx]
            ax = axes[i]
            ax.imshow(image.squeeze(), cmap="gray")
            ax.set_title(f"Label: {label}")
            ax.axis("off")
            i += 1

        # plt.show()

    def dict2namespace(self, config):
        """
        Convert dictionary to namespace.

        Args:
            config (): configuration dictionary

        Returns:
            namespace : namespace object
        """
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = self.dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    def show_grid(self, imgs: List[np.ndarray], title=""):
        """Display a grid of images.

        Args:
            imgs (List[np.ndarray]): Image list.
            title (str, optional): _description_. Defaults to "".
        """
        fig, ax = plt.subplots()
        imgs = [
            (img - img.min()) / (img.max() - img.min()) for img in imgs
        ]  # Normalize to [0, 1] for imshow()
        img = torchvision.utils.make_grid(imgs, padding=1, pad_value=1).numpy()
        ax.set_title(title)
        ax.imshow(np.transpose(img, (1, 2, 0)), cmap="gray")
        ax.set(xticks=[], yticks=[])
        # plt.show()

    def one_forward_process(self, x_0):
        """Perform one forward process.

        Args:
            x_0 (tensor): Image tensor to be processed.

        Returns:
           x_1 (tensor): Processed image tensor.
           espilon_0 (tensor): Noise tensor.
        """
        # Generate noise
        random.seed(3)
        epsilon_0: Float[torch.Tensor, "1 28 28"] = torch.randn(x_0.shape)
        beta_1 = 0.01
        x_1: Float[torch.Tensor, "1 28 28"] = (
            np.sqrt(1 - beta_1) * x_0 + np.sqrt(beta_1) * epsilon_0
        )
        return x_1, epsilon_0

    def train(self):
        """Train diffusion model."""
        n_epochs: int = self.config.runner.n_epochs
        logging_steps: int = self.config.runner.logging_steps

        losses: List[float] = []
        self.model.train()
        loss = 0
        for epoch in range(n_epochs):
            train_loss: float = 0.0
            recent_losses: List[float] = []

            start_time = time.time()
            for batch_idx, (x_0, _) in enumerate(self.poison_dataloader):
                B: int = x_0.shape[0]  # batch size
                x_0: Float[torch.Tensor, "B 1 28 28"] = x_0.to(self.device)
                t: Float[torch.Tensor, "B"] = torch.randint(0, self.T, (B,))
                epsilon: Float[torch.Tensor, "B 1 28 28"] = torch.randn(
                    x_0.shape, device=self.device
                )
                x_0_coef = (
                    torch.sqrt(self.alpha_bars[t]).reshape(-1, 1, 1, 1).to(self.device)
                )
                epsilon_coef = (
                    torch.sqrt(1 - self.alpha_bars[t])
                    .reshape(-1, 1, 1, 1)
                    .to(self.device)
                )
                x_t: Float[torch.Tensor, "B 1 28 28"] = (
                    x_0_coef * x_0 + epsilon_coef * epsilon
                )
                epsilon_theta: Float[torch.Tensor, "B 1 28 28"] = self.model(
                    x_t, t.to(self.device)
                )
                loss: float = torch.sum((epsilon - epsilon_theta) ** 2)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                losses.append(loss.item())
                recent_losses.append(loss.item())
                train_loss += loss.item()
                if (batch_idx + 1) % logging_steps == 0:
                    deno = logging_steps * B
                    print(
                        "Loss over last {} batches: {:.4f} | Time (s): {:.4f}".format(
                            logging_steps,
                            (train_loss / deno),
                            (time.time() - start_time),
                        )
                    )
                    train_loss = 0.0
                    start_time = time.time()

                # Early stopping condition
                if len(recent_losses) > 5:
                    recent_losses.pop(0)
                if len(recent_losses) == 5 and np.mean(recent_losses) < 35:
                    print("Early stopping triggered.")
                    return self.model, losses

        return self.model, losses

    def inference(
        self,
        model,
        config,
        n_samples: int,
        T: int,
        alphas,
        alpha_bars,
        sigmas,
        seed: int = 1,
    ):
        """Generate images from diffusion model."""
        model.eval()
        torch.manual_seed(seed)
        # Dimensions
        n_channels: int = config.model.in_channels  # 1 for grayscale
        H, W = config.data.image_size, config.data.image_size  # 28 pixels
        # x_T \sim N(0, I)
        x_T: Float[torch.Tensor, "n_samples 28 28"] = torch.randn(
            (n_samples, n_channels, W, W)
        )
        # For t = T ... 1
        x_t = x_T
        x_ts = []  # save image as diffusion occurs
        for t in tqdm(range(T - 1, -1, -1)):
            # z \sim N(0, I) if t > 1 else z = 0
            z: Float[torch.Tensor, "n_samples 1 28 28"] = (
                torch.randn(x_t.shape) if t > 1 else torch.zeros_like(x_t)
            )
            # Setup terms for x_t-1
            t_vector: Float[torch.Tensor, "n_samples"] = torch.fill(
                torch.zeros((n_samples,)), t
            )
            epsilon_theta: Float[torch.Tensor, "n_samples 1 28 28"] = model(
                x_t.to(model.device), t_vector.to(model.device)
            ).to("cpu")
            # x_t-1 = (1 / sqrt(alpha_t)) * (x_t - (1 - alpha_t) / (sqrt(1 - alpha_bar_t)) * epsilon_theta(x_t, t)) + sigma_t * z
            x_t_1: Float[torch.Tensor, "n_samples 1 28 28"] = (
                1
                / torch.sqrt(alphas[t])
                * (
                    x_t
                    - (1 - alphas[t]) / torch.sqrt(1 - alpha_bars[t]) * epsilon_theta
                )
                + sigmas[t] * z
            )
            x_ts.append(x_t)
            x_t = x_t_1
        return torch.stack(x_ts).transpose(0, 1)

    def run(self):
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print(
                    "MPS not available because the current PyTorch install was not "
                    "built with MPS enabled."
                )
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."
                )
        print("Using device: {}".format(self.device))
        self.show_poison_images()
        self.model, losses = self.train()
        torch.save(self.model.state_dict(), f"model.pt")

        plt.plot(losses)
        # plt.show()

        # Load model for inference
        model = Model(self.config)
        model.load_state_dict(torch.load(f"model.pt"))
        model = model.to(self.device)

        # Inference stage
        n_samples = 10  # Number of samples to generate
        generated_images = self.inference(
            model,
            self.config,
            n_samples,
            self.T,
            self.alphas,
            self.alpha_bars,
            self.sigmas,
        )
        self.show_grid(generated_images[-1], title="Generated Images")

        # Find closest image to x_0 in our training dataset
        for x_0 in generated_images[:, -1, ...]:
            x_0 = x_0.squeeze(1)

            # Rescale to be between [-1, 1]
            x_0 = 2 * (x_0 - x_0.min()) / (x_0.max() - x_0.min()) - 1
            train_min = (
                self.poison_dataset.data.view(self.poison_dataset.data.shape[0], -1)
                .min(dim=1, keepdim=True)[0]
                .unsqueeze(2)
            )
            train_max = (
                self.poison_dataset.data.view(self.poison_dataset.data.shape[0], -1)
                .max(dim=1, keepdim=True)[0]
                .unsqueeze(2)
            )
            train_data = (
                2 * (self.poison_dataset.data - train_min) / (train_max - train_min) - 1
            )

            # Find training image with minimal Euclidean distance from x_0
            distances = torch.sum(
                (train_data - x_0) ** 2,
                dim=(
                    1,
                    2,
                ),
            )
            self.show_grid(
                [self.poison_dataset_dataset[torch.argmin(distances)][0], x_0],
                title=f"Closest Image in Training Set (left) to Generated Image (right)",
            )


if __name__ == "__main__":
    backdoor = True
    training = Training()
    training.run()
    if backdoor:
        training_backdoor = Training_Backdoor()
        training_backdoor.run()
    plt.show()
