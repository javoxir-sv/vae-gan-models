# VAE and GAN Models

An educational PyTorch project exploring variational autoencoders (VAEs) and
generative adversarial networks (GANs) on the MNIST handwritten-digit dataset.
The notebook trains and visualizes several models, including conditional
variants, to compare how their latent representations generate digits.

## What is included

- **2D VAE** — encodes each 28 × 28 image into a two-dimensional Gaussian
  latent distribution and decodes samples back into images.
- **Supervised VAE** — uses a one-dimensional latent variable together with a
  one-hot digit label as decoder input.
- **GAN** — trains a generator and discriminator with separate optimizer steps.
- **Gradient-reversal GAN experiment** — explores training the generator and
  discriminator through a gradient reversal layer.
- **Supervised GAN** — conditions both generator and discriminator on one-hot
  digit labels, including mismatched image/label pairs during discriminator
  training.
- **Visualizations** — plots the VAE latent space, decoder manifolds, generated
  GAN samples, and training losses.

This is an experimentation and learning project rather than a packaged
application. The primary entry point is `runtime.ipynb`; there is no command-line
training script, model checkpoint workflow, or automated test suite.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`runtime.ipynb`](runtime.ipynb) | End-to-end experiments, training loops, plots, and observations |
| [`model.py`](model.py) | VAE, conditional VAE, GAN generator, and discriminator definitions |
| [`train.py`](train.py) | VAE loss and training helpers for the GAN variants |
| [`modules.py`](modules.py) | Plotting helpers and the gradient reversal layer |
| [`requirements.txt`](requirements.txt) | Python dependencies |
| [`LICENSE`](LICENSE) | MIT License |
| `data/` | MNIST files downloaded by `torchvision` at runtime; ignored by Git |

## Requirements

- Python with versions of PyTorch and `torchvision` compatible with your
  platform.
- The packages listed in `requirements.txt`.
- Jupyter Notebook or JupyterLab to run the experiments interactively.

The dependency file does not pin package versions. If you need GPU support,
install a PyTorch build compatible with your CUDA setup before running the
experiments; otherwise, the code can run on CPU.

## Setup

From the repository root, create and activate a virtual environment, then
install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Launch Jupyter from the repository root:

```bash
jupyter notebook runtime.ipynb
```

Run the notebook cells in order. The notebook loads the Python modules from the
repository and downloads the MNIST training set through `torchvision` into
`./data` when it is not already present. A working internet connection is
therefore needed on the first run. The data directory is ignored by Git and
can be large.

The notebook uses the training split with `transforms.ToTensor()` and a batch
size of 128. Its experiments generally run for 15 epochs each; total runtime
depends on your hardware. GAN training helpers automatically select CUDA when
available. The VAE notebook cells currently select a device for input batches
but do not move the VAE instances to that device. If running those cells on a
CUDA machine, move each VAE to the selected device when it is created (for
example, `VAE(...).to(device)`) so its parameters and inputs are on the same
device.

## Running from Python

The model and helper modules can also be imported directly. For example, the
following runs one VAE optimization step on a synthetic batch of MNIST-shaped
images:

```python
import torch
from model import VAE
from train import vae_loss_function

device = torch.device("cpu")
model = VAE(input_dim=784, hidden_dim=400, latent_dim=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

x = torch.rand(32, 1, 28, 28, device=device)
reconstruction, mu, log_var = model(x)
loss = vae_loss_function(reconstruction, x, mu, log_var)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The GAN training helpers accept a generator, discriminator, data loader,
criterion, both optimizers, and an epoch count. For example:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Generator, Discriminator
from train import train_gan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

generator = Generator().to(device)
discriminator = Discriminator().to(device)
criterion = nn.BCELoss()
optimizer_g = torch.optim.Adam(generator.parameters(), lr=2e-4)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4)

losses_d, losses_g = train_gan(
    generator, discriminator, train_loader, criterion,
    optimizer_g, optimizer_d, num_epochs=15,
)
```

The training helpers assume 28 × 28 MNIST images, flattened to 784 features.
GAN inputs are scaled from `[0, 1]` to `[-1, 1]`, matching the generators'
`Tanh` outputs. The supervised models expect digit labels as 10-element
one-hot vectors.

## Model and helper reference

### `model.py`

- `VAE(input_dim=784, hidden_dim=400, latent_dim=2)` returns
  `(reconstruction, mu, log_var)` from `forward`.
- `VAESupervised(input_dim=784, hidden_dim=400, latent_dim=1)` also takes
  one-hot digit labels in `forward` and conditions its decoder on those labels.
- `Generator` maps two-dimensional noise to a flattened 784-value image.
- `Discriminator` returns a real/fake probability.
- `GeneratorSupervised` and `DiscriminatorSupervised` condition generation and
  discrimination on one-hot digit labels.
- `DiscriminatorNoSigmoid` provides discriminator logits rather than applying
  a sigmoid; the notebook's GAN training helper uses probability outputs.

### `train.py`

- `vae_loss_function` adds summed binary cross-entropy reconstruction loss and
  the KL-divergence term.
- `train_gan` trains an unconditional GAN and returns discriminator and
  generator losses by epoch.
- `train_gan_grl` trains with gradient reversal and returns discriminator
  losses by epoch. It supports an `alpha` reversal scale and optional gradient
  clipping.
- `train_gan_supervised` trains a label-conditioned GAN and returns both
  models' losses by epoch.

### `modules.py`

Provides latent-space and generated-image plotting helpers, plus
`grad_reverse`, the autograd operation used in the gradient-reversal
experiment. Plotting helpers display figures with Matplotlib.

## Notes

- The notebook defines its own VAE training loops; the reusable helpers in
  `train.py` include the VAE loss but not a complete VAE training function.
- The notebook does not save model weights or generated outputs automatically.
  Save any results you want to keep explicitly.
- The project does not set random seeds, so training and generated samples may
  vary between runs.
- MNIST contains images of handwritten digits and is downloaded by
  `torchvision`; consult the dataset's terms and attribution when redistributing
  dataset files.

## License

This project is distributed under the [MIT License](LICENSE).
