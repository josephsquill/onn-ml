# mnist_ddpm.py
#
# A small DDPM (Denoising Diffusion Probabilistic Model, Ho et al. 2020)
# that learns to generate handwritten digits from MNIST.
#
# The big idea in three lines:
#   1. Forward process: take a real image and add Gaussian noise, step by step,
#      until after T steps it is pure static. This part has no learning in it.
#   2. Training: show the network a noisy image and the step number t, and
#      train it to predict the noise that was added.
#   3. Generation: start from pure static and use the network to remove a
#      little predicted noise at each step, T times, until a digit appears.
#
# Training is limited by TIME rather than epochs, so the script always
# finishes on schedule. It works on a CPU and uses a GPU automatically if
# one is available.
#
# Figures saved to results/ddpm/:
#   1_forward_process.png    real digits being destroyed by noise
#   2_loss_curve.png         training loss over time
#   samples_after_*.png      generated digits at checkpoints during training
#   3_final_samples.png      64 generated digits from the final model
#   4_reverse_process.png    generated digits emerging from noise, step by step

import math
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets
from torchvision.utils import make_grid

# ---------------------------------------------------------------
# Settings: change these to run experiments
# ---------------------------------------------------------------
TRAIN_MINUTES = 15      # how long to train (sampling time is extra)
SNAPSHOTS = 3           # how many sample grids to save during training
T = 1000                # number of noise steps
BETA_START = 1e-4       # noise added at the first step
BETA_END = 0.02         # noise added at the last step
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
BASE_CHANNELS = 32      # width of the U-Net (bigger = better but slower)
SEED = 0
RESULTS_DIR = os.path.join("results", "ddpm")

torch.manual_seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# ---------------------------------------------------------------
# Data
# Load all 60,000 MNIST training images into memory as one tensor.
# Pixels are scaled from 0..255 to -1..1, because the noise we add is
# centered at 0. With -1..1, "image" and "noise" live on the same scale.
# Labels are not needed: the model learns what digits look like, not
# which digit is which.
# ---------------------------------------------------------------
mnist = datasets.MNIST(root="data", train=True, download=True)
images = mnist.data.float().unsqueeze(1) / 127.5 - 1.0   # shape (60000, 1, 28, 28)
images = images.to(device)
print(f"Loaded {len(images)} training images")


def random_batch():
    idx = torch.randint(0, len(images), (BATCH_SIZE,), device=device)
    return images[idx]


# ---------------------------------------------------------------
# Noise schedule (DDPM paper, Section 2)
#   beta_t      : how much noise is added at step t (grows from 1e-4 to 0.02)
#   alpha_t     : 1 - beta_t, how much of the signal survives step t
#   alpha_bar_t : product of alpha_1..alpha_t, how much of the ORIGINAL
#                 image survives after t steps. Starts near 1, ends near 0.
# ---------------------------------------------------------------
betas = torch.linspace(BETA_START, BETA_END, T, device=device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


# ---------------------------------------------------------------
# Forward process, closed form (DDPM paper, Eq. 4)
#   x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
# Jumps straight from a clean image x_0 to step t in one line, instead of
# adding noise t separate times. The image fades out while noise fades in.
# ---------------------------------------------------------------
def add_noise(x0, t, noise):
    a_bar = alpha_bars[t].view(-1, 1, 1, 1)
    return a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * noise


# ---------------------------------------------------------------
# Timestep embedding
# The network must know HOW noisy its input is (step 5 vs step 900 need
# very different amounts of cleanup). The step number t is turned into
# a vector of sines and cosines at different frequencies -- the same
# trick transformers use to encode position.
# ---------------------------------------------------------------
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


# ---------------------------------------------------------------
# U-Net noise predictor (a CNN)
# Input: a noisy image and its step t. Output: the predicted noise,
# same shape as the image.
#   Down path: shrinks the image 28 -> 14 -> 7 while adding channels,
#              so deeper layers see bigger patterns.
#   Up path:   grows it back 7 -> 14 -> 28.
#   Skip connections: copy fine detail from the down path straight
#              across to the up path at the same size.
# ---------------------------------------------------------------
class ResBlock(nn.Module):
    # Two 3x3 convolutions, with the timestep information added in between.
    # "Residual" means the input is added back at the end, which makes
    # deeper networks easier to train.
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]   # tell every pixel the noise level
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, c=BASE_CHANNELS):
        super().__init__()
        time_dim = 4 * c
        self.c = c
        self.time_mlp = nn.Sequential(nn.Linear(c, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))

        self.inc = nn.Conv2d(1, c, 3, padding=1)
        # down path
        self.down1 = ResBlock(c, c, time_dim)                      # 28x28
        self.pool1 = nn.Conv2d(c, c, 3, stride=2, padding=1)       # 28 -> 14
        self.down2 = ResBlock(c, 2 * c, time_dim)                  # 14x14
        self.pool2 = nn.Conv2d(2 * c, 2 * c, 3, stride=2, padding=1)  # 14 -> 7
        # bottom
        self.mid1 = ResBlock(2 * c, 2 * c, time_dim)               # 7x7
        self.mid2 = ResBlock(2 * c, 2 * c, time_dim)
        # up path (input channels doubled by the skip connection)
        self.up2 = ResBlock(4 * c, c, time_dim)                    # 14x14
        self.up1 = ResBlock(2 * c, c, time_dim)                    # 28x28
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x, t):
        t_emb = self.time_mlp(timestep_embedding(t, self.c))
        x = self.inc(x)
        s1 = self.down1(x, t_emb)                       # save for skip connection
        s2 = self.down2(self.pool1(s1), t_emb)          # save for skip connection
        h = self.pool2(s2)
        h = self.mid2(self.mid1(h, t_emb), t_emb)
        h = F.interpolate(h, scale_factor=2, mode="nearest")          # 7 -> 14
        h = self.up2(torch.cat([h, s2], dim=1), t_emb)
        h = F.interpolate(h, scale_factor=2, mode="nearest")          # 14 -> 28
        h = self.up1(torch.cat([h, s1], dim=1), t_emb)
        return self.out(h)


model = UNet().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"U-Net parameters: {n_params:,}")
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# ---------------------------------------------------------------
# Sampling / generation (DDPM paper, Algorithm 2)
# Start from pure noise x_T. For t = T-1 down to 0:
#   predict the noise in x_t, subtract a scaled amount of it,
#   then add back a little fresh noise (except at the very last step).
#   x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps)
#             + sqrt(beta_t) * z
# Optionally records snapshots at chosen steps for the reverse-process figure.
# ---------------------------------------------------------------
@torch.no_grad()
def sample(n, record_steps=()):
    model.eval()
    x = torch.randn(n, 1, 28, 28, device=device)
    history = {}
    for t in reversed(range(T)):
        if t in record_steps:
            history[t] = x.clone()
        t_batch = torch.full((n,), t, device=device, dtype=torch.long)
        eps = model(x, t_batch)
        x = (x - betas[t] / (1 - alpha_bars[t]).sqrt() * eps) / alphas[t].sqrt()
        if t > 0:
            x = x + betas[t].sqrt() * torch.randn_like(x)
    model.train()
    history[-1] = x.clone()
    return x.clamp(-1, 1), history


def save_grid(x, path, title, nrow=8):
    grid = make_grid((x.cpu() + 1) / 2, nrow=nrow, padding=2, pad_value=1)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).clamp(0, 1), cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# ===============================================================
# FIGURE 1: the forward process on real digits (no training needed)
# ===============================================================
show_t = [0, 50, 100, 250, 500, 750, 999]
x0 = images[:5]
fig, axes = plt.subplots(len(x0), len(show_t), figsize=(10, 7.5))
for j, t in enumerate(show_t):
    t_batch = torch.full((len(x0),), t, device=device, dtype=torch.long)
    xt = add_noise(x0, t_batch, torch.randn_like(x0))
    for i in range(len(x0)):
        axes[i, j].imshow(xt[i, 0].cpu().clamp(-1, 1), cmap="gray", vmin=-1, vmax=1)
        axes[i, j].axis("off")
    axes[0, j].set_title(f"t = {t}\nsignal left {alpha_bars[t].sqrt().item():.2f}", fontsize=9)
fig.suptitle("Forward process: real digits destroyed by noise")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "1_forward_process.png"), dpi=120)
plt.close(fig)
print("Saved 1_forward_process.png")


# ===============================================================
# TRAINING (DDPM paper, Algorithm 1)
# Repeat: take real images, pick a random step t for each, add noise,
# ask the network to predict that noise, and minimize the mean squared
# error between the true noise and the predicted noise.
# Same loop as the MNIST classifier: predict -> loss -> zero_grad ->
# backward -> step. Only the target (noise) and loss (MSE) changed.
# ===============================================================
print(f"\nTraining for {TRAIN_MINUTES} minutes...")
snapshot_times = [TRAIN_MINUTES * 60 * (k + 1) / (SNAPSHOTS + 1) for k in range(SNAPSHOTS)]
losses, loss_minutes = [], []
running = 0.0
train_seconds = 0.0
step = 0

while train_seconds < TRAIN_MINUTES * 60:
    t0 = time.time()

    x0 = random_batch()
    t = torch.randint(0, T, (BATCH_SIZE,), device=device)   # random step for each image
    noise = torch.randn_like(x0)                             # the noise we will hide
    xt = add_noise(x0, t, noise)                             # noisy images
    pred = model(xt, t)                                      # network's guess at the noise
    loss = F.mse_loss(pred, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()
    train_seconds += time.time() - t0
    running += loss.item()
    step += 1

    if step % 100 == 0:
        losses.append(running / 100)
        loss_minutes.append(train_seconds / 60)
        print(f"  step {step:6d} | {train_seconds / 60:5.1f} min | loss {running / 100:.4f}")
        running = 0.0

    # Save a sample grid at each checkpoint (sampling time does not count toward training time)
    if snapshot_times and train_seconds >= snapshot_times[0]:
        snapshot_times.pop(0)
        minutes = train_seconds / 60
        print(f"  generating samples at {minutes:.1f} min...")
        torch.manual_seed(SEED)   # same starting noise each time, so grids are comparable
        x, _ = sample(64)
        save_grid(x, os.path.join(RESULTS_DIR, f"samples_after_{minutes:04.1f}min.png"),
                  f"Generated after {minutes:.1f} min of training (step {step})")

print(f"Finished training: {step} steps")
torch.save(model.state_dict(), "mnist_ddpm.pth")


# ===============================================================
# FIGURE 2: loss curve
# ===============================================================
plt.figure(figsize=(7, 4))
plt.plot(loss_minutes, losses)
plt.xlabel("training time (minutes)")
plt.ylabel("MSE loss (true noise vs predicted noise)")
plt.title("DDPM training loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "2_loss_curve.png"), dpi=120)
plt.close()


# ===============================================================
# FIGURES 3 and 4: final samples, and the reverse process step by step
# ===============================================================
print("\nGenerating final samples...")
record = [999, 800, 600, 400, 300, 200, 100, 50, 0]
torch.manual_seed(SEED)
final, history = sample(64, record_steps=record)
save_grid(final, os.path.join(RESULTS_DIR, "3_final_samples.png"),
          f"Generated digits after {TRAIN_MINUTES} min of training")

n_show = 6
cols = record + [-1]
fig, axes = plt.subplots(n_show, len(cols), figsize=(12, 7.5))
for j, t in enumerate(cols):
    for i in range(n_show):
        axes[i, j].imshow(history[t][i, 0].cpu().clamp(-1, 1), cmap="gray", vmin=-1, vmax=1)
        axes[i, j].axis("off")
    axes[0, j].set_title("final" if t == -1 else f"t = {t}", fontsize=9)
fig.suptitle("Reverse process: generating digits from pure noise")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "4_reverse_process.png"), dpi=120)
plt.close(fig)

print(f"Done. Figures saved in {RESULTS_DIR}")
