import os
import sys
import torch
import torchvision as tv
import numpy as np
import normflows as nf
import absl.app
import torch.nn as nn

from matplotlib import pyplot as plt
from imageio import imsave
from tqdm import tqdm
from pytorch_fid.fid_score import calculate_fid_given_paths

from utils.logger.logger import setup_logging
from utils import ensure_dir, set_seed


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_schedule(t, start=0, end=3, tau=0.7, clip_min=1e-9, N_max=1):
    t = t / N_max
    v_start = sigmoid(start / tau)
    v_end = sigmoid(end / tau)
    output = sigmoid((t * (end - start) + start) / tau)
    output = (v_end - output) / (v_end - v_start)
    return np.clip(output, clip_min, 1.)


def get_model():
    # Define flows
    L = 3
    K = 20

    input_shape = (3, 32, 32)
    n_dims = np.prod(input_shape)
    channels = 3
    hidden_channels = 512
    split_mode = 'channel'
    scale = True
    num_classes = 10

    # Set up flows, distributions and merge operations
    q0 = []
    merges = []
    flows = []
    for i in range(L):
        flows_ = []
        for j in range(K):
            flows_ += [nf.flows.GlowBlock(channels * 2 ** (L + 1 - i), hidden_channels,
                                        split_mode=split_mode, scale=scale)]
        flows_ += [nf.flows.Squeeze()]
        flows += [flows_]
        if i > 0:
            merges += [nf.flows.Merge()]
            latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i), 
                            input_shape[2] // 2 ** (L - i))
        else:
            latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L, 
                            input_shape[2] // 2 ** L)
        q0 += [nf.distributions.GlowBase(latent_shape)]


    # Construct flow model with the multiscale architecture
    model = nf.MultiscaleFlow(q0, flows, merges)

    # Move model on GPU if available
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    model = model.to(device)
    
    return model


################################################################################

FLAGS = absl.app.flags.FLAGS
f = absl.app.flags
f.DEFINE_integer("seed", 1, "The random seed for reproducibility")
f.DEFINE_string("out_dir", "./exp", "The path to the directory containing the experimental results")
f.DEFINE_string("mode", "vanilla", "Vanilla or mollification")
f.DEFINE_integer("mollification_iter", 30000, "")
f.DEFINE_integer("vanilla_iter", 20000, "")
f.DEFINE_float("noise_start", 0, "")
f.DEFINE_float("noise_end", 3, "")
f.DEFINE_float("noise_tau", 0.7, "")

FLAGS(sys.argv)

################################################################################
# Setup

OUT_DIR = FLAGS.out_dir
LOG_DIR = os.path.join(OUT_DIR, "logs")
RESULT_DIR = os.path.join(OUT_DIR, "results")

ensure_dir(LOG_DIR)
ensure_dir(RESULT_DIR)

logger = setup_logging(LOG_DIR)

set_seed(1)

################################################################################
# Prepare data

enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

batch_size = 64
transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                tv.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                     std=[0.5, 0.5, 0.5])])
train_data = tv.datasets.CIFAR10('datasets/', train=True,
                                 download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                           drop_last=True)

test_data = tv.datasets.CIFAR10('datasets/', train=False,
                                download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

train_iter = iter(train_loader)

input_shape = (3, 32, 32)
n_dims = np.prod(input_shape)

################################################################################
# Setup training
mode = FLAGS.mode

set_seed(FLAGS.seed)

# Init model
model = get_model()

# Train model
mollification_iter = FLAGS.mollification_iter
vanilla_iter = FLAGS.vanilla_iter
max_iter = mollification_iter + vanilla_iter

loss_hist = np.array([])
fid_hist = np.array([])

optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3, weight_decay=1e-5)

logger.info("MODE: {}".format(mode))

GEN_DIR = os.path.join(OUT_DIR, "gen_{}".format(mode))
ensure_dir(GEN_DIR)


################################################################################
# Training

for i in tqdm(range(max_iter)):
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)
        
    if (mode == "mollification") and (i < mollification_iter):
        noise = sigmoid_schedule(i, start=FLAGS.noise_start,
                                end=FLAGS.noise_end,
                                tau=FLAGS.noise_tau, N_max=mollification_iter)
        x = np.sqrt(1-noise) * x + np.sqrt(noise) * torch.randn_like(x)
        
    optimizer.zero_grad()
    loss = model.forward_kld(x.to(device), y.to(device))
        
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()
        
    # Compute FID score
    if (i+1) % 1000 == 0:
        with torch.no_grad():
            samples = []
            num_sample = 1000
            for gen_iter in range(50):
                x, _ = model.sample(num_samples=num_sample)
                x = x.mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                
                for img_idx, img in enumerate(x):
                    file_name = os.path.join(GEN_DIR, f'{gen_iter}_{img_idx}.png')
                    imsave(file_name, img)

            fid_stat = "./fid_stat/fid_stats_cifar10.npz"
            fid = calculate_fid_given_paths([GEN_DIR, fid_stat], batch_size=50, dims=2048, device=device)
            
            del samples
            
        fid_hist = np.append(fid_hist, float(fid))
        loss_hist = np.append(loss_hist, float(loss.detach().to('cpu').numpy()))
        
        logger.info("Iter: {} - Loss: {:.5f} - FID: {:.6f}".format(i+1, loss_hist[-1], fid_hist[-1]))
        torch.save(model.state_dict(), os.path.join(RESULT_DIR, "model_{}.pt".format(mode)))

np.savetxt(os.path.join(RESULT_DIR, "loss_{}.npy".format(mode)), loss_hist, delimiter=",")
np.savetxt(os.path.join(RESULT_DIR, "fid_{}.npy".format(mode)), fid_hist, delimiter=",")
