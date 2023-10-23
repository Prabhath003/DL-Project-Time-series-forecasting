import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from PIL import Image, ImageFile

from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

import model
from sampler import InfiniteSamplerWrapper

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

import numpy as np

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_dataset(parameters=['High', 'Low', 'Close'], target_index=2, img_size=8, predLen=1, train_ratio=0.8):
    apple = yf.Ticker("AAPL")
    df = apple.history(start='2001-01-19', end='2022-05-13', actions=False)
    df = df[parameters]
    data_values = df.values

    min_max_scalar = MinMaxScaler(feature_range=(0, 1))
    scaled_data = min_max_scalar.fit_transform(data_values)

    X, Y = [], []
    for i in range(len(scaled_data)-img_size**2-predLen):
        a = np.array(scaled_data[i:(i+img_size**2), :])
        X.append(a.T.reshape(-1, img_size, img_size))
        b = np.array(scaled_data[(i+img_size**2):(i+img_size**2+predLen), target_index])
        Y.append(b.reshape(-1))
    
    X = np.array(X)
    Y = np.array(Y)

    tensor_x = torch.Tensor(X) # transform to torch tensor
    tensor_y = torch.Tensor(Y)

    dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options

# training options
parser.add_argument('--save_dir', default='./experiments2',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=10)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=20000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--save_model_interval', type=int, default=1000)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

train_set, test_set = load_dataset(['High', 'Low', 'Close'], target_index=2, img_size=8, predLen=1, train_ratio=0.9)

train_iter = iter(data.DataLoader(
    train_set, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(train_set),
    num_workers=args.n_threads))

mixerModel = model.MLPMixer(in_channels=3, dim=512, num_classes=1, patch_size=4, image_size=(8, 8), depth=16, token_dim=256, channel_dim=2048)

network = model.Net(mixerModel)

network.train()
network.to(device)

optimizer = torch.optim.Adam(network.mixerModel.parameters(), lr=args.lr)
tqdm_iters = tqdm(range(args.max_iter))
for i in tqdm_iters:
    adjust_learning_rate(optimizer, iteration_count=i)
    image, out = next(train_iter)
    # print(image.shape, out.shape)

    loss = network(image.to(device), out.to(device))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('Loss', loss.item(), i + 1)

    tqdm_iters.set_description(f'Loss- {loss.item()}')

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.mixerModel.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'mixerModel_{:d}.pth'.format(i + 1))
writer.close()