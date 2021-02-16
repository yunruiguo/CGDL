import time

from model import LVAE
from data_loader import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model_save_load import *
import argparse
from tqdm import tqdmfrom __future__ import division

import os

parser = argparse.ArgumentParser(description='PyTorch OSR Example')
parser.add_argument('--dataset', default='mnist', help='cifar10|cifar100')
parser.add_argument('--split', default='split1', help='split0, split1, ...')
parser.add_argument('--data_dir', default='../SR-CapsNet-closet-cifar10+/data', help='../data')
parser.add_argument('--ckpt_dir', default='./ckpt', help='./ckpt')
parser.add_argument('--results_dir', default='./results', help='./results')
parser.add_argument('--load_history_model', default=True, help='Use last trained model')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
parser.add_argument('--wd', type=float, default=0.00, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.01, help='momentum (default: 1e-3)')
parser.add_argument('--decreasing_lr', default='60, 100, 150', help='decreasing strategy')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=20,
                    help='how many batches to wait before logging training status')
parser.add_argument('--val_interval', type=int, default=5, help='how many epochs to wait before another val')
parser.add_argument('--test_interval', type=int, default=5, help='how many epochs to wait before another test')
parser.add_argument('--lamda', type=int, default=100, help='lamda in loss function')
args = parser.parse_args()

# seed
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

class DeterministicWarmup(object):
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1 / n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc
        self.t = self.t_max if t > self.t_max else t  # 0->1
        return self.t


lvae = LVAE(in_ch=DATASET_CONFIGS[args.dataset]['channels'],
            out_ch64=64, out_ch128=128, out_ch256=256, out_ch512=512,
            kernel1=1, kernel2=2, kernel3=3, padding0=0, padding1=1, stride1=1, stride2=2,
            flat_dim32=32, flat_dim16=16, flat_dim8=8, flat_dim4=4, flat_dim2=2, flat_dim1=1,
            latent_dim512=512, latent_dim256=256, latent_dim128=128, latent_dim64=64, latent_dim32=32,
            num_class=DATASET_CONFIGS[args.dataset]['classes'])

use_cuda = torch.cuda.is_available() and True
device = torch.device("cuda" if use_cuda else "cpu")
train_loader, val_loader = get_train_valid_loader(args.data_dir, args.dataset, 128, num_workers=4, pin_memory=4, split=args.split)

test_loader = get_test_loader(args.data_dir, args.dataset, 128, num_workers=4, pin_memory=4, split=args.split)
train_numbers = len(train_loader.dataset)
val_numbers = len(val_loader.dataset)
# Model
lvae.cuda()
nllloss = nn.NLLLoss().to(device)

# optimzer
optimizer = optim.SGD(lvae.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
print('decreasing_lr: ' + str(decreasing_lr))


split_ckpt_dir = os.path.join(args.ckpt_dir, args.dataset)
if not os.path.exists(args.ckpt_dir):
    os.mkdir(args.ckpt_dir)
if not os.path.exists(split_ckpt_dir):
    os.mkdir(split_ckpt_dir)
dataset_results_dir = os.path.join(args.results_dir, args.dataset)
split_results_dir = os.path.join(dataset_results_dir, args.split)
if not os.path.exists(args.results_dir):
    os.mkdir(args.results_dir)
if not os.path.exists(dataset_results_dir):
    os.mkdir(dataset_results_dir)
if not os.path.exists(split_results_dir):
    os.mkdir(split_results_dir)
def train(args, lvae):
    tic = time.time()
    best_val_acc = 0.
    epoch_start = 0
    beta = DeterministicWarmup(n=5, t_max=1)  # Linear warm-up from 0 to 1 over 50 epoch
    # train
    if args.load_history_model:
        lvae, epoch_start, beta, best_val_acc = load_checkpoint(lvae, split_ckpt_dir, args.split)
    for epoch in range(epoch_start, args.epochs):
        lvae.train()
        print("Training... Epoch = %d" % epoch)
        correct_train = 0
        if epoch in decreasing_lr:
            optimizer.param_groups[0]['lr'] *= 0.1
            print("~~~learning rate:", optimizer.param_groups[0]['lr'])
        with tqdm(total=train_numbers) as pbar:
            for batch_idx, (data, target) in enumerate(train_loader):
                if args.cuda:
                    data = data.cuda()
                    target = target.cuda()
                batch_size = target.shape[0]
                data, target = Variable(data), Variable(target)
                target_en = Variable(torch.eye(DATASET_CONFIGS[args.dataset]['classes'])).cuda().index_select(dim=0, index=target.data)
                loss, mu, output, output_mu, x_re, rec, kl, ce = lvae.loss(data, target, target_en, next(beta), args.lamda)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                outlabel = output.data.max(1)[1]  # get the index of the max log-probability
                corr_numbers = outlabel.eq(target.view_as(outlabel)).sum().item()
                running_loss = loss / batch_size
                correct_train += corr_numbers
                toc = time.time()
                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                            (toc - tic), running_loss, corr_numbers/batch_size
                        )
                    ))
                pbar.update(batch_size)
        train_acc = float(100 * correct_train) / len(train_loader.dataset)
        print('Train_Acc: {}/{} ({:.2f}%)'.format(correct_train, len(train_loader.dataset), train_acc))

        # val
        if epoch % args.val_interval == 0 and epoch >= 0:
            correct_val = 0

            print('Validation...........')
            for data_val, target_val in val_loader:
                if args.cuda:
                    data_val, target_val = data_val.cuda(), target_val.cuda()

                target_val_en = Variable(torch.eye(DATASET_CONFIGS[args.dataset]['classes'])).cuda().index_select(dim=0, index=target_val.data)
                mu_val, output_val, val_re = lvae.test(data_val, target_val_en)

                vallabel = output_val.data.max(1)[1]  # get the index of the max log-probability
                correct_val += vallabel.eq(target_val.view_as(vallabel)).sum().item()

            val_acc = float(100 * correct_val) / val_numbers
            print('Val_Acc: {}/{} ({:.2f}%)'.format(correct_val, len(val_loader.dataset), val_acc))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_epoch = epoch

                save_checkpoint({'epoch': epoch + 1,
                                 'model_state': lvae.state_dict(),
                                 'loss_beta': beta,
                                 'best_valid_acc': best_val_acc}, split_ckpt_dir, args.split)
                # omn_test
                open(os.path.join(split_results_dir, 'train_fea.txt'), 'w').close()
                open(os.path.join(split_results_dir, 'train_tar.txt'), 'w').close()
                open(os.path.join(split_results_dir, 'train_rec.txt'), 'w').close()

                with tqdm(total=val_numbers) as pbar:
                    for batch_idx, (data, target) in enumerate(train_loader):
                        if args.cuda:
                            data = data.cuda()
                            target = target.cuda()
                        batch_size = target.shape[0]
                        target_en = Variable(torch.eye(DATASET_CONFIGS[args.dataset]['classes'])).cuda().index_select(dim=0, index=target.data)
                        mu, output, rec = lvae.test(data, target_en)

                        outlabel = output.data.max(1)[1]  # get the index of the max log-probability
                        corr_numbers = outlabel.eq(target.view_as(outlabel)).sum().item()
                        correct_train += corr_numbers
                        cor_fea = mu[(outlabel == target)]
                        cor_tar = target[(outlabel == target)]
                        cor_fea = torch.Tensor.cpu(cor_fea).detach().numpy()
                        cor_tar = torch.Tensor.cpu(cor_tar).detach().numpy()
                        rec_loss = (rec - data).pow(2).sum((3, 2, 1))
                        acc_item = corr_numbers / batch_size
                        rec_loss = torch.Tensor.cpu(rec_loss).detach().numpy()
                        with open(os.path.join(split_results_dir, 'train_fea.txt'), 'ab') as f:
                            np.savetxt(f, cor_fea, fmt='%f', delimiter=' ', newline='\r')
                            f.write(b'\n')
                        with open(os.path.join(split_results_dir, 'train_tar.txt'), 'ab') as t:
                            np.savetxt(t, cor_tar, fmt='%d', delimiter=' ', newline='\r')
                            t.write(b'\n')
                        with open(os.path.join(split_results_dir, 'train_rec.txt'), 'ab') as m:
                            np.savetxt(m, rec_loss, fmt='%f', delimiter=' ', newline='\r')
                            m.write(b'\n')
                        pbar.set_description(("acc: {:.3f}".format(acc_item)))
                        pbar.update(batch_size)
                i_omn = 0
                open(os.path.join(split_results_dir, 'omn_fea.txt'), 'w').close()
                open(os.path.join(split_results_dir, 'omn_tar.txt'), 'w').close()
                open(os.path.join(split_results_dir, 'omn_pre.txt'), 'w').close()
                open(os.path.join(split_results_dir, 'omn_rec.txt'), 'w').close()
                for data_omn, target_omn in test_loader:
                    i_omn += 1
                    tar_omn = target_omn

                    if args.cuda:
                        data_omn = data_omn.cuda()
                    with torch.no_grad():
                        data_omn = Variable(data_omn)
                    predo_target_en = Variable(torch.eye(DATASET_CONFIGS[args.dataset]['classes'])).cuda().index_select(dim=0, index=torch.zeros_like(target_omn).cuda())
                    mu_omn, output_omn, de_omn = lvae.test(data_omn, predo_target_en)
                    output_omn = torch.exp(output_omn)
                    pre_omn = output_omn.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    rec_omn = (de_omn - data_omn).pow(2).sum((3, 2, 1))
                    mu_omn = torch.Tensor.cpu(mu_omn).detach().numpy()
                    tar_omn = torch.Tensor.cpu(tar_omn).detach().numpy()
                    pre_omn = torch.Tensor.cpu(pre_omn).detach().numpy()
                    rec_omn = torch.Tensor.cpu(rec_omn).detach().numpy()

                    with open(os.path.join(split_results_dir, 'omn_fea.txt'), 'ab') as f_test:
                        np.savetxt(f_test, mu_omn, fmt='%f', delimiter=' ', newline='\r')
                        f_test.write(b'\n')
                    with open(os.path.join(split_results_dir, 'omn_tar.txt'), 'ab') as t_test:
                        np.savetxt(t_test, tar_omn, fmt='%d', delimiter=' ', newline='\r')
                        t_test.write(b'\n')
                    with open(os.path.join(split_results_dir, 'omn_pre.txt'), 'ab') as p_test:
                        np.savetxt(p_test, pre_omn, fmt='%d', delimiter=' ', newline='\r')
                        p_test.write(b'\n')
                    with open(os.path.join(split_results_dir, 'omn_rec.txt'), 'ab') as l_test:
                        np.savetxt(l_test, rec_omn, fmt='%f', delimiter=' ', newline='\r')
                        l_test.write(b'\n')

    return best_val_acc, best_val_epoch


best_val_acc, best_val_epoch = train(args, lvae)
print('Finally!Best Epoch: {},  Best Val Loss: {:.4f}'.format(best_val_epoch, best_val_acc))