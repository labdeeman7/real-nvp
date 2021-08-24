"""Train Real NVP on CIFAR-10.

Train script adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import util
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled=False


from models import RealNVP, RealNVPLoss
from tqdm import tqdm


def main(args):
    device = 'cuda' if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'
    start_epoch = 0

    # Note: No normalization applied, since RealNVP expects inputs in (0, 1).
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    if args.dataset == "MNIST": 
        no_of_channels = 1
        trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform_test)
    elif args.dataset == "CIFAR":
        no_of_channels = 3
        trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
    else:
        os.error("only MNIST and CIFAR currently supported working on LSUN")
    
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    print('Building model..')
    net = RealNVP(num_scales=2, in_channels=no_of_channels, mid_channels=64, num_blocks=8)
    # net = RealNVP(num_scales=2, in_channels=no_of_channels, mid_channels=8, num_blocks=4)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, args.gpu_ids)
        cudnn.benchmark = args.benchmark

    if args.resume:
        # Load checkpoint.
        print('Resuming from checkpoint at ckpts/best.pth.tar...')
        assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('ckpts/best.pth.tar')
        net.load_state_dict(checkpoint['net'])
        global best_loss
        global loss_arr_test
        global loss_arr_train
        best_loss = checkpoint['test_loss']
        loss_arr_test = checkpoint['loss_arr_test']
        loss_arr_train = checkpoint['loss_arr_train']
        start_epoch = checkpoint['epoch']

    loss_fn = RealNVPLoss()
    param_groups = util.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
    optimizer = optim.Adam(param_groups, lr=args.lr)

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train(epoch, net, trainloader, device, optimizer, loss_fn, args.max_grad_norm)
        test(epoch, net, testloader, device, loss_fn, args.num_samples, no_of_channels)

    #store loss and bpd values
    with open(f'samples/loss_arr_test.npy', 'wb') as f:
        np.save(f, loss_arr_test)
    with open(f'samples/loss_arr_train.npy', 'wb') as f:
        np.save(f, loss_arr_train)


def train(epoch, net, trainloader, device, optimizer, loss_fn, max_grad_norm):
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader:
            x = x.to(device)
            optimizer.zero_grad()
            z, sldj = net(x, reverse=False)
            loss = loss_fn(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()

            progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg))
            progress_bar.update(x.size(0))
        
        loss_arr_train.append((loss_meter.avg, util.bits_per_dim(x, loss_meter.avg))) 


def sample(net, batch_size, device, no_of_channels):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    z = torch.randn((batch_size, no_of_channels, 32, 32), dtype=torch.float32, device=device)
    x, _ = net(z, reverse=True)
    x = torch.sigmoid(x)

    return x


def test(epoch, net, testloader, device, loss_fn, num_samples, no_of_channels):
    global best_loss
    net.eval()
    loss_meter = util.AverageMeter()
    with torch.no_grad():
        with tqdm(total=len(testloader.dataset)) as progress_bar:
            for x, _ in testloader:
                x = x.to(device)
                z, sldj = net(x, reverse=False)
                loss = loss_fn(z, sldj)
                loss_meter.update(loss.item(), x.size(0))
                progress_bar.set_postfix(loss=loss_meter.avg,
                                         bpd=util.bits_per_dim(x, loss_meter.avg))
                progress_bar.update(x.size(0))

    # Save checkpoint
    print("loss_meter.avg", loss_meter.avg)
    print("best_loss", best_loss)
    
    state = {
        'net': net.state_dict(),
        'test_loss': loss_meter.avg,
        'epoch': epoch,
        'loss_arr_test': loss_arr_test,
        'loss_arr_train': loss_arr_train
    }
    os.makedirs('ckpts', exist_ok=True)
    torch.save(state, f'ckpts/epoch_{epoch}.pth.tar')

    if loss_meter.avg < best_loss:
        print('Saving best...')
        os.makedirs('ckpts', exist_ok=True)
        torch.save(state, 'ckpts/best.pth.tar')
        best_loss = loss_meter.avg
 
    # Save samples and data
    images = sample(net, num_samples, device, no_of_channels)
    os.makedirs('samples', exist_ok=True)
    images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, 'samples/epoch_{}.png'.format(epoch))
    torchvision.utils.save_image(images[0], 'samples/epoch_{}_specific.png'.format(epoch))

    loss_arr_test.append((loss_meter.avg, util.bits_per_dim(x, loss_meter.avg))) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealNVP on CIFAR-10/MNIST')

    parser.add_argument('--dataset', default="CIFAR", type=str, help='dataset name: CIFAR or MNIST are current options')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--benchmark', action='store_true', help='Turn on CUDNN benchmarking')
    parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=100., help='Max gradient norm for clipping')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--weight_decay', default=5e-5, type=float,
                        help='L2 regularization (only applied to the weight norm scale factors)')

    best_loss = 100000
    loss_arr_test = [] #tuple of loss, bpd for test
    loss_arr_train = [] #tuple of loss, bpd for train
    main(parser.parse_args())
