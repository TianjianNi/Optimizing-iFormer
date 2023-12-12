import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DataParallel as DP
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import os
from time import perf_counter
from model import iformer_small


def parse_args():
    parser = argparse.ArgumentParser(description='Training script for iFormer model')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--data_path', type=str, default='/scratch/tn2151/project/cifar10-128', help='Path to the training data')
    parser.add_argument("--epochs", default=2, type=int, help='Use the first epoch as warmup')

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999), help="Betas for AdamW")
    parser.add_argument("--eps", type=float, default=1e-08, help="Epsilon for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")

    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')

    parser.add_argument('--parallel_mode', choices=['none', 'DP', 'DDP'], default='none',
                        help='Parallelization mode: none, DataParallel, DistributedDataParallel')
    args = vars(parser.parse_args())

    return args


class Customized_CIFAR10_Dataset(Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(os.path.join(self.root, 'train')) if
                               os.path.isdir(os.path.join(self.root, 'train', d))])

        self.data = []
        for class_id, class_name in enumerate(self.classes):
            class_path = os.path.join(self.root, 'train', class_name)
            for filename in os.listdir(class_path):
                if filename.lower().endswith('.png') and filename.startswith('img'):
                    image_path = os.path.join(class_path, filename)
                    # Extract the image number from the filename
                    img_number = int(filename[3:-4])  # Assuming 'img' prefix and '.png' suffix
                    self.data.append((image_path, class_id, img_number))

        # Sort the data based on the image number
        self.data.sort(key=lambda x: x[2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label, img_number = self.data[index]
        image = datasets.folder.default_loader(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label


def ddp_setup(rank: int, world_size: int):
    """
    Args:
    rank: Unique identifier of each process
    world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def train_one_epoch(train_loader, model, criterion, optimizer, device):
    train_loss = 0
    correct = 0
    total = 0
    data_loading_time = 0.0
    training_time = 0.0
    size = len(train_loader)

    start = data_loading_checkpoint = perf_counter()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        data_loading_time += (perf_counter() - data_loading_checkpoint)

        training_start_time = perf_counter()

        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        training_time += (perf_counter() - training_start_time)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        data_loading_checkpoint = perf_counter()

    running_time = perf_counter() - start

    accuracy = 100. * correct / total
    avg_train_loss = train_loss / size

    return accuracy, avg_train_loss, data_loading_time, training_time, running_time


def main():

    args = parse_args()
    batch_size = args['batch_size']
    data_path = args['data_path']
    epochs = args['epochs']

    lr = args['lr']
    betas = args['betas']
    eps = args['eps']
    weight_decay = args['weight_decay']

    seed = args['seed']
    torch.manual_seed(seed)

    parallel_mode = args['parallel_mode']

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = Customized_CIFAR10_Dataset(root=data_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if parallel_mode == 'none':
        model = iformer_small(pretrained=False)
        model = model.to(device)
    elif parallel_mode == 'DP':
        model = iformer_small(pretrained=False)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = DP(model)
        model = model.to(device)
    elif parallel_mode == 'DDP':
        args.distributed = False
        if 'WORLD_SIZE' in os.environ:
            args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.device = 'cuda:0'
        args.world_size = 1
        args.rank = 0  # global rank
        if args.distributed:
            args.device = 'cuda:%d' % args.local_rank
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()
            print(
                'Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                % (args.rank, args.world_size))
        else:
            print('Training with a single process on 1 GPUs.')
        assert args.rank >= 0
        model = iformer_small(pretrained=False)
        # setup distributed training
        if args.distributed:
            if args.local_rank == 0:
                print("Using native Torch DistributedDataParallel.")
                model = DDP(model, device_ids=[args.local_rank], broadcast_buffers=not args.no_ddp_bb)



    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()

        accuracy, avg_train_loss, data_loading_time, training_time, running_time = train_one_epoch(data_loader, model, criterion, optimizer, device)

        print(f"Epoch {epoch + 1}/{epochs} ")
        print(f"Data loading time (sec) is {data_loading_time:.3f}")
        print(f"Training time (sec) is {training_time:.3f}")
        print(f"Running time (sec) is {running_time:.3f}")
        print(f"Loss is {avg_train_loss:.3f} Accuracy is {accuracy}%")


if __name__ == "__main__":
    main()