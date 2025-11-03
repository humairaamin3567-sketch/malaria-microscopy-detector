import torch
from torch import nn, optim
from src.data import prepare_dataloaders
from src.model import get_model
from src.utils import save_checkpoint, AverageMeter, set_seed
from tqdm import tqdm
import argparse, os

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()
    correct = 0
    total = 0
    for imgs, labels in tqdm(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), imgs.size(0))
        preds = out.argmax(dim=1)
        correct += (preds==labels).sum().item()
        total += imgs.size(0)
    return losses.avg, correct/total

def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            losses.update(loss.item(), imgs.size(0))
            preds = out.argmax(dim=1)
            correct += (preds==labels).sum().item()
            total += imgs.size(0)
    return losses.avg, correct/total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/processed')
    parser.add_argument('--model', default='efficientnet_b0')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)
    train_loader, val_loader = prepare_dataloaders(args.data_dir, args.batch_size)
    model = get_model(name=args.model, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = 0.0
    os.makedirs('experiments/baseline', exist_ok=True)
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            save_checkpoint({'epoch': epoch, 'model_state': model.state_dict()}, f'experiments/baseline/checkpoint.pth')

if __name__ == '__main__':
    main()
