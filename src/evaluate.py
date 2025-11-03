import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import argparse, os
from src.model import get_model

def get_loader(data_dir, img_size=224, batch_size=32):
    tf = transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    ds = ImageFolder(os.path.join(data_dir, 'test'), transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4), ds.classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data_dir', default='data/processed')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader, classes = get_loader(args.data_dir)
    model = get_model(name='efficientnet_b0', num_classes=len(classes))
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state['model_state'])
    model.to(device).eval()

    ys, preds, probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            p = torch.softmax(out, dim=1)[:,1].cpu().numpy()
            pred = out.argmax(dim=1).cpu().numpy()
            ys.extend(labels.numpy().tolist())
            preds.extend(pred.tolist())
            probs.extend(p.tolist())

    print(classification_report(ys, preds, target_names=classes))
    print('Confusion matrix:')
    print(confusion_matrix(ys, preds))

if __name__ == '__main__':
    main()
