import torch, argparse, PIL.Image as Image
from torchvision import transforms
from src.model import get_model

def load_image(path, img_size=224):
    tf = transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    img = Image.open(path).convert('RGB')
    return tf(img).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--image-path', required=True)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(name='efficientnet_b0', num_classes=2)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state['model_state'])
    model.to(device).eval()

    x = load_image(args.image_path)
    with torch.no_grad():
        out = model(x.to(device))
        prob = torch.softmax(out, dim=1).cpu().numpy()[0]
        cls = int(out.argmax(dim=1).cpu().numpy()[0])
    print(f"Predicted class: {cls}, prob={prob}")

if __name__ == '__main__':
    main()
