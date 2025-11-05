# infer.py
import torch
from PIL import Image
import cv2
from torchvision import transforms
from model import SiameseMobileNetV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 299
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.485,0.456,0.406], std =[0.229,0.224,0.225]),
])

def load_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def predict(before_path, after_path, ckpt="checkpoints/best.pth"):
    model = SiameseMobileNetV2(num_classes=7, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    b = tf(load_rgb(before_path)).unsqueeze(0).to(DEVICE)
    a = tf(load_rgb(after_path)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(b, a)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = prob.argmax()  
        # pred = 7 - pred           # balik arah skala
        pred += 1 # back to {1..7}
    return pred, prob

if __name__ == "__main__":
    pred, prob = predict(r"C:\Users\Sekar\Downloads\SKRIPSI\lefood\images\before\004_245_DSC_0949_bef.JPG",r"C:\Users\Sekar\Downloads\SKRIPSI\lefood\images\after\004_245_DSC_0978_aft.JPG")
    print("Predicted leftover level:", pred)
    print("Probs:", prob.round(3))

    pred, prob = predict(r"C:\Users\Sekar\Downloads\SKRIPSI\lefood\images\before\001_001_DSC_0059_bef.JPG",r"C:\Users\Sekar\Downloads\SKRIPSI\lefood\images\after\001_001_DSC_0108_aft.JPG")
    print("Predicted leftover level:", pred)
    print("Probs:", prob.round(3))
