# model.py
import torch, torch.nn as nn
from torchvision.models import mobilenet_v2

class SiameseMobileNetV2(nn.Module):
    def __init__(self, num_classes=7, pretrained=True, embed_dim=1280, dropout=0.2):
        super().__init__()
        base = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone = base.features  # shared
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        # head memproses concat([f_after, |f_after - f_before|])
        in_dim = embed_dim*2
        self.classifier = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(in_dim, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(512, num_classes)
            nn.Dropout(dropout),
            nn.Linear(in_dim, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Dropout(dropout),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 1024),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)
        )
        
    def forward_once(self, x):
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        return f

    def forward(self, x_before, x_after):
        f_b = self.forward_once(x_before)
        f_a = self.forward_once(x_after)
        feat = torch.cat([f_a, torch.abs(f_a - f_b)], dim=1)
        logits = self.classifier(feat)
        return logits

# data diitung pake weight, kita pakai observer untuk label karena hal tersebut ada nya kemungkinan data inkonsisten karena pengambilan data bersifat subjektif