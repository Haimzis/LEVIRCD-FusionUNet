import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights


class Unet_Res50_Fusion(nn.Module):
    def __init__(self, finetune=True, out_channels=1, embed_dim=512):
        super(Unet_Res50_Fusion, self).__init__()
        self.name = 'unet_resnet_50'
        # Pretrained ResNet50 encoder
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.encoder1 = nn.Sequential(*list(resnet.children())[:3])   # Conv1 + BN1 + ReLU
        self.encoder2 = nn.Sequential(list(resnet.children())[4])    # Layer1
        self.encoder3 = nn.Sequential(list(resnet.children())[5])    # Layer2
        self.encoder4 = nn.Sequential(list(resnet.children())[6])    # Layer3
        self.encoder5 = nn.Sequential(list(resnet.children())[7])    # Layer4

        if not finetune:
            for param in resnet.parameters():
                param.requires_grad = False

        # Fusion layers for each skip connection
        self.fuse1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)   # Fuse x1_1 and x2_1
        self.fuse2 = nn.Conv2d(512, 128, kernel_size=3, padding=1)  # Fuse x1_2 and x2_2
        self.fuse3 = nn.Conv2d(1024, 256, kernel_size=3, padding=1)  # Fuse x1_3 and x2_3
        self.fuse4 = nn.Conv2d(2048, 512, kernel_size=3, padding=1) # Fuse x1_4 and x2_4
        self.fuse5 = nn.Conv2d(4096, embed_dim, kernel_size=3, padding=1) # Fuse x1_5 and x2_5

        # Decoder layers
        self.decoder5 = nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2)
        self.decoder4 = nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2)  # Skip connection with fused x1_5 and x2_5
        self.decoder3 = nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2)   # Skip connection with fused x1_4 and x2_4
        self.decoder2 = nn.Conv2d(256, 64, kernel_size=3, padding=1) #nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)    # Skip connection with fused x1_3 and x2_3
        self.decoder1 = nn.ConvTranspose2d(128, out_channels, kernel_size=2, stride=2)

        self.decoder5_bn = nn.BatchNorm2d(512)
        self.decoder4_bn = nn.BatchNorm2d(256)
        self.decoder3_bn = nn.BatchNorm2d(128)
        self.decoder2_bn = nn.BatchNorm2d(64)

        self.dropout = nn.Dropout(0.3)

    def forward(self, img1, img2):
        # Encode both images
        x1_1 = self.encoder1(img1)  # [B, 64, H/2, W/2]
        x1_2 = self.encoder2(x1_1)  # [B, 256, H/2, W/2]
        x1_3 = self.encoder3(x1_2)  # [B, 512, H/4, W/4]
        x1_4 = self.encoder4(x1_3)  # [B, 1024, H/8, W/8]
        x1_5 = self.encoder5(x1_4)  # [B, 2048, H/16, W/16]

        x2_1 = self.encoder1(img2)  # [B, 64, H/2, W/2]
        x2_2 = self.encoder2(x2_1)  # [B, 256, H/2, W/2]
        x2_3 = self.encoder3(x2_2)  # [B, 512, H/4, W/4]
        x2_4 = self.encoder4(x2_3)  # [B, 1024, H/8, W/8]
        x2_5 = self.encoder5(x2_4)  # [B, 2048, H/16, W/16]

        # Fuse features for skip connections
        skip1 = F.relu(self.fuse1(torch.cat((x1_1, x2_1), dim=1)))  # [B, 64, H/2, W/2]
        skip2 = F.relu(self.fuse2(torch.cat((x1_2, x2_2), dim=1)))  # [B, 256, H/2, W/2]
        skip3 = F.relu(self.fuse3(torch.cat((x1_3, x2_3), dim=1)))  # [B, 512, H/4, W/4]
        skip4 = F.relu(self.fuse4(torch.cat((x1_4, x2_4), dim=1)))  # [B, 1024, H/8, W/8]
        latent = F.relu(self.fuse5(torch.cat((x1_5, x2_5), dim=1))) # [B, 512, H/16, W/16]

        # Decoder with fused skip connections
        d5 = F.relu(self.decoder5_bn(self.decoder5(latent)))             # [B, 512, H/16, W/16]
        d4 = F.relu(self.decoder4_bn(self.decoder4(torch.cat((d5, self.dropout(skip4)), dim=1))))  # [B, 256, H/8, W/8]
        d3 = F.relu(self.decoder3_bn(self.decoder3(torch.cat((d4, self.dropout(skip3)), dim=1))))  # [B, 128, H/4, W/4]
        d2 = F.relu(self.decoder2_bn(self.decoder2(torch.cat((d3, skip2), dim=1))))  # [B, 64, H/2, W/2]
        d1 = self.decoder1(torch.cat((d2, skip1), dim=1))          # [B, out_channels, H, W]

        return d1


if __name__ == '__main__':
    # Instantiate the model
    model = Unet_Res50_Fusion()

    # Example input (224x224)
    img1 = torch.randn(1, 3, 224, 224)
    img2 = torch.randn(1, 3, 224, 224)

    # Forward pass
    output = model(img1, img2)
    print(output.shape)  # Should output torch.Size([1, 1, 224, 224])
