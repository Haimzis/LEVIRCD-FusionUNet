import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from transformers import ViTModel
from tqdm import tqdm  # For progress bar

# 1. Dataset
class ImagePairDataset(Dataset):
    def __init__(self, dir_A, dir_B, label_dir, transform=None):
        self.dir_A = dir_A
        self.dir_B = dir_B
        self.label_dir = label_dir
        self.transform = transform
        self.image1_files = sorted(os.listdir(dir_A))
        self.image2_files = sorted(os.listdir(dir_B))
        self.label_files = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.image1_files)

    def __getitem__(self, idx):
        image1 = Image.open(os.path.join(self.dir_A, self.image1_files[idx])).convert('RGB')
        image2 = Image.open(os.path.join(self.dir_B, self.image2_files[idx])).convert('RGB')
        label = Image.open(os.path.join(self.label_dir, self.label_files[idx]))
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            label = self.transform(label)
        return image1, image2, label


class SegFormerDecoder(nn.Module):
    def __init__(self, in_channels=768, output_channels=1):
        super(SegFormerDecoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1)
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # Upsample x2
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.upsample2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # Upsample x2
        self.conv4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.upsample3 = nn.ConvTranspose2d(8, output_channels, kernel_size=8, stride=4, padding=2)  # Upsample x4
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.upsample1(x))
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.upsample2(x))
        x = nn.ReLU()(self.conv4(x))
        x = self.upsample3(x)
        x = self.activation(x)
        return x


# 2. Model with Cross-Attention
class CrossAttentionModel(nn.Module):
    def __init__(self, freeze_vit=True):
        super(CrossAttentionModel, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)

        # Freeze the ViT backbone if specified
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False

        # Replace decoder with SegFormerDecoder
        self.decoder = SegFormerDecoder(in_channels=768, output_channels=1)

    def forward(self, image1, image2):
        # Encode images using ViT backbone
        features1 = self.vit(pixel_values=image1).last_hidden_state  # Shape: [B, N, C]
        features2 = self.vit(pixel_values=image2).last_hidden_state  # Shape: [B, N, C]

        # Apply cross-attention
        attended_features, _ = self.cross_attention(features1, features2, features2)  # Shape: [B, N, C]

        # Remove class token
        attended_features = attended_features[:, 1:, :]  # Exclude the first token (class token)

        # Reshape to match spatial dimensions
        B, N, C = attended_features.size()  # Extract dimensions
        h = w = int(N**0.5)  # Compute height and width dynamically
        assert h * w == N, f"Shape mismatch: N={N}, h={h}, w={w}"  # Ensure valid spatial layout
        attended_features = attended_features.transpose(1, 2).view(B, C, h, w)  # Reshape to [B, C, H, W]

        # Decode features into a binary segmentation mask
        mask = self.decoder(attended_features)

        return mask



# 3. Loss Functions
def iou_score(pred, target):
    pred = (pred > 0.5).float()  # Convert probabilities to binary predictions
    intersection = (pred * target).sum(dim=[1, 2, 3])  # Sum over spatial dimensions
    union = (pred + target).sum(dim=[1, 2, 3]) - intersection
    iou = (intersection / (union + 1e-6)).mean()  # Average IOU across batch
    return iou


# 4. Training and Evaluation
def train_model(model, dataloader, val_dataloader, epochs, lr, device, log_dir):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    writer = SummaryWriter(log_dir)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_iou = 0
        progress = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for i, (image1, image2, label) in progress:
            image1, image2, label = image1.to(device), image2.to(device), label.to(device)
            optimizer.zero_grad()

            # Forward pass
            output = model(image1, image2)

            # Compute loss and IOU
            loss = criterion(output, label)
            iou = iou_score(output, label)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_iou += iou.item()

            # Update progress bar
            progress.set_postfix(loss=loss.item(), iou=iou.item())

        # Log images to TensorBoard
        binary_output = (output > 0.5).float()
        writer.add_images('Train/Input1', image1[:4], epoch)  # First 4 images of batch
        writer.add_images('Train/Input2', image2[:4], epoch)
        writer.add_images('Train/Label', label[:4], epoch)
        writer.add_images('Train/Output', binary_output[:4], epoch)

        # Log training metrics
        avg_train_loss = epoch_loss / len(dataloader)
        avg_train_iou = epoch_iou / len(dataloader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('IOU/train', avg_train_iou, epoch)

        # Validation
        model.eval()
        val_loss = 0
        val_iou = 0
        progress = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f"Epoch {epoch+1}/{epochs} [Val]")

        with torch.no_grad():
            for i, (image1, image2, label) in progress:
                image1, image2, label = image1.to(device), image2.to(device), label.to(device)

                # Forward pass
                output = model(image1, image2)

                # Compute loss and IOU
                loss = criterion(output, label)
                iou = iou_score(output, label)

                val_loss += loss.item()
                val_iou += iou.item()

                # Update progress bar
                progress.set_postfix(loss=loss.item(), iou=iou.item())

        # Log images to TensorBoard
        binary_output = (output > 0.5).float()
        writer.add_images('Validation/Input1', image1[:4], epoch)
        writer.add_images('Validation/Input2', image2[:4], epoch)
        writer.add_images('Validation/Label', label[:4], epoch)
        writer.add_images('Validation/Output', binary_output[:4], epoch)
                
        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_iou = val_iou / len(val_dataloader)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('IOU/val', avg_val_iou, epoch)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train IOU: {avg_train_iou:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val IOU: {avg_val_iou:.4f}")

    writer.close()


# 5. Main Execution
if __name__ == "__main__":
    # Directories
    dir_A = "data/train/A"
    dir_B = "data/train/B"
    label_dir = "data/train/label"

    # Directories
    dir_A_val = "data/val/A"
    dir_B_val = "data/val/B"
    label_dir_val = "data/val/label"

    # Hyperparameters
    batch_size = 16
    epochs = 20
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = "logs/semantic_segmentation"

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Dataset and DataLoader
    train_dataset = ImagePairDataset(dir_A, dir_B, label_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, pin_memory=True, persistent_workers=True, shuffle=True)

    val_dataset = ImagePairDataset(dir_A_val, dir_B_val, label_dir_val, transform=transform)  # Use a separate val set in practice
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, persistent_workers=True, shuffle=False)

    # Model
    model = CrossAttentionModel(freeze_vit=False)

    # Train
    train_model(model, train_loader, val_loader, epochs, lr, device, log_dir)
