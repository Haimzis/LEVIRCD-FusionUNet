import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchmetrics.classification import JaccardIndex, Dice
import mlflow
from model import Unet_Res50_Fusion

import matplotlib.pyplot as plt
import random


class CombinedIoULoss(torch.nn.Module):
    def __init__(self, alpha=0.5, smooth=1e-6):
        """
        Combined IoU and BCE Loss for better gradient behavior.
        Args:
            alpha (float): Weight for BCE loss. (1-alpha) will be the weight for IoU loss.
            smooth (float): Smoothing factor for IoU calculation.
        """
        super(CombinedIoULoss, self).__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Forward pass for Combined IoU and BCE Loss.
        Args:
            logits (torch.Tensor): Predicted logits (before applying sigmoid), shape (N, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks, shape (N, 1, H, W).
        Returns:
            torch.Tensor: Combined IoU and BCE loss.
        """
        # Apply BCE loss
        bce = self.bce_loss(logits, targets)

        # Compute IoU loss
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum() - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        iou_loss = 1 - iou

        # Combine BCE and IoU losses
        combined_loss = self.alpha * bce + (1 - self.alpha) * iou_loss
        return combined_loss


def plot_predictions(model, dataset, device, num_samples=10):
    """
    Plots samples: InputA, InputB, Ground Truth, and Predictions.

    Args:
        model: Trained model for inference.
        dataset: Dataset to sample from.
        device: Device (CPU or GPU) where model will run.
        num_samples: Number of samples to plot.
        iou_metric: JaccardIndex metric instance for IoU calculation.
    """
    model.eval()
    model.to(device)

    samples_with_scores = []

    # Iterate over the dataset to compute IoU for all samples
    for idx in range(len(dataset)):
        image1, image2, label = dataset[idx]
        image1 = image1.unsqueeze(0).to(device)  # Add batch dimension
        image2 = image2.unsqueeze(0).to(device)  # Add batch dimension
        label = label.unsqueeze(0).to(device)    # Add batch dimension

        # Get model prediction
        with torch.no_grad():
            output = model(image1, image2)
            prediction = torch.sigmoid(output).squeeze(0)

        samples_with_scores.append((image1.squeeze(0), image2.squeeze(0), label.squeeze(0), prediction.cpu()))

    samples_with_scores = samples_with_scores[:num_samples]

    # Visualization
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, num_samples * 5))
    fig.subplots_adjust(wspace=0.01, hspace=0.01)  # Reduce spacing between images

    for i, (image1, image2, label, prediction) in enumerate(samples_with_scores):
        # Convert tensors to NumPy arrays
        image1_np = image1.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
        image2_np = image2.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
        label_np = label.squeeze(0).cpu().numpy()  # Remove extra channel dimension
        prediction_np = prediction.squeeze(0).cpu().numpy()  # Remove extra batch dimension

        # Denormalize RGB images for proper visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3).cpu().numpy()
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3).cpu().numpy()
        image1_np = (image1_np * std + mean).clip(0, 1)
        image2_np = (image2_np * std + mean).clip(0, 1)

        # Plot images
        axes[i, 0].imshow(image1_np)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(image2_np)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(label_np, cmap="gray")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(prediction_np > 0.5, cmap="gray")
        axes[i, 3].axis("off")

    # Set shared titles for columns
    columns = ["InputA", "InputB", "Ground Truth", "Prediction"]
    for ax, col in zip(axes[0], columns):
        ax.set_title(col, fontsize=20)

    plt.tight_layout()
    plt.savefig('results.png', pad_inches=0)


class ImagePairDataset(Dataset):
    def __init__(self, dir_A, dir_B, label_dir, image_transform=None, label_transform=None):
        self.dir_A = dir_A
        self.dir_B = dir_B
        self.label_dir = label_dir
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.image1_files = sorted(os.listdir(dir_A))
        self.image2_files = sorted(os.listdir(dir_B))
        self.label_files = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.image1_files)

    def __getitem__(self, idx):
        image1 = Image.open(os.path.join(self.dir_A, self.image1_files[idx])).convert('RGB')
        image2 = Image.open(os.path.join(self.dir_B, self.image2_files[idx])).convert('RGB')
        label = Image.open(os.path.join(self.label_dir, self.label_files[idx]))
        if self.image_transform:
            image1 = self.image_transform(image1)
            image2 = self.image_transform(image2)
            label = self.label_transform(label)
        return image1, image2, label


# Training and Evaluation with TensorBoard Logs and pos_weight as a Parameter
def train_model(model, dataloader, val_dataloader, epochs, lr, pos_weight, device, log_dir, params):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    criterion = CombinedIoULoss(alpha=0.7)
    dice_metric = Dice(num_classes=2, average='macro', multiclass="False").to(device)
    iou_metric = JaccardIndex(task="binary", average="macro").to(device)

    # Initialize MLflow
    mlflow.set_experiment("Semantic Segmentation Experiment")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        log_dir = os.path.join(log_dir, run_id)
        os.makedirs(log_dir, exist_ok=True)

        writer = SummaryWriter(log_dir)

        # Log all arguments (parameters)
        params["pos_weight"] = pos_weight  # Add pos_weight to the logged parameters
        mlflow.log_params(params)
        mlflow.log_param("model_name", model.__class__.__name__)  # Log the model module name

        best_val_loss = float("inf")  # Initialize the best validation loss
        patience = 10  # Early stopping patience
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            model.train()
            epoch_loss = 0
            dice_metric.reset()
            iou_metric.reset()

            progress = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs} [Train]")

            for i, (image1, image2, label) in progress:
                image1, image2, label = image1.to(device), image2.to(device), label.to(device)
                optimizer.zero_grad()

                # Forward pass
                output = model(image1, image2)
                probs = torch.sigmoid(output)

                # Compute loss and update metrics
                loss = criterion(output, label)
                dice_metric.update((probs > 0.5).int(), label.int())
                iou_metric.update((probs > 0.5).int(), label.int())

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # Update progress bar
                progress.set_postfix(loss=loss.item())


            # Log images to TensorBoard
            binary_output = (probs > 0.5).float()
            writer.add_images('Train/Input1', image1[:4], epoch)
            writer.add_images('Train/Input2', image2[:4], epoch)
            writer.add_images('Train/Label', label[:4], epoch)
            writer.add_images('Train/Output', binary_output[:4], epoch)
            
            # Compute final metrics for training
            avg_train_loss = epoch_loss / len(dataloader)
            avg_train_dice = dice_metric.compute().item()
            avg_train_iou = iou_metric.compute().item()

            # Log training metrics to MLflow
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("train_iou", avg_train_iou, step=epoch)
            mlflow.log_metric("train_dice", avg_train_dice, step=epoch)

            # Log training metrics to TensorBoard
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('IOU/train', avg_train_iou, epoch)
            writer.add_scalar('Dice/train', avg_train_dice, epoch)

            # Validation
            model.eval()
            val_loss = 0
            dice_metric.reset()
            iou_metric.reset()

            progress = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f"Epoch {epoch+1}/{epochs} [Val]")

            with torch.no_grad():
                for i, (image1, image2, label) in progress:
                    image1, image2, label = image1.to(device), image2.to(device), label.to(device)

                    # Forward pass
                    output = model(image1, image2)
                    probs = torch.sigmoid(output)

                    # Compute loss and update metrics
                    loss = criterion(output, label)
                    dice_metric.update((probs > 0.5).int(), label.int())
                    iou_metric.update((probs > 0.5).int(), label.int())

                    val_loss += loss.item()

                    # Update progress bar
                    progress.set_postfix(loss=loss.item())

            # Log images to TensorBoard
            binary_output = (probs > 0.5).float()
            writer.add_images('Validation/Input1', image1[:4], epoch)
            writer.add_images('Validation/Input2', image2[:4], epoch)
            writer.add_images('Validation/Label', label[:4], epoch)
            writer.add_images('Validation/Output', binary_output[:4], epoch)

            # Compute final metrics for validation
            avg_val_loss = val_loss / len(val_dataloader)
            avg_val_dice = dice_metric.compute().item()
            avg_val_iou = iou_metric.compute().item()

            # Log validation metrics to MLflow
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_iou", avg_val_iou, step=epoch)
            mlflow.log_metric("val_dice", avg_val_dice, step=epoch)

            # Log validation metrics to TensorBoard
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            writer.add_scalar('IOU/val', avg_val_iou, epoch)
            writer.add_scalar('Dice/val', avg_val_dice, epoch)

            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0  # Reset patience counter
                best_model_path = "best_model.pth"
                torch.save(model.state_dict(), best_model_path)
                mlflow.log_artifact(best_model_path)
                os.remove(best_model_path)  # Clean up local storage
            else:
                patience_counter += 1  # Increment patience counter
                if patience_counter > patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}, Train IOU: {avg_train_iou:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}, Val IOU: {avg_val_iou:.4f}")

        # Log the TensorBoard logs to MLflow
        writer.flush()  # Ensure all logs are written
        mlflow.log_artifact(log_dir, artifact_path="tensorboard_logs")

    writer.close()


# Main Execution
if __name__ == "__main__":
    log_dir = "logs/semantic_segmentation"

    # Directories
    dir_A = "data/train/A"
    dir_B = "data/train/B"
    label_dir = "data/train/label"

    dir_A_val = "data/val/A"
    dir_B_val = "data/val/B"
    label_dir_val = "data/val/label"

    # Hyperparameters
    batch_size = 16
    epochs = 75
    lr = 1e-3
    pos_weight = 7.5  # Add pos_weight here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 512

    # Save all parameters and hardcoded arguments
    params = {
        "dir_A": dir_A,
        "dir_B": dir_B,
        "label_dir": label_dir,
        "dir_A_val": dir_A_val,
        "dir_B_val": dir_B_val,
        "label_dir_val": label_dir_val,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "pos_weight": pos_weight,
        "image_size": (image_size, image_size),
        "augmentation": "ColorJitter (brightness=0.35, contrast=0.25, saturation=0.2, hue=0.1)",
    }

    # Data transforms
    training_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ColorJitter(brightness=0.35, contrast=0.25, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    validation_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    label_tranform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Dataset and DataLoader
    train_dataset = ImagePairDataset(dir_A, dir_B, label_dir, image_transform=training_transform, label_transform=label_tranform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, pin_memory=True, shuffle=True, prefetch_factor=5)

    val_dataset = ImagePairDataset(dir_A_val, dir_B_val, label_dir_val, image_transform=validation_transform, label_transform=label_tranform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False)

    # Model
    model = Unet_Res50_Fusion(finetune=False, out_channels=1)

    # Train
    train_model(model, train_loader, val_loader, epochs, lr, pos_weight, device, log_dir, params)
    plot_predictions(model, val_dataset, device, num_samples=10)
