
# Change Detection with UNet ResNet50 Fusion

This project demonstrates a deep learning pipeline for change detection using a custom UNet architecture with ResNet50 as the backbone and fusion of multi-scale feature maps from two input images. The implementation includes training, evaluation, visualization, and prediction steps.

---

## Motivation

Change detection involves identifying changes between two images of the same geographical location, often used in applications like urban development, environmental monitoring, and disaster management.

While traditional approaches rely on post-processing or convolutional operations on difference masks, our approach directly fuses multi-scale features from two images in a UNet-style network. This provides more robust spatial and semantic feature alignment, improving detection accuracy.

---

## Dataset

[LEVIR-CD](https://paperswithcode.com/dataset/levir-cd) dataset consists of paired images (`InputA`, `InputB`) and corresponding ground truth (`Ground Truth`) masks for changes. These can be satellite images, aerial imagery, or other remotely sensed data. Each image pair highlights the differences between the two temporal states.

Directory structure:
```
data/
├── train/
│   ├── A/         # InputA images (e.g., before change)
│   ├── B/         # InputB images (e.g., after change)
│   ├── label/     # Ground truth masks
├── val/
    ├── A/         # Validation InputA
    ├── B/         # Validation InputB
    ├── label/     # Validation ground truth masks
```

---

## Architecture

### UNet-ResNet50 Fusion

The network extends the UNet structure with a ResNet50 backbone. Key components:
- **ResNet50 Backbone**: Frozen Pretrained ResNet50 layers extract features at multiple scales.
- **Feature Fusion**: Features from `InputA` and `InputB` are concatenated and processed with fusion layers at each scale, enabling better semantic alignment.
- **Decoder with Skip Connections**: The decoder reconstructs the change mask by incorporating fused features at each scale.

Why this approach?
- **Multi-scale Feature Fusion**: Combines semantic and spatial information of both signals effectively.
- **End-to-End Training**: Avoids reliance on post-processing steps, which may lose context.
- **Pretrained Backbone**: Leverages transfer learning from ImageNet for better feature extraction.

---

## Training Pipeline

1. **Loss Function**:
   - **CombinedIoULoss**: Combines Binary Cross-Entropy (BCE) and Intersection-over-Union (IoU) loss for better gradient behavior and overlap optimization.

2. **Metrics**:
   - Dice Score
   - Intersection-over-Union (IoU)

3. **Logging and Visualization**:
   - Uses TensorBoard and MLflow for loss/metric tracking.
   - Saves intermediate predictions for qualitative analysis.

4. **Augmentation**:
   - Random transformations (e.g., brightness, contrast, saturation) improve robustness.

---

## How to Use

### Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your dataset in the `data/` directory.

---

### Training

To train the model:
```bash
python main.py
```

Parameters like learning rate, batch size, and epochs can be adjusted in the `main.py` script.

---

### Results

The results are saved as:
- Validation metrics: Mean Dice score ~ 0.9, Positive IoU ~ 0.7 after 30 iterations.
- Visualizations of some predictions: `images/validation_predictions.png`.

Example visualization:
![Validation Predictions](images/validation_predictions.png)

---
## Visualizing Results

You can visualize results using `plot_predictions` from the provided code:
```python
plot_predictions(model, val_dataset, device="cuda", num_samples=10)
```
Output will be saved as `results.png`.

## Future Improvements

1. Further customizations and extended training — this is a basic proof of concept.
2. Extend the model for multi-class change detection.
3. Incorporate cross attention mechanisms for better common features alignment.
4. Experiment with transformer-based backbones.

---

## License

This project is licensed under the MIT License.
