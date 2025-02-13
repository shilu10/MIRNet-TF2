# MIRNet-TF2: Image Restoration with TensorFlow 2.x

A complete re-implementation of [MIRNet (CVPR)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zamir_Learning_Enriched_Features_for_Real_Image_Restoration_and_Enhancement_CVPR_2020_paper.pdf) in **TensorFlow 2.x** for tasks like:
- **Denoising**
- **Low-light enhancement**
- **Super-resolution**

Supports training, evaluation, visualization, and TensorFlow Lite (TFLite) export for edge deployment.

---

## 📁 Project Structure

```bash
.
├── mirnet/                       # Core model and backbone
│   ├── __init__.py
│   ├── mir_backbone.py
│   └── models.py
├── dataloaders/                 # Custom dataloaders for LOL, SIDD, and SR
│   ├── lol_dataloader.py
│   ├── sidd_dataloader.py
│   ├── sr_dataloader.py
│   └── utils.py
├── notebooks/                   # Jupyter notebooks for visualization/training
│   ├── mirnet-all-training.ipynb
│   ├── mirnet-denoising.ipynb
│   ├── mirnet-enhancement.ipynb
│   └── mirnet-super-resolution.ipynb
├── dataset_visualization/       # EDA and visualization notebooks
├── pretrained_weights/          # Pretrained model weights (TF Checkpoints)
│   ├── denoise/
│   ├── enhancement/
│   └── super_resolution/
├── results/                     # Output samples from test runs
│   ├── denoise/
│   ├── enhancement/
│   ├── super_resolution/
│   └── tflite/                  # TFLite inference results
├── test/                        # Raw test images
├── gt/                          # Ground truth images
├── *.py                         # Scripts for training, testing, TFLite
```


## 🚀 Features
✅ TensorFlow 2.x implementation of MIRNet

✅ Supports 3 tasks: denoising, enhancement, super-resolution

✅ Modular dataloaders for LOL, SIDD, and custom datasets

✅ Training scripts and Jupyter notebooks

✅ TFLite model conversion & inference

✅ PSNR/SSIM evaluation metrics

✅ Visualization of output vs ground truth


## 🧪 Training
Train for different tasks using the following scripts:
```bash

# Denoising
python train_denoise.py

# Low-light enhancement
python train_enhancement.py

# Super-resolution
python train_super_resolution.py
```

## 📈 Evaluation
```bash
# Denoising
python test_denoise.py

# Enhancement
python test_enhancement.py

# Super-resolution
python test_super_resolution.py
```

Results are saved in the results/ directory.

## 📱 TFLite Conversion & Inference

```bash
python tflite_conversion.py      # Convert model to TFLite
python tflite_inference.py       # Run inference with TFLite model
```

## 🧠 Notebooks
Explore Jupyter notebooks in the notebooks/ and dataset_visualization/ folders for:

- Training walkthrough

- Model inspection

- Dataset visualization

- TFLite optimization preview


## 📊 Sample Results

| Task               | Dataset | PSNR ↑ | SSIM ↑ | Notes                                 |
|--------------------|---------|--------|--------|---------------------------------------|
| Denoising          | SIDD    | 32.7   | 0.930  | Trained for 50 epochs on ~10K patches |
| Enhancement        | LOL     | 26.5   | 0.880  | Mixed low-light and synthetic dataset |
| Super-Resolution   | DIV2K   | 29.1   | 0.910  | 2x upscaling, pretrained from scratch |
| TFLite Denoising   | SIDD    | 32.1   | 0.924  | Slight degradation due to quantization |
| TFLite Enhancement | LOL     | 25.9   | 0.872  | Converted with float16 optimization   |

## 🏗️ Future Work

| Feature/Improvement                       | Status       | Priority | Notes                                                                 |
|------------------------------------------|--------------|----------|-----------------------------------------------------------------------|
| ✅ Modular TFLite export support          | Completed    | High     | Exported models for all tasks with float16 precision                  |
| 🛠️ Add ONNX export                       | In Progress  | Medium   | Enables cross-framework inference (e.g., TensorRT, OpenVINO)          |
| 🔲 Integrate TensorBoard for metrics      | Not Started  | Medium   | Helps visualize training curves, losses, and learning rates           |
| 🔲 Support for Deblurring task            | Not Started  | Low      | Extend current MIRNet to handle motion blur datasets like GoPro       |
| 🔲 Add CI workflow for testing            | Not Started  | Medium   | Automate tests for inference and output shapes                        |
| 🔲 Hyperparameter tuning module (Optuna)  | Not Started  | Low      | Automate search over learning rate, loss weights, augmentations       |
| 🔲 Support dataset auto-download          | Not Started  | Low      | Integrate Kaggle/GDrive-based downloads for LOL/SIDD/DIV2K            |
| 🔲 Quantitative benchmark vs baseline     | Not Started  | High     | Add ResNet/UNet baseline for PSNR/SSIM comparison                     |
