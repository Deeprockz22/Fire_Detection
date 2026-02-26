# Fire Detection / Segmentation Tasks
- [x] Write `extract_frames.py` – Smokeview automation to extract RGB images and HRRPUV masks
- [x] Write `prepare_dataset.py` – Pairs raw images and binary masks, splits into train/val/test
- [x] Write `train_maskrcnn.py` – Mask R-CNN training loop using PyTorch/torchvision
- [x] Run `extract_frames.py` on the first 51 FDS scenarios
- [x] Run `prepare_dataset.py` to create the final paired dataset
- [x] Run `train_maskrcnn.py` to train the model on the initial subset
- [ ] Evaluate model on held-out test set
