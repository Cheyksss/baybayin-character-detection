# Model Weights

Place your trained model file here:

```
Backend/models/rexnet_baybayin_final.pth
```

This file is produced at the end of the training notebook (`Baybayin_Model.ipynb`) by the cell:

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'num_classes': NUM_CLASSES,
    'class_to_idx': class_to_idx,
    'idx_to_class': idx_to_class,
    'img_size': IMG_SIZE,
    'imagenet_mean': IMAGENET_MEAN,
    'imagenet_std': IMAGENET_STD,
}, final_model_path)
```

Download it from your Google Drive checkpoint folder and copy it here before starting the backend server.

> **Without the model file** the API still starts and returns demo predictions so the frontend UI remains testable.
