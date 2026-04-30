# Model Weights

The seven team checkpoints will be placed here once Prof. Harangi shares them.
This directory is gitignored.

## Expected layout

```
weights/
├── jng/                     # ViT — UNI / H-optimus / Gigapath foundation + LoRA
│   ├── uni_lora.safetensors
│   ├── h_optimus_lora.safetensors
│   └── gigapath_lora.safetensors
├── ymg/                     # ViT — MaxViT hybrid
│   └── maxvit.pth
├── ngu/                     # ViT — EVA-02
│   └── eva02.pth
├── gup/                     # CNN — ResNet ensemble
│   ├── resnet50.pth
│   ├── resnet101.pth
│   └── resnet152.pth
├── wan/                     # CNN — ResNet50 + LIME
│   └── resnet50_lime.pth
├── dpz/                     # Hybrid — SwinV2 + ConvNeXt + SE-ResNeXt
│   ├── swinv2.pth
│   ├── convnext.pth
│   └── se_resnext.pth
└── cha/                     # Hybrid — CNN + Swin Transformer
    └── cnn_swin.pth
```

## Loading

`src/ps3c_robust/baseline/models.py` exposes one builder per team
(`load_jng`, `load_ymg`, …). Each builder validates that the expected files
exist and returns a model in `eval()` mode on the requested device.

If naming from Prof. Harangi differs, update the constants in `models.py` —
do **not** rename the supplied files.
