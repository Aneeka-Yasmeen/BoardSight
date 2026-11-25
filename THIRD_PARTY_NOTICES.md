# Third-party notices

**Built with Llama.** Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. The Llama model and tokenizer are not included in this repository.

| Component | Licence | Use |
|---|---|---|
| Ultralytics YOLO | AGPL-3.0 | Detector training; `models/pcb_yolo11n.pt` and `pcb_yolo.onnx` come from it |
| ExecuTorch | BSD-3-Clause | Runs Llama on the phone |
| ONNX Runtime | MIT | Runs the detector on the phone |
| OpenCV | Apache-2.0 | Reference board comparison |
| AndroidX, CameraX, Jetpack Compose, Kotlin | Apache-2.0 | App framework |

## Datasets

| Dataset | Licence | Use here |
|---|---|---|
| PCB-Defect, Mendeley Data, doi:10.17632/vdj74sngvn.1 | CC BY 4.0 | Training and test. Resized, recompressed images in `demo/` and `android/app/src/main/assets/bench/` |
| DsPCBSD+, Figshare, doi:10.6084/m9.figshare.24970329 | CC BY 4.0 | Training and test. Four crops combined per image in `assets/bench/` |
| DeepPCB, github.com/tangsanli5201/DeepPCB | MIT | Training and test. Images in `demo/` and `assets/bench/`. The `2_mask_fault` sheets paste real DeepPCB defect patches onto clean DeepPCB images |
| PKU-Market-PCB, Huang & Wei, arXiv:1901.08204 | None stated by the authors | Training and test only. No images are included here |

`assets/bench/` file names begin with the dataset: `pcbdefect_`, `deeppcb_`, `deeppcb_clean_`, `dspcb_`.
