## Introduction

YOLOv5-OBB is a variant of YOLOv5 that supports oriented bounding boxes. This model is designed to yield predictions that better fit objects that are positioned at an angle.

[X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) is not only an annotation tool, it’s a leap forward into the future of automated data annotation. It’s designed to not only simplify the process of annotation but also to integrate cutting-edge AI models for superior results. With a focus on practical applications, X-AnyLabeling strives to provide an industrial-grade, feature-rich tool that will assist developers in automating annotation and data processing for a wide range of complex tasks.

## Annotation

### Installation

```bash
cd yolov5_obb
git submodule update --init --recursive
cd X-AnyLabeling
pip install -r requirements.txt
# pip install -r requirements-gpu.txt
python anylabeling/app.py
```

### Toturial

- Prepare a predefined category label file (refer to [this](./X-AnyLabeling/assets/classes.txt)).
- Click on the 'Format' option in the top menu bar, select 'DOTA' and import the file prepared in the previous step.

[Option-1]
Basic usage

- Press the shortcut key "O" to create a rotation shape.
- Open edit mode (shortcut: "Ctrl+J") and click to select the rotation box. 
- rotate the selected box via shortcut "zxcv", where:
    - z: Large counterclockwise rotation
    - x: Small counterclockwise rotation
    - c: Small clockwise rotation
    - v: Large clockwise rotation

[Option-2]
Additionally, you can use the model to batch pre-label the current dataset.

- Press the shorcut key "Ctrl+A" to open the Auto-Labeling mode;
- Choose an appropriate default model or load a custom model.
- Press the shorcut key "Ctrl+M" to run all images once.

![YOLOv5m_obb_dota_result](./docs/YOLOv5m_obb_dota_result.png)

For more detail, you can refer to this [document](https://medium.com/@CVHub520/x-anylabeling-pioneering-the-annotation-revolution-eed0ae788f7d).

## Getting start

### Installation

- Requirements
    - Python 3.7+ 
    - PyTorch ≥ 1.7
    - CUDA 9.0 or higher
    - Ubuntu 16.04/18.04

Note: 
1. please be aware that if you downloaded the source code from the origin [repo](https://github.com/hukaixuan19970627/yolov5_obb.git), it is advisable to make necessary modifications to the poly_nms_cuda.cu file. Failing to do so will likely result in compilation issues.
2. For Windows user, please refer to this [issue](https://github.com/hukaixuan19970627/yolov5_obb/issues/224) if you have difficulty in generating utils/nms_rotated_ext.cpython-XX-XX-XX-XX.so)**

- Install

a. Create a conda virtual environment and activate it:

```bash
conda create -n yolov5_obb python=3.8 -y 
source activate yolov5_obb
```

b. Make sure your CUDA runtime api version ≤ CUDA driver version. (for example 11.3 ≤ 11.4)

```bash
nvcc -V
nvidia-smi
```

c. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/) based on your machine env, and make sure cudatoolkit version same as CUDA runtime api version, e.g.,

```bash
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
nvcc -V
python
>>> import torch
>>> torch.version.cuda
>>> exit()
```

d. Clone the modified version of the follow YOLOv5_OBB repository.
```
git clone https://github.com/CVHub520/yolov5_obb.git
cd yolov5_obb
```

e. Install yolov5-obb.

```python 
pip install -r requirements.txt
cd utils/nms_rotated
python setup.py develop  # or "pip install -v -e ."
```

- DOTA_devkit [Optional]

If you need to split the high-resolution image and evaluate the oriented bounding boxes (OBB), it is recommended to use the following tool:

```
cd yolov5_obb/DOTA_devkit
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```

### Datasets

Prepare custom dataset files

Note: Ensure that the label format is [polygon classname difficulty], for example, you can set **difficulty=0** unless otherwise specified.

```
  x1      y1       x2        y2       x3       y3       x4       y4       classname     diffcult

1686.0   1517.0   1695.0   1511.0   1711.0   1535.0   1700.0   1541.0   large-vehicle      1
```

![image](https://user-images.githubusercontent.com/72599120/159213229-b7c2fc5c-b140-4f10-9af8-2cbc405b0cd3.png)

Then, modify the path parameters and run this [script](./divide.py) if there is no need to split the high-resolution images. Otherwise, you can follow the steps below.

```shell
cd yolov5_obb
python DOTA_devkit/ImgSplit_multi_process.py
```

Ensure that your dataset is organized in the directory structure as shown below:

```
.
└── dataset_demo
    ├── images
    │   └── P0032.png
    └── labelTxt
        └── P0032.txt
```

Finally, you can create a custom data yaml file, e.g., [yolov5obb_demo.yaml](./data/yolov5obb_demo.yaml).

**Note:**
* DOTA is a high resolution image dataset, so it needs to be splited before training/testing to get better performance.
* For single-class problems, it is recommended to add a "None" class, effectively making it a 2-class task, e.g., [DroneVehicle_poly.yaml](./data/DroneVehicle_poly.yaml)

### Train/Val/Detect

> Before formally starting the training task, please follow the following recommendations:
> 1. Ensure that the input resolution is set to a multiple of 32.
> 2. By default, set the batch size to 8. If you increase it to 16 or larger, adjust the scaling factor for the box loss to help the convergence of the `theta`.

- To train on multiple GPUs with Distributed Data Parallel (DDP) mode, please refer to this shell [script](./sh/ddp_train.sh).

- To train the orignal dataset demo without split dataset, please refer to the following command:

```bash
python train.py \
  --weights weights/yolov5n.pt \
  --data data/task.yaml \
  --hyp data/hyps/obb/hyp.finetune_dota.yaml \
  --epochs 300 \
  --batch-size 1 \
  --img 1024 \
  --device 0 \
  --name /path/to/save_dir
```

- To detect a custom image file/folder/video, please refer to the following command:

```bash
python detect.py \
    --weights /path/to/*.pt \
    --source /path/to/image \
    --img 1024 \
    --device 0 \
    --conf-thres 0.25 \
    --iou-thres 0.2 \
    --name /path/to/save_dir
```

> Note: 

For more details, please refer to this [document](./docs/GetStart.md).

## Deploy

- Export *.onnx file:

```bash
python export.py \
    --weights runs/train/task/weights/best.pt \
    --data data/task.yaml \
    --imgsz 1024 \
    --simplify \
    --opset 12 \
    --include onnx
```

### Python

- Detect with the exported onnx file using onnxruntime:

```bash
python deploy/onnxruntime/python/main.py \
    --model /path/to/*.onnx \
    --image /path/to/image
```

### C++

- Enter the directory:

```bash
cd opencv/cpp

.
└── cpp
    ├── CMakeLists.txt
    ├── build
    ├── image
    │   ├── demo.jpg
    ├── main.cpp
    ├── model
    │   └── yolov5m_obb_csl_dotav15.onnx
    └── obb
        ├── include
        └── src
```

- Install the [OpenCV](https://github.com/opencv/opencv) and [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) libraries.

> Note, it is recommended to use OpenCV version 4.6.0 or newer, where v4.7.0 has been successfully tested.

- Place the images and model files in the specified directory.
- Modify the contents of the [CMakeLists.txt](./deploy/opencv/cpp/CMakeLists.txt), [main.cpp](./deploy/opencv/cpp/main.cpp), and [yolo_obb.h](./deploy/opencv/cpp/obb/include/yolo_obb.h) files according to your specific requirements and use case.
- Run the demo:

```bash
mkdir build && cd build
cmake ..
make
```

# Model Zoo

The results on **DOTA_subsize1024_gap200_rate1.0** test-dev set are shown in the table below. (**password: yolo**)

 |Model<br><sup>(download link) |Size<br><sup>(pixels) | TTA<br><sup>(multi-scale/<br>rotate testing) | OBB mAP<sup>test<br><sup>0.5<br>DOTAv1.0 | OBB mAP<sup>test<br><sup>0.5<br>DOTAv1.5 | OBB mAP<sup>test<br><sup>0.5<br>DOTAv2.0 | Speed<br><sup>CPU b1<br>(ms)|Speed<br><sup>2080Ti b1<br>(ms) |Speed<br><sup>2080Ti b16<br>(ms) |params<br><sup>(M) |FLOPs<br><sup>@640 (B) 
 | ----                                                                                                                                                           | ---  | ---   | ---      | ---   | ---   | ---   | ---   | --- | --- | ---
 |yolov5m [[baidu](https://pan.baidu.com/s/1UPNaMuQ_gNce9167FZx8-w)/[google](https://drive.google.com/file/d/1NMgxcN98cmBg9_nVK4axxqfiq4pYh-as/view?usp=sharing)]  |1024  | ×     |**77.3** |**73.2** |**58.0**  |**328.2**      |**16.9**     |**11.3**      |**21.6**   |**50.5**   
 |yolov5s [[baidu](https://pan.baidu.com/s/1Lqw42xlSZxZn-2gNniBpmw?pwd=yolo)]    |1024  | ×     |**76.8**   |-      |-      |-      |**15.6**  | -     |**7.5**     |**17.5**    
 |yolov5n [[baidu](https://pan.baidu.com/s/1Lqw42xlSZxZn-2gNniBpmw?pwd=yolo)]    |1024  | ×     |**73.3**   |-      |-      |-      |**15.2**  | -     |**2.0**     |**5.0**


<details>
  <summary>Table Notes (click to expand)</summary>

* All checkpoints are trained to 300 epochs with [COCO pre-trained checkpoints](https://github.com/ultralytics/yolov5/releases/tag/v6.0), default settings and hyperparameters.
* **mAP<sup>test dota</sup>** values are for single-model single-scale on [DOTA](https://captain-whu.github.io/DOTA/index.html)(1024,1024,200,1.0) dataset.<br>Reproduce Example:
 ```shell
 python val.py --data 'data/dotav15_poly.yaml' --img 1024 --conf 0.01 --iou 0.4 --task 'test' --batch 16 --save-json --name 'dotav15_test_split'
 python tools/TestJson2VocClassTxt.py --json_path 'runs/val/dotav15_test_split/best_obb_predictions.json' --save_path 'runs/val/dotav15_test_split/obb_predictions_Txt'
 python DOTA_devkit/ResultMerge_multi_process.py --scrpath 'runs/val/dotav15_test_split/obb_predictions_Txt' --dstpath 'runs/val/dotav15_test_split/obb_predictions_Txt_Merged'
 zip the poly format results files and submit it to https://captain-whu.github.io/DOTA/evaluation.html
 ```
* **Speed** averaged over DOTAv1.5 val_split_subsize1024_gap200 images using a 2080Ti gpu. NMS + pre-process times is included.<br>Reproduce by `python val.py --data 'data/dotav15_poly.yaml' --img 1024 --task speed --batch 1`

</details>

| Model Name                                      | File Size | Input Size | Configuration File                                      |
| ---------------------------------------------- | --------- | ---------- | -------------------------------------------------------- |
| [yolov5n_obb_drone_vehicle.onnx](https://github.com/CVHub520/X-AnyLabeling/releases/download/v1.0.0/yolov5n_obb_drone_vehicle.onnx) | 8.39MB   | 864       | [yolov5n_obb_drone_vehicle.yaml](./data/DroneVehicle_poly.yaml) |
| [yolov5s_obb_csl_dotav10.onnx](https://github.com/CVHub520/X-AnyLabeling/releases/download/v1.0.0/yolov5s_obb_csl_dotav10.onnx)   | 29.8MB   | 1024      | [dotav1_poly.yaml](./data/dotav1_poly.yaml)                 |
| [yolov5m_obb_csl_dotav15.onnx](https://github.com/CVHub520/X-AnyLabeling/releases/download/v1.0.0/yolov5m_obb_csl_dotav15.onnx) | 83.6MB   | 1024      | [dotav15_poly.yaml](./data/dotav15_poly.yaml)               |
| [yolov5m_obb_csl_dotav20.onnx](https://github.com/CVHub520/X-AnyLabeling/releases/download/v1.0.0/yolov5m_obb_csl_dotav20.onnx) | 83.6MB   | 1024      | [dotav2_poly.yaml](./data/dotav2_poly.yaml)                 |

## Acknowledgements

This project relies on the following open-source projects and resources:

* [hukaixuan19970627/yolov5_obb](https://github.com/hukaixuan19970627/yolov5_obb.git)
