# Detection that trains Stomata Inferences

Here we include the instruction for e

1. Prepare Your Data (data structure and augmentation)
2. Preperation Your Model (training, validation, testing)
3. Export Model to TorchScript and ONNX

## Prepare Your Data

You will need to access the files below:
- `data_preperation.ipynb` 
- `data_augmentation.ipynb` 
- `helper_data.py`- consist of helper functions data preperation tasks.


### STEP 1: Prepare Datasets

run `prepare_data` before training. This is a script that help to prepare your files for training, validation and testing. It randomly splits dataset into three subsets. Users need to define the ratio between them. Default ratio is **8:1:1**.

The dataset directory (indicated by `{dataset_dir}`) should have the same root as your `detection2` folder:

```
{root}
├── detection2
└── {dataset_dir}
```

Your dataset directory should follow the structure below in order to be compatiable with this script.

```
{dataset_dir}
├── {dataset_name}
│   ├── images           < here is where you store images
│   ├── labels
│   │   ├──labels.json   < here is where you store annotations
├── {dataset_name_2}     < you can store multiple dataset under the directory
│   ├── images           
│   ├── labels
│   │   ├──labels.json   
└── README.md 
```

### STEP 2: Data Augmentation

Before feeding your data to the model during training, you can augment your dataset to improve data quality and diversity. Since the number of labelled stomata data is limited, it is possible to cause overfitting. Using correct data augmentation tricks may significantly improve model performance.

> NOTE:\
> We don't train models from scratch. Data augmentation, therefore, may not be that important. So don't worry if you don't see much of improvement after applying data augmentation.

There are many ways to augmentation in computer vision and instance segmentation. Below are some examples:

- [How to do agumetnation for instance segmentation](https://www.kaggle.com/code/blondinka/how-to-do-augmentations-for-instance-segmentation)
- [Complete guide to data augmentation for computer vision](https://towardsdatascience.com/complete-guide-to-data-augmentation-for-computer-vision-1abe4063ad07)
- [Image data augmentation for computer vision](https://viso.ai/computer-vision/image-data-augmentation-for-computer-vision/#:~:text=Computer%20Vision%20Teams-,What%20Is%20Data%20Augmentation%3F,using%20label%2Dpreserving%20data%20transformations.)


## Prepare Your Model
- `stomata_train_new.ipynb` trains detectron2 models based on pretrained model `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml`.
- `stomata_traininference.ipynb` loads pre-trained models, and raw images as inputs and output labelled images with 1) bounding box 2) instance segmentation.
- `stomata-evaluation` evaluate model performance against ground truth.

### STEP 3: Training and Validation

Run `stomata_train.ipynb`

### STEP 5: Inference Example (Optional)

Run `stomata_inference.ipynb`

### STEP 6: Testing / Evaluation (Optional)

Run `stomata_evaluation.ipynb`

## Model Export

Detecetron2 provides a tool to export model from `*.pth` to `*.pt`. Use `/export_model.sh` script to manage arguments and export models.

``` bash
# Export stomata200-mix for CPU

python ../tools/deploy/export_model.py \
--config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
--format torchscript --export-method scripting \
--output export/stomata200-mix/cpu \
--sample-image ../../google-drive/stomata200-mix/images/0611_01-01.3.4.tif \
--device cpu \
MODEL.WEIGHTS output/stomata200-mix_output_ep32000_2023_02_14_19_32_50/train_output/model_final.pth \
MODEL.DEVICE cpu \
MODEL.ROI_HEADS.NUM_CLASSES 1 \
MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.3
```

> **NOTE**   
> There is a bug when using `/tools/deploy/export_model.py` provided by the official at line 83. It forces you to export for `cuda` devices when cuda is avaible. Therefore, you cannot export model for CPU on a CUDA-enabled machine. We modified the code exporting for CPU on a CUDA-enabled machine, ensure you set `--device cpu` when running the Python script.
