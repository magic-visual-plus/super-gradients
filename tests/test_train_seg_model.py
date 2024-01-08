import os

from super_gradients.training import Trainer, dataloaders, models
from super_gradients.training.datasets import CoCoSegmentationDataSet
from super_gradients.training.datasets.segmentation_datasets.coco_segmentation import CoCoSegmentationDataSet

from super_gradients.training.utils.distributed_training_utils import setup_device
from loguru import logger

from super_gradients.training.transforms.transforms import (
    SegResize,
    SegRescale,
    SegRandomFlip,
    SegRandomRescale,
    SegCropImageAndMask,
    SegPadShortToCropSize,
    SegColorJitter,
    SegStandardize,
    SegNormalize,
    SegConvertToTensor
)

from super_gradients.training.utils.callbacks import BinarySegmentationVisualizationCallback, Phase
from super_gradients.training.metrics.segmentation_metrics import BinaryIOU, BinaryPrecisionRecall
from super_gradients.common.object_names import Models
 

num_classes = 1

cur_dir = os.path.dirname(os.path.abspath(__file__))
logger.info(f"cur_dir {cur_dir}")

crop_size = 1280

train_dataset_params = {
    'root_dir': '/opt/ml/datasets/jiaocheng',
    'samples_sub_directory': 'train',
    'targets_sub_directory': 'train',
    'list_file': '_annotations.coco.json',
    'dataset_classes_inclusion_tuples_list': [[0, 'bg'], [1, 'ng']],
    'cache_images': False,
    'cache_labels': False,
    'transforms': [
            SegRandomFlip(),
            SegRandomRescale(scales=[0.25, 4]),
            SegPadShortToCropSize(crop_size=crop_size),
            SegCropImageAndMask(crop_size=crop_size, mode="random"),
            SegStandardize(max_value=255),
            SegNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            SegConvertToTensor(),
        ],
}

train_dataloader_params = {
    'batch_size': 4,
    'drop_last': True,
    'num_workers': 1,
    'shuffle': True,
}

train_dataset = CoCoSegmentationDataSet(**train_dataset_params)

val_dataset_params = {
    'root_dir': '/opt/ml/datasets/jiaocheng',
    'samples_sub_directory': 'valid',
    'targets_sub_directory': 'valid',
    'list_file': '_annotations.coco.json',
    'dataset_classes_inclusion_tuples_list': [[0, 'bg'], [1, 'ng']],
    'cache_images': False,
    'cache_labels': False,
    'transforms': [
            SegRescale(short_size=crop_size),
            SegCropImageAndMask(crop_size=crop_size, mode="random"),
            SegStandardize(max_value=255),
            SegNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            SegConvertToTensor(),
    ],
}
val_dataset = CoCoSegmentationDataSet(**val_dataset_params)

train_dataloader = dataloaders.build_data_loader(
    config_name="coco_segmentation_dataset_params",
    dataset=train_dataset,
    train=True,
    dataset_params=None,
    dataloader_params=train_dataloader_params,
)

val_dataloader_params = {
    'batch_size': 4,
    'drop_last': False,
    'num_workers': 1,
}


val_dataloader = dataloaders.build_data_loader(
    config_name="coco_segmentation_dataset_params",
    dataset=val_dataset,
    train=False,
    dataset_params=None,
    dataloader_params=val_dataloader_params,
)

print("Dataloader parameters:")
print(train_dataloader.dataloader_params)
print("Dataset parameters")
print(train_dataloader.dataset)


# train model part
model = models.get(Models.REGSEG48, num_classes=num_classes)

train_params = {
    "max_epochs": 200,
    "lr_mode": "CosineLRScheduler",
    "initial_lr": 0.008,  # for batch_size=16
    "optimizer_params": {"momentum": 0.843, "weight_decay": 0.00036, "nesterov": True},
    "cosine_final_lr_ratio": 0.1,
    "multiply_head_lr": 10,
    "optimizer": "SGD",
    "loss": "BCEDiceLoss",
    "ema": True,
    "zero_weight_decay_on_bias_and_bn": True,
    "average_best_models": True,
    "mixed_precision": False,
    # "metric_to_watch": "target_IOU",
    "metric_to_watch": "mean_IOU",
    "greater_metric_to_watch_is_better": True,
    "train_metrics_list": [BinaryIOU(), BinaryPrecisionRecall()],
    "valid_metrics_list": [BinaryIOU(), BinaryPrecisionRecall()],
    "phase_callbacks": [BinarySegmentationVisualizationCallback(phase=Phase.VALIDATION_BATCH_END, freq=1, last_img_idx_in_batch=4)],
}


CHECKPOINT_DIR = "./notebook_ckpts/"
trainer = Trainer(experiment_name="regseg_jcheng", ckpt_root_dir=CHECKPOINT_DIR)
trainer.train(model=model, training_params=train_params, train_loader=train_dataloader, valid_loader=val_dataloader)



test_dataset_params = {
    'root_dir': '/opt/ml/datasets/jiaocheng',
    'samples_sub_directory': 'test',
    'targets_sub_directory': 'test',
    'list_file': '_annotations.coco.json',
    'dataset_classes_inclusion_tuples_list': [[0, 'bg'], [1, 'ng']],
    'cache_images': False,
    'cache_labels': False,
    'transforms': [
            SegRescale(short_size=crop_size),
            SegCropImageAndMask(crop_size=crop_size, mode="random"),
            SegStandardize(max_value=255),
            SegNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            SegConvertToTensor(),
    ],
}
test_dataset = CoCoSegmentationDataSet(**test_dataset_params)

test_dataloader = dataloaders.build_data_loader(
    config_name="coco_segmentation_dataset_params",
    dataset=test_dataset,
    train=False,
    dataset_params=None,
    dataloader_params=val_dataloader_params,
)

test_results = trainer.test(model=model, test_loader=test_dataloader, test_metrics_list=[BinaryIOU(), BinaryPrecisionRecall()])
logger.info("test_results {}", test_results)