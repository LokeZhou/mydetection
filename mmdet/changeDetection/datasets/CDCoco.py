
from mmdet.datasets.registry import DATASETS
from mmdet.datasets.coco import CocoDataset

@DATASETS.register_module
class CDDataset(CocoDataset):

    CLASSES = ('change')