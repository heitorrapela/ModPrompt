# RemovePoliceCVPR
from .assigners import BatchATSSAssigner, BatchTaskAlignedAssigner
from .coders import YOLOv5BBoxCoder, YOLOXBBoxCoder

__all__ = [
    'YOLOv5BBoxCoder', 'YOLOXBBoxCoder', 'BatchATSSAssigner',
    'BatchTaskAlignedAssigner'
]
