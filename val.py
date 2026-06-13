import numpy as np
from ultralytics.utils.torch_utils import model_info

if __name__ == "__main__":
    model_path = "RPN_YOLO.pt"
    result = model.val(
        data="./TT100K/data.yaml",
        split="val", 
        imgsz=640,
        batch=48,
    )
