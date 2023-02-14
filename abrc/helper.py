# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
from PIL import Image
from datetime import datetime
import os, json, cv2, random, pathlib, shutil

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode


CATEGORIES = ["Open", "close", "Unknown"]
COLORS = {
    "Open": (255, 0, 0),  # Red
    "close": (0, 255, 0),  # Green
    "Unknown": (0, 0, 255)  # Blue
}

def filter_annotations(img_dir, label_filename):
    r""" Filter annotations by existing image files in `img_dir` folder
    
    Returns a dictionary with fields:
        "image": image file path
        "annotations": annotation JSON dict with Label studio format. E.g.,
            {
                "original_width": 1920,
                "original_height": 1280,
                "image_rotation": 0,
                "value": {
                    "x": 3.1,
                    "y": 8.2,
                    "radiusX": 20,
                    "radiusY": 16,
                    "ellipselabels": ["Car"]
                }
            }
    """
    orig_annotations = None
    with open(label_filename, "r") as f:
        orig_annotations = json.load(f)

    annotations = []
    for annotation in orig_annotations:
        # Process the label file, remove the first 9 random charactors 
        img_filename = os.path.basename(annotation["data"]["image"])[9:]
        
        # Check whether images exist with TIFF format
        filename, file_extension = os.path.splitext(img_filename)
        tif_filename = filename + '.tif'
        # if TIFF image file name starts with "2020", removes it
        if tif_filename.startswith("2020"):
            tif_filename = tif_filename[4:]
        tif_img_filepath = os.path.join(img_dir, tif_filename)
        
        if os.path.exists(tif_img_filepath):
            annotations.append({
                "image": tif_img_filepath,
                "annotations": annotation["annotations"][0]["result"]
            }) 
        else:
            print("Not exist: {}".format(tif_img_filepath))
    return annotations

def gen_ellipse_from_annotation(label):
    r"""
    Generate ellipse from Label Studio ellipse annotation
    Warning: this function only handles annotation with "image_rotation" = 0, otherwise, return False.

    Returns a tuple of
        - A bool flag to indicate whether the operation is successful
        - ellipse center (x, y)
        - ellipse axis (horizontal axis, vertical axis)
        - ellipse angle
        - category of the annotation

    """
    image_rotation = label["image_rotation"]
    if image_rotation != 0:
        return False, (0, 0), (0, 0), 0, ""
    img_w, img_h = label["original_width"], label["original_height"]
    rx, ry = label["value"]["radiusX"] * img_w / 100, label["value"]["radiusY"] * img_h / 100   # horizontal, verticle axies 
    # According to Label Studio, (x, y) coordinate of the top left corner before rotation (0, 100), but here it is actually the centre, weird.
    cx, cy = label["value"]["x"] * img_w / 100, label["value"]["y"] * img_h / 100
    angle = label["value"]["rotation"]  # clockwise degree
    category = label["value"]["ellipselabels"][0]
    return True, (cx, cy), (rx, ry), angle, category

def gen_polygon_from_annotation(label, delta=10):
    r""" 
    Generate polygon from Label Studio ellipse annotation
    Warning: this function only handles annotation with "image_rotation" = 0, otherwise, return False.

    Returns a tuple of
        - A bool flag to indicate whether the operation is successful
        - a closed polygon if successful, otherwise an empty list. E.g, [[x1, y1], [x2, y2], ...]
        - category of the annotation

    """
    success, center, axes, angle, category = gen_ellipse_from_annotation(label)
    if success:
        int_center = (int(center[0]), int(center[1]))
        int_axes = (int(axes[0]), int(axes[1]))
        int_angle = int(angle)
        poly = cv2.ellipse2Poly(center=int_center, axes=int_axes, angle=int_angle, arcStart=0, arcEnd=360, delta=delta)
        return True, poly, category
    else:
        return False, [], ""

def gen_polygon_w_boundingbox_from_annotation(label, delta=10):
    r"""
    Generate polygon and non-rotated bounding_box from Label Studio ellipse annotation
    Warning: this function only handles annotation with "image_rotation" = 0, otherwise, return False.

    Returns a tuple of
        - A bool flag to indicate whether the operation is successful
        - a closed polygon if successful, otherwise an empty list. E.g, [[x1, y1], [x2, y2], ...]
        - a bounding box in the format of (top_left_x, top_left_y, width, height)
        - category of the annotation

    """
    success, poly, category = gen_polygon_from_annotation(label, delta)
    if success:
        # bounding box format: (tlx, tly, w, h)
        bb = cv2.boundingRect(poly)
        return True, poly, bb, category
    else:
        return False, [], (0, 0, 0, 0), ""
    
def draw_annotations(img_filename, annotations, draw_polygon=True, draw_boundingbox=True, thickness=1):
    img = cv2.imread(img_filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img_h, img_w = img.shape[:2]
    # print("Image size = {}".format((img_h, img_w)))

    for v in annotations:
        if draw_boundingbox:
            success_bb, _, (x, y, w, h), category = gen_polygon_w_boundingbox_from_annotation(v, delta=10)
            if success_bb:
                cv2.rectangle(img, pt1=(x, y), pt2=(x+w, y+h), color=COLORS[category], thickness=thickness)
        if draw_polygon:
            success_poly, poly, category = gen_polygon_from_annotation(v)
            if success_poly:
                cv2.polylines(img, [poly], isClosed=True, thickness=thickness, color=COLORS[category])
                # cv2.fillConvexPoly(img, poly, color=COLORS[category])
    return img

def get_detectron2_dicts_raw(img_dir, json_filename, delta=5):
    labelstudio_annotations = filter_annotations(img_dir, json_filename)
    dataset_dicts = []
    for idx, v in enumerate(labelstudio_annotations):
        record = {}
        img_filename = v["image"]
        annotations = v["annotations"]
        img_h, img_w = cv2.imread(img_filename).shape[:2]
        
        record["file_name"] = img_filename
        record["image_id"] = idx
        record["height"] = img_h
        record["width"] = img_w

        objs = []
        for anno in annotations:
            if anno["original_width"] != record["width"] or anno["original_height"] != record["height"]:
                print("Generate record error!")
                return []
            
            success, poly, _, category = gen_polygon_w_boundingbox_from_annotation(anno, delta=delta)
            # Convert from [[x1, y1], [x2, y2], ...] to [x1, y1, x2, y2, ...]
            px = [x for x, _ in poly]
            py = [y for _, y in poly]
            poly = [(float(x), float(y)) for x, y in poly]
            poly = [p for x in poly for p in x]
            if success:
                if len(poly) <= 4:
                    continue
                obj = {
                    # Was numpy...changed to int32 to see if it works for json.dump()
                    #"bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox": [np.min(px).item(), np.min(py).item(), np.max(px).item(), np.max(py).item()],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,  # single category
                }
                objs.append(obj)
            else:
                print("Generate record error!")
                return []
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def get_detectron2_dicts(json_filename):
    with open(json_filename, 'r') as f:
        return json.load(f)
