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

from skimage import measure

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
        if file_extension == '.tif':
            img_filename = filename + '.tif'
        elif file_extension == '.png':
            img_filename = filename + '.png'
        elif file_extension == '.jpg':
            img_filename = filename + '.jpg'
        else:
            img_filename = filename
        # if TIFF image file name starts with "2020", removes it
        if img_filename.startswith("2020"):
            img_filename = img_filename[4:]
        img_filepath = os.path.join(img_dir, img_filename)
        
        if os.path.exists(img_filepath):
            annotations.append({
                "image": img_filepath,
                "annotations": annotation["annotations"][0]["result"]
            }) 
        else:
            print("Not exist: {}".format(img_filepath))
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

def get_detectron2_dicts_abrc(img_dir: str, json_filename: str, delta=5) -> list:
    r"""
    This function parse the JSON file prepared by ABRC with Label Studio into a list of COCO compatible annotation dictionaries.
    
    Retrun a list of:
        - COCO compatiable annotation dictionaries. Each dictionay consists of these indices:
            - file_name
            - image_id
            - height, 
            - width, 
            - annotations
    """
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


def shuffle_datset(label_dicts: list, seed_num):
    r"""
    # random shuffle a list of label dictionaries
    
    Args:
     - label_dicts (list): list of label dictionaries consisting annotations.
     - seed_num: seed for Python randon function.
    """    
    random.seed(seed_num)
    random.shuffle(label_dicts)
    print("Randomly shuffle label dicts...")



def split_dataset(label_dicts: list, split_ratio: list) -> dict:
    r"""
    split dataset into three subsets: train, val and test.
    
    Args:
     - label_dicts (list): is the label dictionary. User must shuffle before split.
     - split_ratio (list): this is a list consisting of the ratio between train, val, test.
       e.g. split_ratio = [8,1,1] denotes train:val:test = 8:1:1
    """
    
    # split dataset
    train_num = int(len(label_dicts) * (split_ratio[0]/sum(split_ratio)))
    val_num = int(len(label_dicts) * (split_ratio[1]/sum(split_ratio)))
    test_num = int(len(label_dicts) * (split_ratio[2]/sum(split_ratio)))
    print(f"set [train:val:test] to [{train_num}:{val_num}:{test_num}]")
    
    # initialise variables
    label_cat = {}
    label_cat['train'] = label_dicts[:train_num]
    label_cat['val'] = label_dicts[train_num:train_num+val_num]
    label_cat['test'] = label_dicts[train_num+val_num:]
    
    return  label_cat

def save_ext_dataset(dataset_name: str, data_path: str, label_cat: dict):
    r"""
    save dataset and label to its correponding folders.
    
    Args:
     - dataset_name (str): name of the dataset.
     - dataset_path (str): path of the dataset.
     - label_cat (dict): a dictionary consisting of the labels of train, val and test set. Get this with function split_dataset()
    """

    # iterate through different subset.
    for d in ['train', 'val', 'test']:
        
        # INITIALISATION for {d}
        # set up output dir for {d}
        sub_img_dir = os.path.join(data_path, dataset_name, d)
        shutil.rmtree(sub_img_dir, ignore_errors=True)
        os.makedirs(sub_img_dir, exist_ok=True)
        
        # set up label file for {d}
        sub_label_filenames = os.path.join(data_path, dataset_name, 'labels', f"labels_{d}.json")
        open(sub_label_filenames, 'w').close() # reset file
        
        # CREATE CORRESPONDING SET for {d}
        # construct label file
        with open(sub_label_filenames, 'a') as f:
            f.write('[')
                
        # extract training and validation imgs and labels from /images and labels.json
        for idx, v in enumerate(label_cat[d]):
            
            # copy training set from images to /train
            dst = os.path.join(sub_img_dir, os.path.basename(v["file_name"]))
            shutil.copyfile(v["file_name"], dst)
            
            # save dict to label_train.json
            with open(sub_label_filenames, 'a') as f:
                json.dump(v, f)
                if idx < len(label_cat[d]) - 1: # do not add , at the end of the 
                    f.write(',')
                    
        with open(sub_label_filenames, 'a') as f:
            f.write(']')
            f.close()
    
    
    
########### For Inferences ############
def random_color() -> list:
    return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

def get_inference_dicts(img_dir, extensions=None):
    if extensions is None:
        extensions = [".png", ".tif", "jpg"]
        
    dataset_dicts = []
    img_idx = 0
    for ext in extensions:
        for fname in os.listdir(img_dir):
            # check the file ends with the extension
            if fname.endswith(ext):
                img_filepath = os.path.join(img_dir, fname)
                record = {}
                img_h, img_w = cv2.imread(img_filepath).shape[:2]
                record["file_name"] = img_filepath
                record["image_id"] = img_idx
                record["height"] = img_h
                record["width"] = img_w
                
                dataset_dicts.append(record)
                img_idx += 1
    return dataset_dicts

def binary_mask_to_polygon(binary_mask, tolerance=0):
    r""" Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated polygonal chain. If tolerance is 0, the original coordinate array is returned.
    
    """
    def close_contour(contour):
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        return contour
    
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    for contour in contours:
        contour = close_contour(contour)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
#         segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    return polygons

def fit_polygens_to_rotated_bboxes(polygons):
    rbboxes = []
    for p in polygons:
        pts_x = p[::2]
        pts_y = p[1::2]
        pts = [[x, y] for x,y in zip(pts_x, pts_y)]
        pts = np.array(pts, np.float32)
        rect = cv2.minAreaRect(pts)  #  ((cx, cy), (w, h), a)
        rbboxes.append(rect)
    return rbboxes

def draw_polygons(img_filename, polygons, texts, thickness=1):
    img = cv2.imread(img_filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tl = thickness
    tf = max(tl-1, 1)
    for p, text in zip(polygons, texts):
        color = random_color()
        pts_x = p[::2]
        pts_y = p[1::2]
        pts = [[x, y] for x,y in zip(pts_x, pts_y)]
        pts = np.array(pts, np.int32)
        cv2.polylines(img, [pts], isClosed=True, thickness=thickness, color=color)
        # bounding box format: (tlx, tly, w, h)
        x, y, w, h = cv2.boundingRect(pts)
        cv2.rectangle(img, pt1=(x, y), pt2=(x+w, y+h), color=color, thickness=thickness, lineType=cv2.LINE_AA)
        t_size = cv2.getTextSize(text, 0, fontScale=tl/3, thickness=thickness)[0]
        c1 = (x, y)
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] -3
        cv2.rectangle(img, c1, c2, color=color, thickness=-1, lineType=cv2.LINE_AA)  # filled
        cv2.putText(img, text, (c1[0], c1[1]-2), 0, tl/3, [255,255,255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def draw_rotated_bboxes(img_filename, rboxes, texts, thickness=1, color=None):
    img = cv2.imread(img_filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = draw_rotated_bboxes_on_image(img, rboxes, texts, thickness, color)
    return img

def draw_rotated_bboxes_on_image(img, rboxes, texts, thickness=1, color=None):
    img_draw = img.copy()
    tl = thickness
    tf = max(tl-1, 1)
    for rb, text in zip(rboxes, texts):
        c = random_color() if color is None else color
        box = cv2.boxPoints(rb)
        box = np.int0(box)
        cv2.drawContours(img_draw, [box], 0, color=c, thickness=thickness)
        t_size = cv2.getTextSize(text, 0, fontScale=tl/3, thickness=thickness)[0]
        pt = np.amin(box, axis=0)
        c1 = (pt[0], pt[1])
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] -3
        cv2.rectangle(img_draw, c1, c2, color=color, thickness=-1, lineType=cv2.LINE_AA)  # filled
        cv2.putText(img_draw, text, (c1[0], c1[1]-2), 0, tl/3, [255,255,255], thickness=tf, lineType=cv2.LINE_AA)
    return img_draw


