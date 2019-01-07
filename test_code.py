import os
import sys
import random
import numpy as np
import pydicom
import glob
#from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd

DATA_DIR = '../Input' # to be modified

# Directory to save logs and trained model
ROOT_DIR = '.'

# Import Mask RCNN
sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

train_dicom_dir = os.path.join(DATA_DIR, 'stage_1_train_images')
test_dicom_dir = os.path.join(DATA_DIR, 'stage_1_test_images')

#Some setup functions and classes for Mask-RCNN (move to utilities.py)
from utilities import DetectorDataset, find_lastest_model_and_save_config_class
#get_dicom_fps
#parse_dataset

######################################################################
#model parameters setting for tranning data
######################################################################
# Original DICOM image size: 1024 x 1024
ORIG_SIZE = 1024
#NUM_EPOCHS = 10
# The following parameters have been selected to reduce running time for demonstration purposes
# These are not optimal
class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """

    # Give the configuration a recognizable name
    NAME = 'pneumonia'
    NUM_EPOCHS = 8
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 2
    IMAGES_PER_GPU = 8 #8 for 512^2; 2 for 1024^2
    # number of images in training dataset: 25684
    # cross-validation ratio: 0.9
    # 25684 * 0.9 / bash size(GPU_COUNT*IMAGES_PER_GPU = 32)
    # ~= 722
    STEPS_PER_EPOCH = 1200 #2400/(GPU_COUNT*IMAGES_PER_GPU) #2890 #720 for 512^2; 2890 for 1024^2
    VALIDATION_STEPS = 80
    BACKBONE = 'resnet101'

    NUM_CLASSES = 2  # background + 1 pneumonia classes

    IMAGE_MIN_DIM = 512 #1024
    IMAGE_MAX_DIM = 512
    #MEAN_PIXEL = np.array([0, 0, 0])#np.array([124.9, 124.9, 124.9])
    RPN_ANCHOR_SCALES = (32, 64, 128, 256)

    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 4
    DETECTION_MAX_INSTANCES = 4
    DETECTION_MIN_CONFIDENCE = 0.9
    DETECTION_NMS_THRESHOLD = 0.3
    #FINE_TUNING = '3 stages 1-2-2'
    PRE_TRAINED = 'pneumonia20181013T1954'
    USE_MINI_MASK = False
    LEARNING_RATE = 0.0001
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }
    TARGET_1_ONLY = True  # 5669 images
    VIEW_POSITION_MDOE = ''  # ['AP', 'PA', '']


config = DetectorConfig()
#config.display()
candidates = set()
# Filter view positions (if it's set to)
if DetectorConfig.VIEW_POSITION_MDOE or DetectorConfig.TARGET_1_ONLY:
    image_bbox_df = pd.read_csv('../Input/image_bbox_full.csv')
    image_df = image_bbox_df.groupby('patientId').apply(lambda x: x.sample(1))
    if DetectorConfig.TARGET_1_ONLY:
        print('Select images with pneumonia only.')
        image_df = image_df[image_df.Target == 1]
    if DetectorConfig.VIEW_POSITION_MDOE:
        print('Select images in view position-' + DetectorConfig.VIEW_POSITION_MDOE + ' only.')
        image_df = image_df[image_df.ViewPosition == DetectorConfig.VIEW_POSITION_MDOE]
    candidates = set(image_df.patientId)

#Some setup functions and classes for Mask-RCNN
def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir+'/'+'*.dcm')
    return list(set(dicom_fps))

def parse_dataset(dicom_dir, anns):
    image_fps = get_dicom_fps(dicom_dir)
    if DetectorConfig.VIEW_POSITION_MDOE or DetectorConfig.TARGET_1_ONLY:
        def f(i):
            return i.split('/')[3].split('.')[0] in candidates
        image_fps = list(filter(f ,image_fps))
        anns = anns[anns.patientId.isin(candidates)] #********************
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows():
        fp = os.path.join(dicom_dir, row['patientId']+'.dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations

# training dataset
anns = pd.read_csv(os.path.join(DATA_DIR, 'stage_1_train_labels.csv'))
#anns.head()
image_fps, image_annotations = parse_dataset(train_dicom_dir, anns=anns)

print(len(image_annotations))
print(len([1 for key in image_annotations if image_annotations[key][0].Target == 1]))