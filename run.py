import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# # The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import random
import numpy as np
import pydicom
import glob
from imgaug import augmenters as iaa
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
    NUM_EPOCHS = 24
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 2
    IMAGES_PER_GPU = 2 #8 for 512^2; 2 for 1024^2
    # number of images in training dataset: 25684
    # cross-validation ratio: 0.9
    # 25684 * 0.9 / bash size(GPU_COUNT*IMAGES_PER_GPU = 32)
    # ~= 722
    STEPS_PER_EPOCH = 400 #2400/(GPU_COUNT*IMAGES_PER_GPU) #2890 #720 for 512^2; 2890 for 1024^2
    #VALIDATION_STEPS = 36
    BACKBONE = 'resnet50'

    NUM_CLASSES = 2  # background + 1 pneumonia classes

    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    #MEAN_PIXEL = np.array([0, 0, 0])#np.array([124.9, 124.9, 124.9])
    RPN_ANCHOR_SCALES = (64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 4
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.78
    DETECTION_NMS_THRESHOLD = 0.01
    FINE_TUNING = '3 stages 0.008 2-0.004 8-0.0001 32 with augmentation'
    PRE_TRAINED = 'COCO'
    #USE_MINI_MASK = False
    LEARNING_RATE = 0.0008
    # LOSS_WEIGHTS = {
    #     "rpn_class_loss": 1.,
    #     "rpn_bbox_loss": 1.,
    #     "mrcnn_class_loss": 1.,
    #     "mrcnn_bbox_loss": 1.,
    #     "mrcnn_mask_loss": 1.
    # }
    TARGET_1_ONLY = False #5669 images
    VIEW_POSITION_MDOE = '' #['AP', 'PA', '']



config = DetectorConfig()
config.display()
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
#ds = pydicom.read_file(image_fps[0]) # read dicom image from filepath
#image = ds.pixel_array # get image array

#Split the data into training and validation datasets
######################################################################
# Modify this line to use more or fewer images for training/validation.
# To use all images, do: image_fps_list = list(image_fps)
image_fps_list = list(image_fps) #image_fps[:1000]
#####################################################################

# split dataset into training vs. validation dataset
# split ratio is set to 0.9 vs. 0.1 (train vs. validation, respectively)
sorted(image_fps_list)
random.seed(42)
random.shuffle(image_fps_list)

#validation_split = 0.1
split_index = 1500#int((1 - validation_split) * len(image_fps_list))
image_fps_val = image_fps_list[:split_index]
image_fps_train = image_fps_list[split_index:]
#print(len(image_fps_train), len(image_fps_val))

# prepare the training dataset
dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()
# prepare the validation dataset
dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_val.prepare()

#Create Mask R-CNN Model
model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)

#Image augmentation (light but constant)
augmentation = iaa.Sequential([
    iaa.OneOf([ ## geometric transform
        iaa.Affine(
            scale={"x": (0.98, 1.02), "y": (0.98, 1.04)},
            translate_percent={"x": (-0.02, 0.02), "y": (-0.04, 0.04)},
            #rotate=(-2, 2),
            shear=(-1, 1),
        ),
        iaa.PiecewiseAffine(scale=(0.001, 0.025)),
    ]),
    iaa.OneOf([ ## brightness or contrast
        iaa.Multiply((0.9, 1.1)),
        iaa.ContrastNormalization((0.9, 1.1)),
    ]),
    iaa.OneOf([ ## blur or sharpen
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
        iaa.Sharpen(alpha=(0.0, 0.1)),
    ]),
])

#load pre-trained weight in prior for transferring learning
# Exclude the last layers because they require a matching
# number of classes
#model.load_weights('pneumonia20181020T0615_0163/mask_rcnn_pneumonia_0032.h5', by_name=True)
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=[
     "mrcnn_class_logits", "mrcnn_bbox_fc",
     "mrcnn_bbox", "mrcnn_mask"]) # at model line 2103

######################################################################
# Train Mask-RCNN Model
######################################################################
import warnings
warnings.filterwarnings("ignore")
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=config.NUM_EPOCHS,
#            layers='all') #at model.py line 2284 for more fine-tuning detail

#Training - Stage 1
print("Training network heads")
model.train(dataset_train, dataset_val,
            learning_rate=0.008,
            epochs=2,
            layers='heads')
#
# # Training - Stage 2
# # Finetune layers from ResNet stage 4 and up
# print("Fine tune Resnet stage 4 and up")
# model.train(dataset_train, dataset_val,
#             learning_rate=0.004,
#             epochs=6,
#             layers='4+')

# Training - Stage 3
# # Fine tune all layers
# print("Fine tune all layers")
model.train(dataset_train, dataset_val,
            learning_rate=0.004,
            epochs=6,
            layers='all',
            augmentation=augmentation
            )

model.train(dataset_train, dataset_val,
            learning_rate=0.0001,
            epochs=32,
            layers='all',
            augmentation=augmentation
            )

# model.train(dataset_train, dataset_val,
#             learning_rate=0.00001,
#             epochs=56,
#             layers='all',
#             augmentation=augmentation
#             )
# model.train(dataset_train, dataset_val,
#             learning_rate=0.00008,
#             epochs=16,
#             layers='all',
#             augmentation=augmentation
#             )
######################################################################
# find and select trained model
######################################################################
model_path = find_lastest_model_and_save_config_class(model.model_dir, config)
print('Found model {}'.format(model_path))

######################################################################
#model parameter setting for testing data
######################################################################
class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    #DETECTION_MAX_INSTANCES = 4
    #DETECTION_MIN_CONFIDENCE = 0.9

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode='inference',
                          config=inference_config,
                          model_dir=ROOT_DIR)

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Get filenames of test dataset DICOM images
test_image_fps = get_dicom_fps(test_dicom_dir)

######################################################################
# Predict the result on testing data and output to submission.csv
######################################################################
# Make predictions on test images, write out sample submission
def predict(inference_model, image_fps, filepath='submission.csv', min_conf=0.95):
    # assume square image
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
    # resize_factor = ORIG_SIZE
    with open(filepath, 'w') as file:
        file.write("patientId,predictionString\n") #header
        for image_id in tqdm(image_fps):
            ds = pydicom.read_file(image_id)
            image = ds.pixel_array
            # If grayscale. Convert to RGB for consistency.
            if len(image.shape) != 3 or image.shape[2] != 3:
                image = np.stack((image,) * 3, -1)
            image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)

            patient_id = os.path.splitext(os.path.basename(image_id))[0]

            results = inference_model.detect([image])
            r = results[0]

            out_str = ""
            out_str += patient_id
            out_str += ","
            assert (len(r['rois']) == len(r['class_ids']) == len(r['scores']))
            if len(r['rois']) == 0:
                pass
            else:
                num_instances = len(r['rois'])

                for i in range(num_instances):
                    if r['scores'][i] > min_conf:
                        out_str += ' '
                        out_str += str(round(r['scores'][i], 2))
                        out_str += ' '

                        # x1, y1, width, height
                        x1 = r['rois'][i][1]
                        y1 = r['rois'][i][0]
                        width = r['rois'][i][3] - x1
                        height = r['rois'][i][2] - y1
                        bboxes_str = "{} {} {} {}".format(x1 * resize_factor, y1 * resize_factor, width * resize_factor, height * resize_factor)
                        #bboxes_str = "{} {} {} {}".format(x1, y1, width, height)
                        out_str += bboxes_str

            file.write(out_str + "\n")

# predict only the first 50 entries
if DetectorConfig.VIEW_POSITION_MDOE:
    submission_fp = os.path.join(ROOT_DIR, DetectorConfig.VIEW_POSITION_MDOE + '_submission.csv')
else:
    submission_fp = os.path.join(ROOT_DIR, 'submission.csv')
#print(submission_fp)
predict(model, test_image_fps, filepath=submission_fp)

#email alert notification to be finished.