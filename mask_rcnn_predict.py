import os
import sys
import numpy as np
import pydicom
from tqdm import tqdm
# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from utilities import get_dicom_fps, parse_dataset, DetectorDataset, find_lastest_model_and_save_config_class
ROOT_DIR = '.'
DATA_DIR = '../Input' # to be modified
sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  # To find local version of the library
test_dicom_dir = os.path.join(DATA_DIR, 'stage_2_test_images')
# Directory to save logs and trained model
ROOT_DIR = '.'
ORIG_SIZE = 1024

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# # The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"]="3"


#*************retreive mask RCNN******************
# retreive path to model and config class
model_path = 'pneumonia20181020T2012_0165/mask_rcnn_pneumonia_0032.h5'  # find_lastest_model(ROOT_DIR)
# ap_model_path = 'pneumonia20181021T0631_0165_AP/mask_rcnn_pneumonia_0032.h5'
# pa_model_path = 'pneumonia20181021T1543_0165_PA/mask_rcnn_pneumonia_0032.h5'
# retrieve DetectorConfig from pickle file

######################################################################
#model parameter setting for testing data
######################################################################
class InferenceConfig(Config):
    NAME = 'pneumonia'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BACKBONE = 'resnet50'
    NUM_CLASSES = 2  # background + 1 pneumonia classes
    IMAGE_MIN_DIM = 256  # 512
    IMAGE_MAX_DIM = 256  # 512
    # MEAN_PIXEL = np.array([0, 0, 0])#np.array([124.9, 124.9, 124.9])
    RPN_ANCHOR_SCALES = (32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 4
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.78
    DETECTION_NMS_THRESHOLD = 0.01
    # DETECTION_MIN_CONFIDENCE = 0.9

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode='inference',
                              config=inference_config,
                              model_dir=ROOT_DIR)
#
# Load trained weights (fill in path to trained weights here)
print('Retrieving mask RCNN...')
assert model_path != "", "Provide path to trained weights for mask RCNN"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# ap_model = modellib.MaskRCNN(mode='inference',
#                               config=inference_config,
#                               model_dir=ROOT_DIR)

# Load trained weights (fill in path to trained weights here)
# print('Retrieving AP mask RCNN...')
# assert ap_model_path != "", "Provide path to trained weights for mask RCNN"
# print("Loading weights from ", ap_model_path)
# ap_model.load_weights(ap_model_path, by_name=True)
#
# pa_model = modellib.MaskRCNN(mode='inference',
#                               config=inference_config,
#                               model_dir=ROOT_DIR)

# Load trained weights (fill in path to trained weights here)
# print('Retrieving PA mask RCNN...')
# assert pa_model_path != "", "Provide path to trained weights for mask RCNN"
# print("Loading weights from ", pa_model_path)
# pa_model.load_weights(pa_model_path, by_name=True)
# *************retreive image classifier******************
DROPOUT = 0.2
DENSE_COUNT = 128
image_classifier_path = 'classifier/ResNet50/384_384_lr_1e-4_balanced_orig_data_prtrained_imagenet_20epoch_all_layers/RESNET50_full_model.h5'
#LEARN_RATE = 1e-4
SHAPE = 2  # t_y.shape[1] in lung_opacity_classifier.py
print('Retrieving classifier...')
assert image_classifier_path != "", "Provide path to trained weights for classifier"
from keras.applications.resnet50 import ResNet50 as PTModel, preprocess_input as Resnet50_preprocess
base_pretrained_model = PTModel(input_shape=(384, 384, 3),
                                include_top=False)

from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, \
    LocallyConnected2D, Lambda, AvgPool2D
from keras.models import Model
from keras.optimizers import Adam
# catch the output as feature from picked pre-trained model
pt_features = Input(base_pretrained_model.get_output_shape_at(0)[1:], name='feature_input')
pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
# Normalize the activations of the previous layer at each batch,
# i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
from keras.layers import BatchNormalization
bn_features = BatchNormalization()(pt_features)
# Global average pooling operation for spatial data
gap = GlobalAveragePooling2D()(bn_features)
# Dropout layer
gap_dr = Dropout(DROPOUT)(gap)
# Another Dropout after a fully connected network after first dropout layer
dr_steps = Dropout(DROPOUT)(Dense(DENSE_COUNT, activation='elu')(gap_dr))
out_layer = Dense(SHAPE, activation='softmax')(dr_steps)
attn_model = Model(inputs=[pt_features],
                   outputs=[out_layer], name='trained_model')
from keras.models import Sequential
image_classifier = Sequential(name='combined_model')
image_classifier.add(base_pretrained_model)
image_classifier.add(attn_model)
#load pre-trained weights
print("Loading weights from ", image_classifier_path)
image_classifier.load_weights(image_classifier_path)

# *************retreive bbox classifier******************
DROPOUT = 0.2
DENSE_COUNT = 256
bbox_classifier_path = 'bbox_classifier/Xception_200_200_lr1e-4_256dense/Xception_full_model.h5'
#LEARN_RATE = 1e-4
SHAPE = 2  # t_y.shape[1] in lung_opacity_classifier.py
print('Retrieving classifier...')
assert bbox_classifier_path != "", "Provide path to trained weights for classifier"
from keras.applications.xception import Xception as PTModel, preprocess_input as Xception_preprocess
base_pretrained_model = PTModel(input_shape=(200, 200, 3),
                                include_top=False)

from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, \
    LocallyConnected2D, Lambda, AvgPool2D
from keras.models import Model
from keras.optimizers import Adam
# catch the output as feature from picked pre-trained model
pt_features = Input(base_pretrained_model.get_output_shape_at(0)[1:], name='feature_input')
pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
# Normalize the activations of the previous layer at each batch,
# i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
from keras.layers import BatchNormalization
bn_features = BatchNormalization()(pt_features)
# Global average pooling operation for spatial data
gap = GlobalAveragePooling2D()(bn_features)
# Dropout layer
gap_dr = Dropout(DROPOUT)(gap)
# Another Dropout after a fully connected network after first dropout layer
dr_steps = Dropout(DROPOUT)(Dense(DENSE_COUNT, activation='elu')(gap_dr))
out_layer = Dense(SHAPE, activation='softmax')(dr_steps)
attn_model = Model(inputs=[pt_features],
                   outputs=[out_layer], name='trained_model')
from keras.models import Sequential
bbox_classifier = Sequential(name='combined_model')
bbox_classifier.add(base_pretrained_model)
bbox_classifier.add(attn_model)
#load pre-trained weights
print("Loading weights from ", bbox_classifier_path)
bbox_classifier.load_weights(bbox_classifier_path)

# Get filenames of test dataset DICOM images
test_image_fps = get_dicom_fps(test_dicom_dir)

from scipy import misc
######################################################################
# Predict the result on testing data and output to submission.csv
######################################################################
# Make predictions on test images, write out sample submission
from keras.preprocessing import image as img
def predict(image_fps, filepath='submission.csv', min_conf=0.95):
    # assume square image
    resize_factor = ORIG_SIZE / inference_config.IMAGE_SHAPE[0]
    with open(filepath, 'w') as file:
        file.write("patientId,predictionString\n") #header
        for image_id in tqdm(image_fps):
            ds = pydicom.read_file(image_id)
            #view_position = ds.ViewPosition
            original_image = ds.pixel_array
            # If grayscale. Convert to RGB for consistency.
            if len(original_image.shape) != 3 or original_image.shape[2] != 3:
                original_image = np.stack((original_image,) * 3, -1)

            image_for_maskRCNN, window, scale, padding, crop = utils.resize_image(
                original_image,
                min_dim=inference_config.IMAGE_MIN_DIM,
                min_scale=inference_config.IMAGE_MIN_SCALE,
                max_dim=inference_config.IMAGE_MAX_DIM,
                mode=inference_config.IMAGE_RESIZE_MODE)

            image_for_classifier, window2, scale2, padding2, crop2 = utils.resize_image(
                original_image,
                min_dim=384,
                min_scale=0,
                max_dim=384,
                mode=inference_config.IMAGE_RESIZE_MODE)
            image_for_classifier = np.expand_dims(image_for_classifier.astype(np.float32), axis=0)
            image_for_classifier = Resnet50_preprocess(image_for_classifier)
            image_pred = image_classifier.predict(image_for_classifier) # [no opacity, opacity]

            patient_id = os.path.splitext(os.path.basename(image_id))[0]

            results = model.detect([image_for_maskRCNN])
            r = results[0]

            out_str = ""
            out_str += patient_id
            out_str += ","
            assert (len(r['rois']) == len(r['class_ids']) == len(r['scores']))
            if len(r['rois']) == 0:# no bbox detected
                pass
            else:
                if False: #pred[0][0] > 0.99: #classified as no opacity with high prob
                    pass
                else:
                    num_instances = len(r['rois'])
                    for i in range(num_instances):
                        if r['scores'][i] > min_conf:#threshold and view_position != 'PA':
                            #prediction on bbox
                            x = r['rois'][i][1]
                            y = r['rois'][i][0]
                            width = r['rois'][i][3] - x
                            height = r['rois'][i][2] - y

                            x1 = int(x * resize_factor)
                            y1 = int(y * resize_factor)
                            width1 = int((r['rois'][i][3] - x) * resize_factor)
                            height1 = int((r['rois'][i][2] - y) * resize_factor)

                            masked = np.array(original_image[y1:y1 + height1, x1:x1 + width1, :], dtype=np.int32)
                            resized_patch = misc.imresize(masked, (200, 200), 'nearest')
                            image = np.expand_dims(resized_patch.astype(np.float32), axis=0)
                            image = Xception_preprocess(image)
                            pred = bbox_classifier.predict(image)
                            # prob of false positive on bbox, confidence of bbox, prob of false positive on whole image
                            #if pred[0][0] > 0.95 and r['scores'][i] < 0.98 and image_pred[0][0] > 0.95: 0.13
                            #if pred[0][0] > 0.9 and r['scores'][i] < 0.97 and image_pred[0][0] > 0.9: 0132
                            if pred[0][0] > 0.9 and r['scores'][i] < 0.97 and image_pred[0][0] > 0.9:
                                continue
                            out_str += ' '
                            out_str += str(round(r['scores'][i], 2))
                            out_str += ' '
                            # x1, y1, width, height
                            bboxes_str = "{} {} {} {}".format(x * resize_factor, y * resize_factor, width * resize_factor, height * resize_factor)
                            #bboxes_str = "{} {} {} {}".format(x1, y1, width, height)
                            out_str += bboxes_str

            file.write(out_str + "\n")

#tag = model_path.split('/')[1]
submission_fp = os.path.join(ROOT_DIR, 'submission.csv')
#print(submission_fp)
predict(test_image_fps, filepath=submission_fp, min_conf=0.965)
