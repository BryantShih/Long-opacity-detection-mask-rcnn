import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# # The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import time
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd
import glob


DATA_DIR = '../Input' # to be modified
# Directory to save logs and trained model
ROOT_DIR = '.'
ORIG_SIZE = 1024

sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

train_dicom_dir = os.path.join(DATA_DIR, 'stage_1_train_images')
test_dicom_dir = os.path.join(DATA_DIR, 'stage_1_test_images')

#Some setup functions and classes for Mask-RCNN
def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir+'/'+'*.dcm')
    return list(set(dicom_fps))

def parse_dataset(dicom_dir, anns):
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows():
        fp = os.path.join(dicom_dir, row['patientId']+'.dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations

# If grayscale. Convert to RGB for consistency.

#Image augmentation
#to form 3 channels
class DetectorDataset(utils.Dataset):
    """Dataset class for training pneumonia detection on the RSNA pneumonia dataset.
    """
    augmentation_channel_1 = iaa.Sequential([iaa.Sharpen(alpha=0.5, lightness=2.0)])

    augmentation_channel_2 = iaa.Sequential([iaa.ContrastNormalization((2, 1))])

    #MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes", "Fliplr", "Flipud", "CropAndPad", "Affine", "PiecewiseAffine"]

    augmentation_mode = False

    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)

        # Add classes
        self.add_class('pneumonia', 1, 'Lung Opacity')

        # add images
        for i, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image('pneumonia', image_id=i, path=fp,
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        ds = pydicom.read_file(fp)
        image = ds.pixel_array

        if len(image.shape) != 3 or image.shape[2] != 3:

            if self.augmentation_mode:
            #add augmentation as 2 of 3 channels
            # Store shapes before augmentation to compare
                image_shape = image.shape
                channel_2 = self.augmentation_channel_1.augment_image(image)
                channel_3 = self.augmentation_channel_2.augment_image(image)
                image = np.stack((image, channel_2, channel_3), -1)
            else:
                image = np.stack((image,)*3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            if self.augmentation_mode:
                def hook(images, augmenter, parents, default):
                    """Determines which augmenters to apply to masks."""
                    return augmenter.__class__.__name__ in self.MASK_AUGMENTERS
            for i, a in enumerate(annotations):
                if a['Target'] == 1:
                    x = int(a['x'])
                    y = int(a['y'])
                    w = int(a['width'])
                    h = int(a['height'])
                    mask_instance = mask[:, :, i].copy()
                    cv2.rectangle(mask_instance, (x, y), (x + w, y + h), 255, -1)
                    mask[:, :, i] = mask_instance
                    class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)

def build_training_data_to_pkl(output = 'training_dataset.pkl'):
    # training dataset
    anns = pd.read_csv(os.path.join(DATA_DIR, 'stage_1_train_labels.csv'))
    # anns.head()
    image_fps, image_annotations = parse_dataset(train_dicom_dir, anns=anns)
    dataset_train = DetectorDataset(image_fps, image_annotations, ORIG_SIZE, ORIG_SIZE)
    dataset_train.prepare()
    import pickle
    f = open(output, 'wb')
    print('Dumping training dataset to ' + output + '...')
    pickle.dump(dataset_train, f)
    print('Done.')

def read_training_data_from_pkl(input = 'training_dataset.pkl'):
    import pickle
    f = open(input, 'rb')
    print('Reading training dataset from ' + input)
    data = pickle.load(f)
    return data

def find_lastest_model_and_save_config_class(root_dir, config):
    dir_names = next(os.walk(root_dir))[1]  # all directories under root_dir
    if not dir_names:
        import errno

        raise FileNotFoundError(
            errno.ENOENT,
            "Could not find model directory under {}".format(root_dir))
    key = config.NAME.lower()
    dir_names = filter(lambda f: f.startswith(key), dir_names)
    dir_names = sorted(dir_names)

    fps = []
    # Pick last directory, which includes th lasted trained model
    for d in dir_names:
        dir_name = os.path.join(root_dir, d)
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            print('No weight files in {}'.format(dir_name))
        else:
            checkpoint = os.path.join(dir_name, checkpoints[-1])
            fps.append(checkpoint)

    model_path = sorted(fps)[-1] #model and config should in the same folder
    config_path = os.path.join(*(model_path.split('/')[:-1]))

    print('Saving config setting to DetectorConfig.txt.')
    with open(os.path.join(config_path, 'DetectorConfig.txt'), 'w') as f:
        for i in (i for i in dir(config) if (not callable(i) and '__' not in i and i != 'display')):
            f.write("{:<30}:{} \n".format(i, config.__getattribute__(i)))
    return model_path

def find_lastest_model(root_dir):
    dir_names = next(os.walk(root_dir))[1]  # all directories under root_dir
    if not dir_names:
        import errno

        raise FileNotFoundError(
            errno.ENOENT,
            "Could not find model directory under {}".format(root_dir))
    key = 'pneumonia'
    dir_names = filter(lambda f: f.startswith(key), dir_names)
    dir_names = sorted(dir_names) #all directories with name starting with 'pneumonia' and sorted by time stamp
    fps = []
    # Pick last directory, which includes th lasted trained model
    for d in dir_names:
        dir_name = os.path.join(root_dir, d)
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints) #all files with name starting with 'mask_rcnn' and sorted by order
        if not checkpoints:
            print('No weight files in {}'.format(dir_name))
        else:
            checkpoint = os.path.join(dir_name, checkpoints[-1]) #first one: 0; latest one: -1
            fps.append(checkpoint)

    model_path = sorted(fps)[-1]
    return model_path

def count_falsePositiveCase(bbox_conf_threshold = 0.95):
    # retreive path to model and config class
    model_path = 'pneumonia20181020T2012_0165/mask_rcnn_pneumonia_0032.h5'  # find_lastest_model(ROOT_DIR)
    # retrieve DetectorConfig from pickle file
    bbox_conf_threshold = 0.95

    class inference_config(Config):
        NAME = 'pneumonia'
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        BACKBONE = "resnet50"  # 'resnet50'
        NUM_CLASSES = 2
        IMAGE_MAX_DIM = 256
        RPN_ANCHOR_SCALES = (32, 64, 128, 256)
        TRAIN_ROIS_PER_IMAGE = 32
        MAX_GT_INSTANCES = 4
        DETECTION_MAX_INSTANCES = 3
        DETECTION_MIN_CONFIDENCE = 0.78
        DETECTION_NMS_THRESHOLD = 0.01

    inference_config = inference_config()
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode='inference',
                              config=inference_config,
                              model_dir=ROOT_DIR)

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    dataset = read_training_data_from_pkl()
    count_higher_than_threshold = 0
    count = 0
    wanted = []
    for image_id in dataset.image_ids:
        print(image_id)
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, inference_config,
                                   image_id, use_mini_mask=False)
        #print('Ground truth: ',gt_bbox)
        if len(gt_bbox) != 0:  # !=0 to skip normal; ==0 to skip lung opacity
            continue
        results = model.detect([original_image])  # , verbose=1)
        r = results[0]
        #print("Predition: ",r['rois'], r['class_ids'], r['scores'])
        # check if confidence of bbox is higher then threshold 0.95
        #print('# of bbox: ', len(r['scores']))
        if len(r['scores']):
            count += 1
        index2remove = r['scores'] <= bbox_conf_threshold
        if len(index2remove[index2remove == True]) == len(r['scores']):
            continue
        count_higher_than_threshold += (len(r['scores']) - len(index2remove[index2remove == True]))
        wanted.append((image_id))
    print('# of false positive higher than threshold: ' + str(count_higher_than_threshold))
    print('# of false positive: ' + str(count))
    print('List of image id')
    print(wanted)

def visualize_and_comparison(bbox_conf_threshold = 0.95):
    #retreive path to model and config class
    model_path = 'pneumonia20181020T2012_0165/mask_rcnn_pneumonia_0032.h5' #find_lastest_model(ROOT_DIR)
    #retrieve DetectorConfig from pickle file

    class inference_config(Config):
        NAME = 'pneumonia'
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        BACKBONE = "resnet50" #'resnet50'
        NUM_CLASSES = 2
        IMAGE_MAX_DIM = 256
        RPN_ANCHOR_SCALES = (32, 64, 128, 256)
        TRAIN_ROIS_PER_IMAGE = 32
        MAX_GT_INSTANCES = 4
        DETECTION_MAX_INSTANCES = 3
        DETECTION_MIN_CONFIDENCE = 0.78
        DETECTION_NMS_THRESHOLD = 0.01

    inference_config = inference_config()
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode='inference',
                              config=inference_config,
                              model_dir=ROOT_DIR)

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # set color for class
    def get_colors_for_class_ids(class_ids):
        colors = []
        for class_id in class_ids:
            if class_id == 1:
                colors.append((.941, .204, .204))
        return colors

    # Show few example of ground truth vs. predictions on the validation dataset
    dataset = read_training_data_from_pkl('test_dataset.pkl')
    # image_id = random.choice(dataset.image_ids)
    # original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
    #     modellib.load_image_gt(dataset, inference_config,
    #                             image_id, use_mini_mask=False)
    # results = model.detect([original_image])  # , verbose=1)
    # r = results[0]
    # visualize.display_differences(original_image,
    #                               gt_bbox, gt_class_id, gt_mask,
    #                               r['rois'], r['class_ids'], r['scores'], r['masks'],
    #                               dataset.class_names, title="Comparison (left: GT; right: prediction)")
    #*******************To be done********************
    #*******************Add classifier to see the result***************
    fig = plt.figure(figsize=(10, 30))

    i = -1
    #temp = [15731, 15732, 15735, 15738, 15743, 15748, 15752, 15757, 15759, 15764]
    while i < 3:
        image_id = random.choice(dataset.image_ids)#dataset.image_ids)
        print(image_id)
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, inference_config,
                                   image_id, use_mini_mask=False)

        # if len(gt_bbox) != 0: # !=0 to skip normal; ==0 to skip lung opacity
        #     continue
        results = model.detect([original_image])  # , verbose=1)
        r = results[0]
        # check if confidence of bbox is higher then threshold 0.95
        index2remove = r['scores'] <= bbox_conf_threshold
        if len(index2remove[index2remove == True]) == len(r['scores']):
            continue
        i += 1
        index2keep = index2remove==False
        r['rois'] = r['rois'][index2keep]
        #print(r['masks'].shape)
        r['masks'] = np.array([[ele[index2keep] for ele in row] for row in r['masks']])
        #print(r['masks'].shape)
        r['class_ids'] = r['class_ids'][index2keep]
        r['scores'] = r['scores'][index2keep]

        print('image size: ', original_image.shape)
        print('Image path: ', dataset.image_reference(image_id))
        print('Bounding Box: ', gt_bbox)
        plt.subplot(2, 4, 2 * i + 1) #
        visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dataset.class_names,
                                    colors=get_colors_for_class_ids(gt_class_id), ax=fig.axes[-1])
        plt.subplot(2, 4, 2 * i + 2)

        visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], dataset.class_names,
                                    r['scores'],
                                    colors=get_colors_for_class_ids(r['class_ids']), ax=fig.axes[-1])
    plt.show()
def visualizd_falsePositiveCase_from_pkl(input='classifier_threshold_analysis.pkl'):
    import pickle
    f = open(input, 'rb')
    print('Reading training dataset from ' + input)
    pp, fp = pickle.load(f)
    plt.style.use('seaborn-deep')
    bins = np.linspace(0, 1, 100)
    plt.hist([pp, fp], bins, label=['tp','fp'])
    plt.legend(loc='upper right')
    plt.show()


def count_falsePositiveCase_to_pkl(bbox_conf_threshold = 0.95):
    #*************retreive mask RCNN******************
    # retreive path to model and config class
    model_path = 'pneumonia20181020T2012_0165/mask_rcnn_pneumonia_0032.h5'  # find_lastest_model(ROOT_DIR)
    # retrieve DetectorConfig from pickle file
    bbox_conf_threshold = 0.95

    class inference_config(Config):
        NAME = 'pneumonia'
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        BACKBONE = "resnet50"  # 'resnet50'
        NUM_CLASSES = 2
        IMAGE_MAX_DIM = 256
        IMAGE_MIN_DIM = 256
        RPN_ANCHOR_SCALES = (32, 64, 128, 256)
        TRAIN_ROIS_PER_IMAGE = 32
        MAX_GT_INSTANCES = 4
        DETECTION_MAX_INSTANCES = 3
        DETECTION_MIN_CONFIDENCE = 0.78
        DETECTION_NMS_THRESHOLD = 0.01

    inference_config = inference_config()
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode='inference',
                              config=inference_config,
                              model_dir=ROOT_DIR)

    resize_factor = ORIG_SIZE / inference_config.IMAGE_SHAPE[0]
    # Load trained weights (fill in path to trained weights here)
    print('Retrieving mask RCNN...')
    assert model_path != "", "Provide path to trained weights for mask RCNN"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # *************retreive classifier******************
    DROPOUT = 0.2
    DENSE_COUNT = 128
    classifier_path = 'bbox_classifier/Xception_200_200_lr1e-4/Xception_full_model.h5'
    #LEARN_RATE = 1e-4
    SHAPE = 2  # t_y.shape[1] in lung_opacity_classifier.py
    print('Retrieving bbox classifier...')
    assert classifier_path != "", "Provide path to trained weights for bbox classifier"
    from keras.applications.xception import Xception as PTModel, preprocess_input
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
    from keras.optimizers import Adam
    classifier = Sequential(name='combined_model')
    classifier.add(base_pretrained_model)
    classifier.add(attn_model)
    # load pre-trained weights
    print("Loading weights from ", classifier_path)
    classifier.load_weights(classifier_path)

    dataset = read_training_data_from_pkl()
    tp = []
    fp = []
    from scipy import misc
    for image_id in dataset.image_ids:
        #image_id = random.choice(dataset.image_ids)
        print(image_id)
        #load_image_gt() resizes image according to config already
        image_256, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, inference_config,
                                   image_id, use_mini_mask=False)
        results = model.detect([image_256])  # , verbose=1)
        r = results[0]
        #find positive only: goal-to find classifier threshold to reduce false positive
        index2remove = r['scores'] <= bbox_conf_threshold
        if len(index2remove[index2remove == True]) == len(r['scores']):
            continue #no bbox
        #resize for classifier (may be changed to the same size later)
        original_image = dataset.load_image(image_id)
        num_instances = len(r['rois'])
        for i in range(num_instances):
            if r['scores'][i] > bbox_conf_threshold:  # threshold and view_position != 'PA':
                # prediction on bbox
                x = r['rois'][i][1]
                y = r['rois'][i][0]
                x1 = int(x * resize_factor)
                y1 = int(y * resize_factor)
                width1 = int((r['rois'][i][3] - x) * resize_factor)
                height1 = int((r['rois'][i][2] - y) * resize_factor)
                masked = np.array(original_image[y1:y1 + height1, x1:x1 + width1, :], dtype=np.int32)
                resized_patch = misc.imresize(masked, (200, 200), 'nearest')
                #print(resized_patch.shape)
                #print(resized_patch)
                image = np.expand_dims(resized_patch.astype(np.float32), axis=0)
                image = preprocess_input(image)
                #image = preprocess_input(image)
                pred = classifier.predict(image) # [bbox as FP, bbox as TP]
                #print('GT: ', ('Opacity' if len(gt_bbox) > 0 else 'No opacity'))
                #print('Prediction: ', ('No opacity' if pred[0][0] > pred[0][1] else 'Opacity'))
                print(pred)
                if len(gt_bbox): #pred[0][0] probability of being false positive bbox
                    tp.append(pred[0][0])
                else: #fp
                    fp.append(pred[0][0])
    print('tp______________')
    print(len(tp))
    print('fp______________')
    print(len(fp))
    #save result to pkl
    import pickle
    output = 'classifier_threshold_analysis_bbox_classifier.pkl'
    f = open(output, 'wb')
    print('Dumping data to ' + output + '...')
    pickle.dump((tp, fp), f)
    print('Done.')

def visualizd_falseNegativeCase_from_pkl(input='classifier_threshold_analysis_n.pkl'):
    import pickle
    f = open(input, 'rb')
    print('Reading training dataset from ' + input)
    pn, fn = pickle.load(f)
    plt.style.use('seaborn-deep')
    bins = np.linspace(0, 1, 100)
    plt.hist([pn, fn], bins, label=['tn','fn'])
    plt.legend(loc='upper right')
    plt.show()

def count_falseNegativeCase_to_pkl(bbox_conf_threshold = 0.95):
    #*************retreive mask RCNN******************
    # retreive path to model and config class
    model_path = 'pneumonia20181020T2012_0165/mask_rcnn_pneumonia_0032.h5'  # find_lastest_model(ROOT_DIR)
    # retrieve DetectorConfig from pickle file
    bbox_conf_threshold = 0.95

    class inference_config(Config):
        NAME = 'pneumonia'
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        BACKBONE = "resnet50"  # 'resnet50'
        NUM_CLASSES = 2
        IMAGE_MAX_DIM = 256
        IMAGE_MIN_DIM = 256
        RPN_ANCHOR_SCALES = (32, 64, 128, 256)
        TRAIN_ROIS_PER_IMAGE = 32
        MAX_GT_INSTANCES = 4
        DETECTION_MAX_INSTANCES = 3
        DETECTION_MIN_CONFIDENCE = 0.78
        DETECTION_NMS_THRESHOLD = 0.01

    inference_config = inference_config()
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode='inference',
                              config=inference_config,
                              model_dir=ROOT_DIR)

    # Load trained weights (fill in path to trained weights here)
    print('Retrieving mask RCNN...')
    assert model_path != "", "Provide path to trained weights for mask RCNN"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # *************retreive classifier******************
    DROPOUT = 0.2
    DENSE_COUNT = 128
    classifier_path = 'classifier/ResNet50/384_384_lr_1e-4_balanced_orig_data_prtrained_imagenet_20epoch_all_layers/RESNET50_full_model.h5'
    #LEARN_RATE = 1e-4
    SHAPE = 2  # t_y.shape[1] in lung_opacity_classifier.py
    print('Retrieving classifier...')
    assert classifier_path != "", "Provide path to trained weights for classifier"
    from keras.applications.resnet50 import ResNet50 as PTModel, preprocess_input
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
    from keras.optimizers import Adam
    classifier = Sequential(name='combined_model')
    classifier.add(base_pretrained_model)
    classifier.add(attn_model)
    # load pre-trained weights
    print("Loading weights from ", classifier_path)
    classifier.load_weights(classifier_path)

    dataset = read_training_data_from_pkl()
    pn = []
    fn = []
    for image_id in dataset.image_ids:
        print(image_id)
        #load_image_gt() resizes image according to config already
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, inference_config,
                                   image_id, use_mini_mask=False)
        results = model.detect([original_image])  # , verbose=1)
        r = results[0]
        #find negative only: goal-to find classifier threshold to reduce false negative
        index2keep = r['scores'] > bbox_conf_threshold
        if len(index2keep[index2keep == True]) > 0:
            continue #has bbox
        #resize for classifier (may be changed to the same size later)
        image = dataset.load_image(image_id)
        image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=384,
            min_scale=0,
            max_dim=384,
            mode=inference_config.IMAGE_RESIZE_MODE)
        image = np.expand_dims(image, axis=0)
        pred = classifier.predict(image) #[no opacity, opacity]
        #print('GT: ', ('Opacity' if len(gt_bbox) > 0 else 'No opacity'))
        #print('Prediction: ', ('No opacity' if pred[0][0] > pred[0][1] else 'Opacity'))
        #print(pred)
        if len(gt_bbox) == 0: #pn
            pn.append(pred[0][1])
        else: #fn
            fn.append(pred[0][1])
    print('tn______________')
    print(len(pn))
    print('fn______________')
    print(len(fn))
    #save result to pkl
    import pickle
    output = 'classifier_threshold_analysis_n.pkl'
    f = open(output, 'wb')
    print('Dumping data to ' + output + '...')
    pickle.dump((pn, fn), f)
    print('Done.')

def calculate_mean_pixel():

    dataset = read_training_data_from_pkl()
    def image_pixel_generator():
        for id in dataset.image_ids:
            yield dataset.load_image(id)
    result = 0
    for image in image_pixel_generator():
        result += np.mean(image)
    print('Mean pixel of training dataset:', result / len(dataset.image_ids))

def show_augmentation():
    dataset = read_training_data_from_pkl()

    class inference_config(Config):
        NAME = 'pneumonia'
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        BACKBONE = "resnet101" #'resnet50'
        NUM_CLASSES = 2
        IMAGE_MAX_DIM = 1024
        RPN_ANCHOR_SCALES = (32, 64, 128, 256)
        TRAIN_ROIS_PER_IMAGE = 32
        MAX_GT_INSTANCES = 3
        DETECTION_MAX_INSTANCES = 4
        DETECTION_MIN_CONFIDENCE = 0.9
        DETECTION_NMS_THRESHOLD = 0.1

    fig = plt.figure(figsize=(10, 30))

    i = -1
    while i < 2:
        image_id = random.choice(dataset.image_ids)

        original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, inference_config,
                                   image_id, use_mini_mask=False)

        if len(gt_bbox) != 0:  # 0 to skip normal; !=0 to skip lung opacity
            continue
        else:
            i += 1
        print('image size: ', original_image.shape)
        print('Image path: ', dataset.image_reference(image_id))
        print('Bounding Box: ', gt_bbox)
        plt.subplot(3, 3, 3 * i + 1) #orignal image
        plt.imshow(original_image)
        plt.subplot(3, 3, 3 * i + 2) #augmentation 1
        channel_1 = DetectorDataset.augmentation_channel_1.augment_image(original_image)
        plt.imshow(channel_1)
        plt.subplot(3, 3, 3 * i + 3) #augmentation 2
        channel_2 = DetectorDataset.augmentation_channel_2.augment_image(original_image)
        plt.imshow(channel_2)
    plt.show()

def extractBboxFromTrainingData(bbox_conf_threshold = 0.9, tp_dir='bbox_patch_data/TP_bbox', fp_dir='../Input/bbox_patch_data/FP_bbox'):
    # To extract ground truth bbox patches and false positive patches bbox
    # to train a post-classifier on bbox detected by mask-rcnn
    # save image im original patched size (to be resized when training)

    # retreive path to model and config class
    model_path = 'pneumonia20181020T2012_0165/mask_rcnn_pneumonia_0032.h5'  # find_lastest_model(ROOT_DIR)
    # retrieve DetectorConfig from pickle file
    bbox_conf_threshold = 0.95

    class inference_config(Config):
        NAME = 'pneumonia'
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        BACKBONE = "resnet50"  # 'resnet50'
        NUM_CLASSES = 2
        IMAGE_MAX_DIM = 256
        IMAGE_MIN_DIM = 256
        RPN_ANCHOR_SCALES = (32, 64, 128, 256)
        TRAIN_ROIS_PER_IMAGE = 32
        MAX_GT_INSTANCES = 4
        DETECTION_MAX_INSTANCES = 3
        DETECTION_MIN_CONFIDENCE = 0.78
        DETECTION_NMS_THRESHOLD = 0.01

    inference_config = inference_config()
    config = Config()
    config.IMAGE_MIN_DIM = ORIG_SIZE
    config.IMAGE_MAX_DIM = ORIG_SIZE
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode='inference',
                              config=inference_config,
                              model_dir=ROOT_DIR)

    resize_factor = ORIG_SIZE / inference_config.IMAGE_SHAPE[0]
    # Load trained weights (fill in path to trained weights here)
    print('Retrieving mask RCNN...')
    assert model_path != "", "Provide path to trained weights for mask RCNN"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    #read data from pkl file
    dataset = read_training_data_from_pkl()
    from scipy import misc

    for image_id in dataset.image_ids:
        print(image_id)
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        resize_image, resize_image_meta, resize_gt_class_id, resize_gt_bbox, resize_gt_mask = \
            modellib.load_image_gt(dataset, inference_config,
                                   image_id, use_mini_mask=False)
        # extract patch from ground truth
        if len(gt_bbox):
            continue
            # num_instances = len(gt_bbox)
            # for i in range(num_instances):
            #     x1 = gt_bbox[i][1] #top left
            #     y1 = gt_bbox[i][0]
            #     width = gt_bbox[i][3] - x1 #bottom righ
            #     height = gt_bbox[i][2] - y1
            #     masked = np.zeros((height, width))
            #     #print(x1, y1, width, height)
            #     masked = np.array(original_image[y1:y1+height, x1:x1+width, 0], dtype=np.int32)
            #     file_name = os.path.join(tp_dir, str(image_id) + '_' + str(i) + '_GT.png')
            #     misc.imsave(file_name, masked)
        else:
            # extract patch from false postive result
            results = model.detect([resize_image])  # , verbose=1)
            r = results[0]
            index2remove = r['scores'] <= bbox_conf_threshold
            if len(index2remove[index2remove == True]) == len(r['scores']):
                 continue  # no bbox in prediction
            num_instances = len(r['rois'])
            for i in range(num_instances):
                if r['scores'][i] > bbox_conf_threshold:
                    x1 = r['rois'][i][1]
                    y1 = r['rois'][i][0]
                    width = int((r['rois'][i][3] - x1) * resize_factor)
                    height = int((r['rois'][i][2] - y1) * resize_factor)
                    x1 = int(x1 * resize_factor)
                    y1 = int(y1 * resize_factor)
                    masked = np.array(original_image[y1:y1+height, x1:x1+width, 0], dtype=np.int32)
                    file_name = os.path.join(fp_dir, str(image_id) + '_' + str(i) + '_FP.png')
                    misc.imsave(file_name, masked)


if __name__ =='__main__':
    #testing
    #calculate_mean_pixel()
    #show_augmentation()

    visualize_and_comparison(0.965)

    #count_falsePositiveCase_to_pkl()
    #visualizd_falsePositiveCase_from_pkl('classifier_threshold_analysis_bbox_classifier.pkl')
    #count_falseNegativeCase_to_pkl()
    #visualizd_falseNegativeCase_from_pkl('classifier_threshold_analysis_bbox_confidence_higherthan_095.pkl')
    #extractBboxFromTrainingData()

    #count_falsePositiveCase()
    #build_training_data_to_pkl()
    #print(find_lastest_model_and_config_class('.'))

    # ind_lastest_model_and_config_class('.', DetectorConfig)