from glob import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16 as PTModel, preprocess_input
# params we will probably want to do some hyperparameter optimization later
BASE_MODEL= 'VGG16' #'VGG16' #'VGG16' # ['VGG16', 'RESNET52', 'InceptionV3', 'Xception', 'DenseNet169', 'DenseNet121']
IMG_SIZE = (384, 384) # [(224, 224), (384, 384), (512, 512), (640, 640)]
GPU_COUNT = 1
IMAGES_PER_GPU = 6
BATCH_SIZE = 24 #GPU_COUNT * IMAGES_PER_GPU# [1, 8, 16, 24]
DENSE_COUNT = 128 # [32, 64, 128, 256]
DROPOUT = 0.25 # [0, 0.25, 0.5]
LEARN_RATE = 1e-4 # [1e-4, 1e-3, 4e-3]
TRAIN_SAMPLES = 60000 # [3000, 6000, 15000] #if 3 classes balanced, should be divisible by 3
TEST_SAMPLES = 200 # data size for validation
#USE_ATTN = False # [True, False]
PRETRAIN_MODEL_TRAINABLE = False
TEST_SIZE = 0.2
EPOCH = 10
NUM_CLASS = 2
PATIENCE = 10 # parameters for early stop call back
VIEW_POINT_MODE = '' # ['PA', 'AP', '']
INCLUDE_NIH_CHEST = True

# To specify which gpu to be used by set the env variable
if GPU_COUNT == 1:
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # The GPU id to use, usually either "0" or "1"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

image_bbox_df = pd.read_csv('../../Input/image_bbox_full.csv')
image_bbox_df = image_bbox_df[['patientId', 'Target', 'path']]
#print(image_bbox_df[:2])
image_bbox_df['path'] = image_bbox_df['path'].map(lambda x: x.replace('path2data', '../../Input/stage_1_train_images'))
image_bbox_df = image_bbox_df.groupby('patientId').apply(lambda x: x.sample(1))
#print(image_bbox_df[:2])
#print(image_bbox_df.columns)
if VIEW_POINT_MODE:
    image_bbox_df = image_bbox_df[image_bbox_df.ViewPosition == VIEW_POINT_MODE]
if INCLUDE_NIH_CHEST:
    image_bbox_df_NIH = pd.read_csv('../../Input/NIH_chest_image_full.csv')
    image_bbox_df_NIH = image_bbox_df_NIH[['patientId', 'Target']]
    image_bbox_df_NIH['path'] = image_bbox_df_NIH['patientId'].map(lambda x: '../../Input/NIH_CHEST_XRAY/images/' + x) #patientId = file name
    #print(image_bbox_df_NIH.columns)
    #merge 2 dataset
    frames = [image_bbox_df, image_bbox_df_NIH]
    #print(image_bbox_df[:2])
    #print(image_bbox_df_NIH[:2])
    image_bbox_df = pd.concat(frames, ignore_index=True)
# print(image_bbox_df.shape[0], 'images')
# print(image_bbox_df[:3])

# get the labels in the right format
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
class_enc = LabelEncoder()
image_bbox_df['class_idx'] = class_enc.fit_transform(image_bbox_df['Target'])
oh_enc = OneHotEncoder(sparse=False)
image_bbox_df['class_vec'] = oh_enc.fit_transform(image_bbox_df['class_idx'].values.reshape(-1, 1)).tolist()
image_df = image_bbox_df
# split data into training and validation set
from sklearn.model_selection import train_test_split
#print(image_df.shape)
# split data into training and validation set
from sklearn.model_selection import train_test_split

raw_train_df, valid_df = train_test_split(image_df, test_size=0.02 , random_state=2018, stratify=image_df['Target'])
#balance training dataset nd reduce the total image count
train_df = raw_train_df.groupby('Target').apply(lambda x: x.sample(48000//2)).reset_index(drop=True)

#Image Transplantation
try:
    # keras 2.2
    import keras_preprocessing.image as KPImage
except:
    # keras 2.1
    import keras.preprocessing.image as KPImage

from PIL import Image
import pydicom


def read_dicom_image(in_path):
    img_arr = pydicom.read_file(in_path).pixel_array
    return img_arr / img_arr.max()


class medical_pil():
    @staticmethod
    def open(in_path):
        if '.dcm' in in_path:
            c_slice = read_dicom_image(in_path)
            int_slice = (255 * c_slice).clip(0, 255).astype(np.uint8)  # 8bit images are more friendly
            return Image.fromarray(int_slice)
        else:
            return Image.open(in_path)

    fromarray = Image.fromarray
KPImage.pil_image = medical_pil

# pick model
from keras.preprocessing.image import ImageDataGenerator
if BASE_MODEL=='VGG16':
    from keras.applications.vgg16 import VGG16 as PTModel, preprocess_input
elif BASE_MODEL == 'VGG19':
    from keras.applications.vgg19 import vgg19 as PTModel, preprocess_input
elif BASE_MODEL=='RESNET52':
    from keras.applications.resnet50 import ResNet50 as PTModel, preprocess_input
elif BASE_MODEL=='INCEPTION_RESNET_V2': # default image size: 299 x 299
    from keras.applications.inception_resnet_v2 import InceptionResNetV2 as PTModel, preprocess_input
elif BASE_MODEL=='InceptionV3':
    from keras.applications.inception_v3 import InceptionV3 as PTModel, preprocess_input
elif BASE_MODEL=='Xception':
    from keras.applications.xception import Xception as PTModel, preprocess_input
elif BASE_MODEL=='DenseNet169':
    from keras.applications.densenet import DenseNet169 as PTModel, preprocess_input
elif BASE_MODEL=='DenseNet121':
    from keras.applications.densenet import DenseNet121 as PTModel, preprocess_input
else:
    raise ValueError('Unknown model: {}'.format(BASE_MODEL))

# data generation and augmentation
# https://keras.io/preprocessing/image/
img_gen_args = dict(samplewise_center=False,
                              samplewise_std_normalization=False,
                              horizontal_flip = False,
                              vertical_flip = False,
                              height_shift_range = 0.05,
                              width_shift_range = 0.02,
                              rotation_range = 3,
                              shear_range = 0.01,
                              fill_mode = 'nearest',
                              zoom_range = 0.05,
                              preprocessing_function=preprocess_input)

img_gen = ImageDataGenerator(**img_gen_args)

def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, seed = None, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways: seed: {}'.format(seed))
    df_gen = img_data_gen.flow_from_directory(base_dir,
                                     class_mode = 'sparse',
                                              seed = seed,
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values,0)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen

train_gen = flow_from_dataframe(img_gen, train_df,
                             path_col = 'path',
                            y_col = 'class_vec',
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = BATCH_SIZE)

# valid_gen = flow_from_dataframe(img_gen, valid_df,
#                              path_col = 'path',
#                             y_col = 'class_vec',
#                             target_size = IMG_SIZE,
#                              color_mode = 'rgb',
#                             batch_size = VALIDATION_BATCH_SIZE) # we can use much larger batches for evaluation
# used a fixed dataset for evaluating the algorithm
valid_X, valid_Y = next(flow_from_dataframe(img_gen, valid_df,
                             path_col = 'path',
                            y_col = 'class_vec',
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = TEST_SAMPLES)) # one big batch

t_x, t_y = next(train_gen)
base_pretrained_model = PTModel(input_shape =  t_x.shape[1:],
                              include_top = False, weights = 'imagenet')
base_pretrained_model.trainable = PRETRAIN_MODEL_TRAINABLE

#Model Supplements
#Here we add a few other layers to the model to make it better suited for the classification problem.

from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda, AvgPool2D
from keras.models import Model
from keras.optimizers import Adam
# catch the output as feature from picked pre-trained model
pt_features = Input(base_pretrained_model.get_output_shape_at(0)[1:], name = 'feature_input')
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
dr_steps = Dropout(DROPOUT)(Dense(DENSE_COUNT, activation = 'elu')(gap_dr))
out_layer = Dense(t_y.shape[1], activation = 'softmax')(dr_steps)

attn_model = Model(inputs = [pt_features],
                   outputs = [out_layer], name = 'trained_model')

attn_model.summary()

from keras.models import Sequential
from keras.optimizers import Adam
pneu_model = Sequential(name = 'combined_model')
base_pretrained_model.trainable = PRETRAIN_MODEL_TRAINABLE
pneu_model.add(base_pretrained_model)
pneu_model.add(attn_model)
#train with multiple GPU
from keras.utils.training_utils import multi_gpu_model
#multi_gpu_model bug: val_loss: 1.1921e-07, infinity or a value too large for dtype('float32') in prediction
gpu_pneu_model = pneu_model #multi_gpu_model(pneu_model, gpus=GPU_COUNT)
#load pre-trained weights
gpu_pneu_model.load_weights('VGG16_full_model.h5')
gpu_pneu_model.compile(optimizer = Adam(lr = LEARN_RATE), loss = 'categorical_crossentropy',
                           metrics = ['categorical_accuracy'])
gpu_pneu_model.summary()

#predict
pred_Y = pneu_model.predict(valid_X,
                          batch_size = BATCH_SIZE,
                          verbose = True)
print(pred_Y)
#summary of result
valid_r = np.argmax(valid_Y, -1)
pred_r  = np.argmax(pred_Y,-1)
from sklearn.metrics import classification_report, confusion_matrix
plt.matshow(confusion_matrix(valid_r, pred_r))
print(classification_report(valid_r, pred_r, target_names = ['0', '1'])) #class_enc.classes_ [0, 1]

#visualize AUC and save it
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, _ = roc_curve(np.argmax(valid_Y,-1)==0, pred_Y[:,0]) # indices of max probability => class
fig, ax1 = plt.subplots(1,1, figsize = (5, 5), dpi = 250)
ax1.plot(fpr, tpr, 'b.-', label = BASE_MODEL + ' (AUC:%2.2f)' % roc_auc_score(np.argmax(valid_Y,-1)==0, pred_Y[:,0]))
ax1.plot(fpr, fpr, 'k-', label = 'Random Guessing')
ax1.legend(loc = 4)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate');
ax1.set_title('Lung Opacity ROC Curve')
fig.savefig('roc_valid.pdf')