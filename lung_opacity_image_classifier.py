import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# params we will probably want to do some hyperparameter optimization later
BASE_MODEL= 'Xception' #'VGG16' #'VGG16' # ['VGG16', 'RESNET50', 'InceptionV3', 'Xception', 'DenseNet169', 'DenseNet121']
IMG_SIZE = (299, 299) # [(224, 224), (384, 384), (512, 512), (640, 640)]
GPU_COUNT = 1
IMAGES_PER_GPU = 8
BATCH_SIZE = 24 #GPU_COUNT * IMAGES_PER_GPU# [1, 8, 16, 24]
DENSE_COUNT = 256 # [32, 64, 128, 256]
DROPOUT = 0.2 # [0, 0.2, 0.25, 0.5]
LEARN_RATE = 5e-5 # [1e-4, 1e-3, 4e-3]
TRAIN_SAMPLES = 60000 # [3000, 6000, 15000] #if 3 classes balanced, should be divisible by 3
TEST_SAMPLES = 2400 #validation size
#USE_ATTN = False # [True, False]
PRETRAIN_MODEL_TRAINABLE = True
TEST_SIZE = 0.1 #class balance training size: 4800; full size :137804, 0.2 will be enough
EPOCH = 30
NUM_CLASS = 2
PATIENCE = 12 # parameters for early stop call back
VIEW_POINT_MODE = '' # ['PA', 'AP', '']
INCLUDE_NIH_CHEST = True

#To specify last gpu to be used by set the env variable
if GPU_COUNT == 1:
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # The GPU id to use, usually either "0" or "1"
    os.environ["CUDA_VISIBLE_DEVICES"]="3"

#VALIDATION_BATCH_SIZE = 256
# get ground truth
#image_bbox_df = pd.read_csv('../../Input/image_bbox_full_binary_class.csv') # data for binary classifier
image_bbox_df = pd.read_csv('../../Input/image_bbox_full.csv')
image_bbox_df = image_bbox_df[['patientId', 'Target', 'path']]
image_bbox_df['path'] = image_bbox_df['path'].map(lambda x: x.replace('path2data', '../../Input/stage_2_train_images'))
s1_test_bbox_df = pd.read_csv('../../Input/stage_1_test_labels.csv')
s1_test_bbox_df = s1_test_bbox_df[['patientId', 'Target']]
s1_test_bbox_df['path'] = s1_test_bbox_df['patientId'].map(lambda x: '../../Input/stage_2_train_images/' + x + '.dcm')
frames = [image_bbox_df, s1_test_bbox_df]
image_bbox_df = pd.concat(frames, ignore_index=True)
image_bbox_df = image_bbox_df.groupby('patientId').apply(lambda x: x.sample(1))
#print(image_bbox_df.columns)
if VIEW_POINT_MODE:
    image_bbox_df = image_bbox_df[image_bbox_df.ViewPosition == VIEW_POINT_MODE]
if INCLUDE_NIH_CHEST:
    image_bbox_df_NIH = pd.read_csv('../../Input/NIH_chest_image_full.csv')
    image_bbox_df_NIH = image_bbox_df_NIH[['patientId', 'Target']]
    image_bbox_df_NIH['path'] = image_bbox_df_NIH['patientId'].map(lambda x: '../../Input/NIH_CHEST_XRAY/images/' + x) #patientId = file name
    #print(image_bbox_df_NIH.columns)
    #print(image_bbox_df.shape)
    #print(image_bbox_df_NIH.shape)
    #to balance the data and keep original RSNA data as primary one
    image_bbox_df_NIH_opacity = image_bbox_df_NIH[image_bbox_df_NIH['Target'] == 1].sample(24400) #5669 image with opacity from RSNA
    image_bbox_df_NIH_normal = image_bbox_df_NIH[image_bbox_df_NIH['Target'] == 0].sample(10000) #20015 image without opacity from RSNA
    #merge 2 dataset
    frames = [image_bbox_df, image_bbox_df_NIH_opacity, image_bbox_df_NIH_normal] #totally 30k opacity 30k normal
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

#balance training dataset (validation set is not balance) and reduce the total image count
image_df = image_df.groupby('Target').apply(lambda x: x.sample(TRAIN_SAMPLES//NUM_CLASS)).reset_index(drop=True)
#keep the proportion of classes in training and validation set the same
train_df, valid_df = train_test_split(image_df, test_size=TEST_SIZE , random_state=2018, stratify=image_df['Target'])
#train_df = raw_train_df.groupby('Target').apply(lambda x: x.sample(60000//NUM_CLASS)).reset_index(drop=True)

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
# more model option : https://keras.io/applications/
from keras.preprocessing.image import ImageDataGenerator
if BASE_MODEL=='VGG16':
    from keras.applications.vgg16 import VGG16 as PTModel, preprocess_input
elif BASE_MODEL == 'VGG19':
    from keras.applications.vgg19 import VGG19 as PTModel, preprocess_input
elif BASE_MODEL=='RESNET50':
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

# In fit_generator: validation_data could be either a generator or a tuple

# valid_gen = flow_from_dataframe(img_gen, valid_df,
#                              path_col = 'path',
#                             y_col = 'class_vec',
#                             target_size = IMG_SIZE,
#                              color_mode = 'rgb',
#                             batch_size = VALIDATION_BATCH_SIZE) # we can use much larger batches for evaluation

# We use a fixed validation set here for evaluating the algorithm
valid_X, valid_Y = next(flow_from_dataframe(img_gen, valid_df,
                             path_col = 'path',
                            y_col = 'class_vec',
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = TEST_SAMPLES)) # one big batch

t_x, t_y = next(train_gen)
base_pretrained_model = PTModel(input_shape =  t_x.shape[1:],
                              include_top = False,
                            weights = 'imagenet')
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
# Other activation functions: https://keras.io/activations/

attn_model = Model(inputs = [pt_features], outputs = [out_layer], name = 'trained_model')
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
#gpu_pneu_model.load_weights('VGG16_full_model.h5')
gpu_pneu_model.compile(optimizer = Adam(lr = LEARN_RATE), loss = 'categorical_crossentropy',
                           metrics = ['categorical_accuracy'])
# Other optimizers option: https://keras.io/optimizers/
# Other loss functions: https://keras.io/losses/
gpu_pneu_model.summary()

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
weight_path="{}_weights.best.hdf5".format('lung_opacity')

#call back list
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                                   patience=10, verbose=1, mode='auto',
                                   epsilon=0.0001, cooldown=5, min_lr=0.000001)
tensorboard = TensorBoard(log_dir='./Logs', histogram_freq=0, batch_size=BATCH_SIZE,
                          write_graph=True, write_grads=False, write_images=False,
                          embeddings_freq=0, embeddings_layer_names=None,
                          embeddings_metadata=None)
early = EarlyStopping(monitor="val_loss",
                      mode="min",
                      patience=PATIENCE) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat, tensorboard]

train_gen.batch_size = BATCH_SIZE
gpu_pneu_model.fit_generator(train_gen,
                         validation_data = (valid_X, valid_Y),
                         max_queue_size = BATCH_SIZE,
                         epochs=EPOCH,
                         callbacks=callbacks_list,
                         workers=2)
#save model with original model rather than multi_gpu_model (they share the same weights)
#pneu_model.load_weights(weight_path)
pneu_model.save(BASE_MODEL + '_full_model.h5')

#predict
pred_Y = pneu_model.predict(valid_X,
                          batch_size = TRAIN_SAMPLES * TEST_SIZE, #BATCH_SIZE,
                          verbose = True)
#summary of result
from sklearn.metrics import classification_report, confusion_matrix
plt.matshow(confusion_matrix(np.argmax(valid_Y, -1), np.argmax(pred_Y,-1)))
print(classification_report(np.argmax(valid_Y, -1),
                            np.argmax(pred_Y,-1), target_names = ['0', '1']))

#visualize AUC and save it
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, _ = roc_curve(np.argmax(valid_Y,-1)==0, pred_Y[:,0]) # indices of max probability => class
fig, ax1 = plt.subplots(1,1, figsize = (5, 5), dpi = 250)
ax1.plot(fpr, tpr, 'b.-', label = BASE_MODEL + ' (AUC:%2.2f)' % roc_auc_score(np.argmax(valid_Y,-1)==0, pred_Y[:,0]))
ax1.plot(fpr, fpr, 'k-', label = 'Random Guessing')
ax1.legend(loc = 4)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('Lung Opacity ROC Curve')
fig.savefig('roc_valid.pdf')

#predict testing data and generate submission file
# from glob import glob
# #**************************************testing data path to be added
# sub_img_df = pd.DataFrame({'path': glob('../../Isnput/stage_1_test_images/*.dcm')})
# sub_img_df['patientId'] = sub_img_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0]) #file name
#
# submission_gen = flow_from_dataframe(img_gen,
#                                      sub_img_df,
#                              path_col = 'path',
#                             y_col = 'patientId',
#                             target_size = IMG_SIZE,
#                              color_mode = 'rgb',
#                             batch_size = BATCH_SIZE,
#                                     shuffle=False)
#
# #Predict for each image twice and average the results
# from tqdm import tqdm
# sub_steps = 2*sub_img_df.shape[0]//BATCH_SIZE
# out_ids, out_vec = [], []
# for _, (t_x, t_y) in zip(tqdm(range(sub_steps)), submission_gen):
#     out_vec += [pneu_model.predict(t_x)]
#     out_ids += [t_y]
# out_vec = np.concatenate(out_vec, 0)
# out_ids = np.concatenate(out_ids, 0)
#
# pred_df = pd.DataFrame(out_vec, columns=class_enc.classes_)
# pred_df['patientId'] = out_ids
# pred_avg_df = pred_df.groupby('patientId').agg('mean').reset_index()
#
# #We use the Lung Opacity as our confidence and predict the image image.
# # It will hopefully do a little bit better than a trivial baseline, and can be massively improved.
# pred_avg_df['PredictionString'] = pred_avg_df['Lung Opacity'].map(lambda x: ('%2.2f 0 0 1024 1024' % x) if x>0.5 else '')
# pred_avg_df[['patientId', 'PredictionString']].to_csv('submission.csv', index=False)