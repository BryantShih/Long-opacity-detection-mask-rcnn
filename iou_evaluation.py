import os
import time
import random
import numpy as np
import mrcnn.model as modellib
import pandas as pd
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from run import DetectorConfig
from utilities import find_lastest_model, read_training_data_from_pkl, parse_dataset, DetectorDataset

DATA_DIR = '../Input' # to be modified

# Directory to save logs and trained model
ROOT_DIR = '.'
train_dicom_dir = os.path.join(DATA_DIR, 'stage_1_train_images')
test_dicom_dir = os.path.join(DATA_DIR, 'stage_1_test_images')
ORIG_SIZE = 1024
############################################################
#  Result Evaluation
############################################################
def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_bbx(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):

    """Runs IOU evaluation.(modified from evaluate_coco from COCO API)
        dataset: A Dataset object with valiadtion data
        eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
        limit: if not 0, it's the number of images to use for evaluation
        """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"], r["masks"])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results) # a new COCO object returned from loadRes directly.

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)

if __name__ == '__main__':
    # retreive path to model and config class
    model_path = find_lastest_model(ROOT_DIR)

    class InferenceConfig(DetectorConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.9

    inference_config = InferenceConfig()
    inference_config.display()

    model = modellib.MaskRCNN(mode='inference',
                              config=inference_config,
                              model_dir=ROOT_DIR)

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    dataset = read_training_data_from_pkl()
    anns = pd.read_csv(os.path.join(DATA_DIR, 'stage_1_train_labels.csv'))
    image_fps, image_annotations = parse_dataset(train_dicom_dir, anns=anns)
    # Split the data into training and validation datasets
    ######################################################################
    # Modify this line to use more or fewer images for training/validation.
    # To use all images, do: image_fps_list = list(image_fps)
    image_fps_list = list(image_fps)  # image_fps[:1000]
    #####################################################################
    # split dataset into training vs. validation dataset
    # split ratio is set to 0.9 vs. 0.1 (train vs. validation, respectively)
    sorted(image_fps_list)
    random.seed(66)
    random.shuffle(image_fps_list)
    validation_split = 0.1
    split_index = int((1 - validation_split) * len(image_fps_list))
    image_fps_val = image_fps_list[split_index:]
    # print(len(image_fps_train), len(image_fps_val))

    # prepare the validation dataset
    dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
    dataset_val.prepare()
    print("Running COCO evaluation on 1/10 of all images randomly.")
    evaluate_bbx(model, dataset_val, coco, "bbox", limit=int(2000))