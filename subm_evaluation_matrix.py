import os
import pandas as pd

DATA_DIR = '../Input' # to be modified
# Directory to save logs and trained model
ROOT_DIR = '.'
THRESHOLD_STEP = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

def is_over_lapped(l1_x, l1_y, r1_x, r1_y, l2_x, l2_y, r2_x, r2_y):
    '''
        To tell if two rectangle is overlapped or not.
        l1: top left point of rectangle 1
        r1: bottom right of rectangle 1
        l2: top left point of rectangle 2
        r2: bottom right of rectangle 2
    :return bool
    '''
    # If once rectangle is on left side of other
    if l1_x > r2_x or l2_x > r1_x:
        return False
    # If once rectangle is above other
    if l1_y < r2_y or l2_y < r1_y:
        return False
    return True

def IOU(rect1, rect2): #(tl_x, tl_y, br_x, br_y)
    x_overlap = max(0 ,min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]))
    y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]))
    SI = x_overlap * y_overlap
    S = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1]) + (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    return SI / (S - SI)

def image_mean_precision(predicted_bbox, gt_bbox):
    '''
        To calculate the mean precision across given thresholds
    :return: float
    '''
    if len(gt_bbox) == 0 and len(predicted_bbox) > 0: # if only false positive result
        return 0
    if len(predicted_bbox) == 0: # no bbox when there should be pneumonia (true negative result is removed previously)
        return 0
    precisions = [] #order by score desc
    gb2pb = []
    for g_bbox in gt_bbox:
        candidate = [] # in the order of predicted_bbox
        for p_bbox in predicted_bbox:
            candidate.append(IOU(p_bbox, g_bbox))
        gb2pb.append(candidate)
    iter = len(gt_bbox)
    for c in range(iter):
        best_iou = 0
        best_iou_index = (-1, -1)
        for i in range(len(gt_bbox)):
            for j in range(len([predicted_bbox])):
                if gb2pb[i][j] > best_iou: # 0 doesn't count
                    best_iou = gb2pb[i][j]
                    best_iou_index = (i, j)
        precisions.append(best_iou) # could be 0 if no match

        if best_iou == 0: #no more match with score higher then 0
            #remove row only
            precisions += [0] * (iter - c - 1)
            break
        else: #delete i row and j column(because they shouldn't be matched twice)
            for row in gb2pb:
                del row[best_iou_index[1]] #delete col first
            del gb2pb[best_iou_index[0]] #delete row then
    result = .0
    for th in THRESHOLD_STEP:
        TP = len(list(filter(lambda x: x >= th, precisions))) #prediction with score higher then threshold
        FP = len(predicted_bbox) - TP #all prediction - acceptable result
        FN = len(gt_bbox) - TP
        result += TP / (TP + FP + FN)
    return result / len(THRESHOLD_STEP)

def final_mean_precision(subm_file_name = 'submission.csv', gt_file_name = 'stage_1_test_labels.csv'):
    '''
    1. Finding amount of patient, who had no real boxes and had no predicted boxes.
       Remove these patient from our test set. Let's denote amount of remained patient by n
    2. For each other case we calculate average precision. (False positive result adds 0 score but increases n)
       So, we have n values: AP_1, AP_2, ..., AP_n
    3. Final score is (AP_1 + AP_2 + AP_n) / n
    :return:
    '''
    # read testing data in stage 1
    print('Reading labels from stage-1 testing data:')
    gt_bbox = {}
    anns = pd.read_csv(os.path.join(DATA_DIR, gt_file_name))
    for index, row in anns.iterrows():
        gt_bbox[row['patientId']] = []
        if row['Target'] == 1:
            tlx = int(row['x'])
            tly = int(row['y'])
            brx = tlx + int(row['width']) - 1
            bry = tly + int(row['height']) - 1
            gt_bbox[row['patientId']].append((tlx, tly, brx, bry))
    print('# of images in stage-1 testing data: ' + str(len(gt_bbox)))
    #read submission file from model
    print('Reading labels from submission file predicted by model:')
    predicted_bbox = {}
    anns = pd.read_csv(os.path.join(ROOT_DIR, subm_file_name), converters={'predictionString': str})
    for index, row in anns.iterrows():
        predicted_bbox[row['patientId']] = []
        if row['predictionString']:
            arr = list(map(float, row['predictionString'].split()))
            for cord in [arr[i+1:i+5] for i in range(0, len(arr), 5)]:
                 predicted_bbox[row['patientId']].append((cord[0], cord[1], cord[0] + cord[2] - 1, cord[1] + cord[3] - 1))
    print('# of images in submission file: ' + str(len(gt_bbox)))
    assert len(gt_bbox) == len(gt_bbox), 'Number of images does not math!'
    assert sorted(gt_bbox.keys()) == sorted(predicted_bbox.keys()), 'PatientId does not match!'

    print('Evaluating score...')
    print('Step 1. Removing images without bbox in both prediction and ground truth...')
    #step 1
    patientIds = list(predicted_bbox.keys())
    for id in patientIds:
        if not predicted_bbox[id] and not gt_bbox[id]:
            del predicted_bbox[id]
            del gt_bbox[id]
    #step 2
    print('# of images in gt_bbox pool: ' + str(len(gt_bbox)))
    print('# of images in predicted_bbox pool: ' + str(len(predicted_bbox)))
    assert len(gt_bbox) == len(gt_bbox), 'Number of images does not math!'
    print('Step 2. Calculating the score for each image...')
    #iterate by patient Id:
    total_score = .0
    count = 0
    for patientId in predicted_bbox:
        print(patientId)
        score = image_mean_precision(predicted_bbox[patientId] ,gt_bbox[patientId])
        print(score)
        count += 1
        total_score += score

    return total_score / count

if __name__ == '__main__':
    # test IOU
    # rect1 = (0, 0, 4, 4)
    # rect2 = (2, 2, 6, 6)
    # print(IOU(rect1, rect2))
    result = final_mean_precision('submission_comb_s1.csv')
    print(result)