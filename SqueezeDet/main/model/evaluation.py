import numpy as np
import main.utils.utils as utils

def filter_prediction(boxes, probs, cls_idx, config):
    """Filter bounding box predictions with probability threshold and
    non-maximum supression.
    
    Args:
      boxes: array of [cx, cy, w, h].
      probs: array of probabilities
      cls_idx: array of class indices
    Returns:
      final_boxes: array of filtered bounding boxes.
      final_probs: array of filtered probabilities
      final_cls_idx: array of filtered class indices
    """

    #check for top n detection flags
    if config.TOP_N_DETECTION < len(probs) and config.TOP_N_DETECTION > 0:
      order = probs.argsort()[:-config.TOP_N_DETECTION-1:-1]
      probs = probs[order]
      boxes = boxes[order]
      cls_idx = cls_idx[order]
      
    else:

      filtered_idx = np.nonzero(probs>config.PROB_THRESH)[0]
      probs = probs[filtered_idx]
      boxes = boxes[filtered_idx]
      cls_idx = cls_idx[filtered_idx]
    
    final_boxes = []
    final_probs = []
    final_cls_idx = []

    #go trough classes
    for c in range(config.CLASSES):
      idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]

      #do non maximum suppresion
      keep = utils.nms(boxes[idx_per_class], probs[idx_per_class], config.NMS_THRESH)
      for i in range(len(keep)):
        if keep[i]:
          final_boxes.append(boxes[idx_per_class[i]])
          final_probs.append(probs[idx_per_class[i]])
          final_cls_idx.append(c)

    return final_boxes, final_probs, final_cls_idx

def filter_batch( y_pred,config):
    """filters boxes from predictions tensor
    
    Arguments:
        y_pred {[type]} -- tensor of predictions
        config {[type]} -- squeezedet config
    
    Returns:
        lists -- list of all boxes, list of the classes, list of the scores
    """




    #slice predictions vector
    pred_class_probs, pred_conf, pred_box_delta = utils.slice_predictions_np(y_pred, config)
    det_boxes = utils.boxes_from_deltas_np(pred_box_delta, config)

    #compute class probabilities
    probs = pred_class_probs * np.reshape(pred_conf, [config.BATCH_SIZE, config.ANCHORS, 1])
    det_probs = np.max(probs, 2)
    det_class = np.argmax(probs, 2)



    #count number of detections
    num_detections = 0


    all_filtered_boxes = []
    all_filtered_scores = []
    all_filtered_classes = [ ]

    #iterate batch
    for j in range(config.BATCH_SIZE):

        #filter predictions with non maximum suppression
        filtered_bbox, filtered_score, filtered_class = filter_prediction(det_boxes[j], det_probs[j],
                                                                          det_class[j], config)


        #you can use this to use as a final filter for the confidence score
        keep_idx = [idx for idx in range(len(filtered_score)) if filtered_score[idx] > float(config.FINAL_THRESHOLD)]

        final_boxes = [filtered_bbox[idx] for idx in keep_idx]

        final_probs = [filtered_score[idx] for idx in keep_idx]

        final_class = [filtered_class[idx] for idx in keep_idx]


        all_filtered_boxes.append(final_boxes)
        all_filtered_classes.append(final_class)
        all_filtered_scores.append(final_probs)


        num_detections += len(filtered_bbox)


    return all_filtered_boxes, all_filtered_classes, all_filtered_scores