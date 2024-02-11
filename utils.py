import torch

def intersection_over_union(boxes_preds, boxes_labels, box_format):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes
        boxes_labels (tensor): Correct labels of Bounding Boxes 
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
        
    Process:
        if boxes (x,y,w,h) [box_format: midpoint]
            we need to create (min_x, min_y, max_x, max_y) from (x, y, w, h)
        if boxes (x1,y1,x2,y2) [box_format: corner]
            we don't need to create. but need to create width and height

    Returns:
        tensor: Intersection over union for all examples

    Reference:
        https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/utils.py
        https://gaussian37.github.io/math-algorithm-iou/

    """
    
    # Create (min_x, min_y, max_x, max_y) if box_format is "midpoint" 
    if box_format == "midpoint":
        x_center_p = boxes_preds[..., 0:1]
        y_center_p = boxes_preds[..., 1:2]
        width_p = boxes_preds[..., 2:3]
        height_p = boxes_preds[..., 3:4]

        x_center_l = boxes_labels[..., 0:1]
        y_center_l = boxes_labels[..., 1:2]
        width_l = boxes_labels[..., 2:3]
        height_l = boxes_labels[..., 3:4]

        min_x_p = x_center_p - width_p / 2
        max_x_p = x_center_p + width_p / 2
        min_y_p = y_center_p - height_p / 2
        max_y_p = y_center_p + height_p / 2

        min_x_l = x_center_l - width_l / 2
        max_x_l = x_center_l + width_l / 2
        min_y_l = y_center_l - height_l / 2
        max_y_l = y_center_l + height_l / 2

    # Define (min_x, min_y, max_x, max_y) if box_format is "corner"
    if box_format == "corner":
        min_x_p = boxes_preds[..., 0:1]
        max_x_p = boxes_preds[..., 1:2]
        min_y_p = boxes_preds[..., 2:3]
        max_y_p = boxes_preds[..., 3:4]

        min_x_l = boxes_labels[..., 0:1]
        max_x_l = boxes_labels[..., 1:2]
        min_y_l = boxes_labels[..., 2:3]
        max_y_l = boxes_labels[..., 3:4]

        width_p = max_x_p - min_x_p
        height_p = max_y_p - min_y_p
        width_l = max_x_l - min_x_l
        height_l = max_y_l - min_y_l

    # Find the length of the x and y sides in rectangle of intersection.
    intersection_x_length = torch.min(max_x_p, max_x_l) - torch.max(min_x_p, min_x_l)
    intersection_y_length = torch.min(max_y_p, max_y_l) - torch.max(min_y_p, min_y_l)

    # Define intersection and union to find iou
    intersection = intersection_x_length * intersection_y_length
    union = width_p * height_p + width_l * height_l - intersection
    
    return intersection / union