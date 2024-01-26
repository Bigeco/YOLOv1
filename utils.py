import torch

"""
midpoint -> box_* : S * S * (x, y, w, h)     
    : (x, y, w, h) -> (min_x, min_y, max_x, max_y)
corner   -> box_* : S * S * (min_x, min_y, max_x, max_y)
    : (min_x, min_y, max_x, max_y) -> not to change
"""

def intersection_over_union(box_preds, box_labels, box_format):
    
    # 
    if box_format == "midpoint":
        x_center_p = box_preds[..., 0]
        y_center_p = box_preds[..., 1]
        width_p = box_preds[..., 2]
        height_p = box_preds[..., 3]

        x_center_l = box_labels[..., 0]
        y_center_l = box_labels[..., 1]
        width_l = box_labels[..., 2]
        height_l = box_labels[..., 3]

        min_x_p = x_center_p - width_p / 2
        max_x_p = x_center_p + width_p / 2
        min_y_p = y_center_p - height_p / 2
        max_y_p = y_center_p + height_p / 2

        min_x_l = x_center_l - width_l / 2
        max_x_l = x_center_l + width_l / 2
        min_y_l = y_center_l - height_l / 2
        max_y_l = y_center_l + height_l / 2

    # 
    if box_format == "corner":
        min_x_p = box_preds[..., 0]
        max_x_p = box_preds[..., 1]
        min_y_p = box_preds[..., 2]
        max_y_p = box_preds[..., 3]

        min_x_l = box_labels[..., 0]
        max_x_l = box_labels[..., 1]
        min_y_l = box_labels[..., 2]
        max_y_l = box_labels[..., 3]

        width_p = max_x_p - min_x_p
        height_p = max_y_p - min_y_p
        width_l = max_x_l - min_x_l
        height_l = max_y_l - min_y_l

    intersection_x_length = torch.min(max_x_p, max_x_l) - torch.max(min_x_p, min_x_l)
    intersection_y_length = torch.min(max_y_p, max_y_l) - torch.max(min_y_p, min_y_l)

    intersection = intersection_x_length * intersection_y_length
    union = width_p * height_p + width_l * height_l - intersection
    
    return intersection / union