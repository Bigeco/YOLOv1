import torch
import torch.nn as nn
import numpy as np
from utils import intersection_over_union

class YoloLoss():
    '''
    Calculate the loss for YOLOv1 model
    '''
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        self.lambda_noobj = 0.5
        self.lambda_coord = 5


    def localizationLoss(self, bbox_true, responsible_box):
        '''
        Parameters:
            bbox_true: 실제 (x,y,w,h) 값
            responsible_box: 예측된 (x,y,w,h) 값

        Returns: localization loss값
        '''

        #실제 x,y,w,h값과 예측된 x,y,w,h값의 차이 계산
        localization_err_x = torch.pow(torch.subtract(bbox_true[0], responsible_box[0]), 2)  # x손실값 계산
        localization_err_y = torch.pow(torch.subtract(bbox_true[1], responsible_box[1]), 2)  # y손실값 계산
        localization_err_w = torch.pow(torch.subtract(torch.sqrt(bbox_true[2]), torch.sqrt(responsible_box[2])), 2)  # w손실값 계산
        localization_err_h = torch.pow(torch.subtract(torch.sqrt(bbox_true[3]), torch.sqrt(responsible_box[3])), 2)  # h손실값 계산

        # 셀 안에 객체가 존재하지 않을 경우 w, h값은 null값
        # nan값 제거하기
        if torch.isnan(localization_err_w).detach().numpy() == True:
            localization_err_w = torch.zeros_like(localization_err_w)
        if torch.isnan(localization_err_h).detach().numpy() == True:
            localization_err_h = torch.zeros_like(localization_err_h)

        # x,y끼리 w,z끼리 더하기
        localization_err_xy = torch.add(localization_err_x, localization_err_y)
        localization_err_wh = torch.add(localization_err_w, localization_err_h)
        localization_err = torch.add(localization_err_xy, localization_err_wh)

        localization_err = torch.multiply(localization_err, self.obj_exist) #1obj(i) 곱하기
        weighted_localization_err = torch.multiply(localization_err, 5.0)  # λ_coord 곱하기

        return weighted_localization_err #localizationLoss
    
    def confidence_loss(self, y_pred, y_true):
        """
        Parameters:
            y_pred (tensor): (BATCH_SIZE, [x1, y1, w1, h1, p1, x2, y2, w2, h2, p2, c1,...,c20])
            y_true (tnesor): (BATCH_SIZE, [x, y, w, h, p, c1,..., c20])
        
        Progress:
            (1) Calculate IoU for each
            (2) Take the box with highest IoU out of the two prediction
            (3) Calculate confidence loss - Object Loss 
            (4) Calculate confidence loss - No Object Loss

        References:
            https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/loss.py

        """
        # y_pred (tensor): [c1,...,c20, p1, x1, y1, w1, h1, p2, x2, y2, w2, h2, p2]
        # y_true (tnesor): [c1,...,c20, p, x, y, w, h] 
        # 라고 가정할 때는
        # y_pred[..., 20], y_pred[..., 25]
        # y_true[..., 20]
        
        
        # (1) Calculate IoU for each
        iou_b1 = intersection_over_union(y_pred[..., :4], y_true[..., :4], "midpoint")
        iou_b2 = intersection_over_union(y_pred[..., 5:9], y_true[..., :4], "midpoint")
        

        # (2) Take the box with highest IoU out of the two prediction
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0) # shape: [2, BATCH_SIZE, S, S, 30]
        # Return values and indices
        # values (iou_maxes): Maximum value Tensor(BATCH_SIZE * S * S * 30) in each column
        # indices (bestbox): the index of the maximum values(BATCH_SIZE * S * S * 30) ​​in each column.
        # Examples: 
        #   [[[0, 1, 4, 5],[5, 6, 7, 8]],[[9, 10, 11, 12],[13, 14, 15, 16]]]
        #   [[[1, 2, 3, 4],[5, 6, 7, 8]],[[9, 10, 11, 12],[13, 14, 15, 16]]]
        #   return 
        #       [[[1, 2, 4, 5],[5, 6, 7, 8]],[[9, 10, 11, 12],[13, 14, 15, 16]]]
        #       [[[0, 0, 1, 1],[0, 0, 0, 0]],[[0, 0, 0, 0],[0, 0, 0, 0]]]
        #       0: 0 idx element
        #       1: 1 idx element
        # Real Code:
        iou_maxes, bestbox = torch.max(ious, dim=0) 
        
        
        # (3) Calculate confidence loss - Object Loss 
        confidence_pred_box1 = y_pred[..., 4]
        confidence_pred_box2 = y_pred[..., 9]
        exists_box = y_true[..., 4] # Iobj_i

        pred_box = (
            bestbox * confidence_pred_box2 + (1-bestbox) * confidence_pred_box1
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * confidence_pred_box1)
        )

                 
        # (4) Calculate confidence loss - No Object Loss
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * confidence_pred_box1, start_dim=1),
            torch.flatten((1 - exists_box) * exists_box, start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * confidence_pred_box2, start_dim=1),
            torch.flatten((1 - exists_box) * exists_box, start_dim=1)
        )

        return object_loss, no_object_loss
        

    def classificationLoss(self, class_true, class_pred):
        # 1. 실제 확률값에서 예측 확률값 빼고 제곱
        classification_err = torch.pow(torch.subtract(class_true, class_pred), 2.0)
        # 2. 원소 값들을 모두 더하기
        classification_err = torch.sum(classification_err)
        # 3. Iobj_i값과 곱하기
        # obj_exist값은 위에서 정의됨-값이 존재하는지 여부를 판단하는 값
        classification_err = torch.multiply(classification_err, self.obj_exist)

        return classification_err #classificationLoss

    def forward(self, y_pred, y_true):
        """
        Parameters:
            y_pred (tensor): [x1, y1, w1, h1, p1, x2, y2, w2, h2, p2, c1,...,c20]
            y_true (tnesor): [x, y, w, h, p, c1,..., c20]

        """
        count = len(y_true) #y_true의 행의 개수
        for i in range(0, count):
            #파라미터로 받은 값을 복제
            y_true_unit = y_true[i].clone().detach().requires_grad_(True)
            y_pred_unit = y_pred[i].clone().detach().requires_grad_(True)

            #❓velog 코드에서는 -1 자리에 49를 썼는데 왜 49를 썼는지 이해하지 못함
            y_true_unit = torch.reshape(y_true_unit, (-1,25))
            y_pred_unit = torch.reshape(y_pred_unit, (-1,30))

            loss = 0

            for j in range(0, len(y_true_unit)):
                # 클래스 확률
                # 첫번째 bounding box의 예측값과 confidence score
                bbox1_pred = y_pred_unit[j, :4].clone().detach().requires_grad_(True)
                bbox1_pred_confidence = y_pred_unit[j, 4].clone().detach().requires_grad_(True)
                # 두번째 bounding box의 예측값과 confidence score
                bbox2_pred = y_pred_unit[j, 5:9].clone().detach().requires_grad_(True)
                bbox2_pred_confidence = y_pred_unit[j, 9].clone().detach().requires_grad_(True)
                # 클래스 확률
                class_pred = y_pred_unit[j, 10:].clone().detach().requires_grad_(True)

                # bounding box의 실제값
                bbox_true = y_true_unit[j, :4].clone().detach().requires_grad_(True)
                # bounding box의 실제 confidence score
                bbox_true_confidence = y_true_unit[j, 4].clone().detach().requires_grad_(True)
                # bounding box의 실제 클래스 확률
                class_true = y_true_unit[j, 5:].clone().detach().requires_grad_(True)

                #Iou 값 구하기 (midpoint format으로 구함)
                iou_bbox1_pred = intersection_over_union(bbox1_pred, bbox_true, "midpoint")
                iou_bbox2_pred = intersection_over_union(bbox2_pred, bbox_true, "midpoint")

                #Iou가 더 큰 bounding box 선택
                if iou_bbox1_pred >= iou_bbox2_pred :
                    responsible_box = bbox1_pred.clone().detach().requires_grad_(True)
                    responsible_bbox_confidence = bbox1_pred_confidence.clone().detach().requires_grad_(True)
                    non_responsible_bbox_confidence = bbox2_pred_confidence.clone().detach().requires_grad_(True)
                else :
                    responsible_box = bbox2_pred.clone().detach().requires_grad_(True)
                    responsible_bbox_confidence = bbox2_pred_confidence.clone().detach().requires_grad_(True)
                    non_responsible_bbox_confidence = bbox1_pred_confidence.clone().detach().requires_grad_(True)
                responsible_box = responsible_box
                responsible_bbox_confidence = responsible_bbox_confidence
                non_responsible_bbox_confidence = non_responsible_bbox_confidence

                #Iobj_i 값 구하기
                self.obj_exist = torch.ones_like(bbox_true_confidence)
                box_true_np = bbox_true.detach().numpy()
                if box_true_np[0] == 0.0 and box_true_np[1] == 0.0 and box_true_np[2] == 0.0 and box_true_np[3] == 0.0:
                    self.obj_exist = torch.zeros_like(bbox_true_confidence)

                #localization loss 구하기
                weighted_localization_err = self.localizationLoss(bbox_true, responsible_box)

                #classification loss 구하기
                classification_err = self.classificationLoss(class_true, class_pred)

        # Get confidence loss 
        object_loss, no_object_loss = self.confidence_loss(y_true, y_pred)

        loss = (
            self.lambda_coord
            + object_loss
            + self.lambda_noobj * no_object_loss
        )

        return loss


