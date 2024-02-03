import torch
import os
import pandas as pd
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_file,
        # img_dir,
        # label_dir,
        S=7, B=2, C=20, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = "C:/data/pascalvoc/data/images"
        self.label_dir = "C:/data/pascalvoc/data/labels"
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                #(bounding box) 11 0.34419263456090654 0.611 0.4164305949008499 0.262
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x) # 형변환
                    for x in label.split()
                ]

                boxes.append([class_label, x, y, width, height]) #라벨 읽어오기

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0]) #이미지 읽어오기
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform: #이미지 해상도 등 변경
            #print('image:',image,'boxes:',boxes)
            image = self.transform(image)
            #boxes = self.transform(boxes)
            #image, boxes = self.transform(image = image, target = boxes)

        #이미지 데이터에서 셀 나누기
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            #i, j는 셀의 행, 열
            i, j = int(self.S * y), int(self.S * x)

            #셀의 x, y좌표
            x_cell, y_cell = self.S * x - i, self.S * y - j

            #셀의 너비, 높이 (❓로직 다시 생각해봐야 함.)
            '''
            https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/dataset.py
            1. 바운딩 박스의 너비를 픽셀 단위로 계산
                바운딩 박스의 픽셀 단위 너비 = 상대너비 * 실제 이미지 크기
            2. 이미지의 너비를 셀 단위로 바꾸기위해 이이미지의 너비 이용
                cell_pixels = self.image_width
            3. 바운딩 박스의 너비를 셀 단위로 바꾸기
                바운딩 박스의 셀 단위 너비 =  width_pixels/cell_pixels
            '''
            width_cell, height_cell = width * self.S, height * self.S

            if label_matrix[i, j, 20] == 0:#특정 셀에서 객체가 발견되었다면 더이상 객체를 찾지 않음
                label_matrix[i, j, 20] = 1

                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, 21:25] = box_coordinates

                label_matrix[i, j, class_label] = 1 #원핫 인코딩 (나머지는 다 0으로 저장됨)

        return image, label_matrix