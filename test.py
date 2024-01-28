import os

label_path = os.path.join("C:/Users/esybd/Documents/카카오톡 받은 파일/000001.txt")
with open(label_path) as f:
    for label in f.readlines():
        class_label, x, y, width, height = [
            print(x, type(x))
            for x in label.split()
        ]
        print(class_label, x, y, width, height)