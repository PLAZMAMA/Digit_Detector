import numpy as np
from PIL import Image
from random import randint
import os

x = []
y = []

for i in range(10000):
    num = randint(0,9)
    file_len = os.listdir(f"/Users/maor/Documents/src/Digit_Detector/digit_dataset/{num}")
    file_len = [file for file in file_len if ".png" in file]#makes sure it only counts the pictures
    file_len = len(file_len)
    pic_num = randint(0, file_len - 1) # "-1" because it counts from "0"
    img = Image.open(f"/Users/maor/Documents/src/Digit_Detector/digit_dataset/{num}/conv_pic_{pic_num}.png")
    img_arr = np.asarray(img).tolist()
    x.append(img_arr)
    y.append(num)

x = np.array(x)
y = np.array(y)
np.save("x_test.npy", x)
np.save("y_test.npy", y)