import pandas as pd
import cv2
import os
import numpy as np
from sklearn.svm import SVC
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import hsv_to_rgb
from numpy import asarray


class preprocessing:

    def __init__(self):
        self.train_data = pd.read_csv(r'C:\Users\chinm\PycharmProjects\ISM\groundtruth_train.csv', index_col=False)
        self.im_path = r'C:\Users\chinm\PycharmProjects\ISM\ISIC_2019_Training_Input\ISIC_2019_Training_Input'
        self.imageName_set = self.train_data['image']
        self.train_data = self.train_data.drop(['image'], axis=1)
        Y_labels = []
        X_set = []
        new_list = []
        counter=0
        for i in range(self.imageName_set.shape[0]):

            image = cv2.imread(os.path.join(self.im_path, self.imageName_set[i] + '.jpg'))

            image = cv2.resize(image, (620, 620))
            array_image = np.asarray(image)
            array_image.flatten()
            img_array = array_image.reshape(-1, 1).T
            new_list.append(img_array)
            #so now i want to convert this to csv
            print("working")
            ind_lab = self.train_data.iloc[i].values
            Y_labels.append((np.where(ind_lab == 1))[0][0])
            counter=counter+1
            print(counter)
        x= np.asarray(new_list)
        self.Y_labels = np.asarray(Y_labels)
        X = pd.DataFrame(x)
        X.to_csv("train_Data.csv", index=False)
        Y = pd.Series(self.Y_labels)
        Y.to_csv("labels.csv", index=False)


    def read_csv(self):
        self.train_csv = pd.read_csv(self.csv_name)
        self.test_csv = pd.read_csv(os.path.join(self.csv_path, "groundtruth_val"))

    def prepess(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r, g, b = cv2.split(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        light_red = (0, 180, 180)
        dark_red = (10, 250, 255)
        light_orange = (10, 150, 200)
        dark_orange = (18, 255, 255)
        light_yellow = (18, 60, 140)
        dark_yellow = (30, 250, 255)

        light_green = (52, 75, 80)
        dark_green = (80, 255, 255)

        light_blue = (85, 50, 75)
        dark_blue = (125, 250, 255)

        light_ink = (100, 1, 50)
        dark_ink = (140, 220, 255)

        m = img.shape[0]
        n = img.shape[1]
        template = np.ones((m, n))
        template = template.astype(int)

        a = r.flatten()
        counts = np.bincount(a)
        r_max = np.argmax(counts)

        a = g.flatten()
        counts = np.bincount(a)
        g_max = np.argmax(counts)

        a = b.flatten()
        counts = np.bincount(a)
        b_max = np.argmax(counts)

        rn = template * r_max
        gn = template * g_max
        bn = template * b_max
        skin = np.dstack([rn, gn, bn])

        orange_mask = cv2.inRange(hsv, light_orange, dark_orange)
        red_mask = cv2.inRange(hsv, light_red, dark_red)
        yellow_mask = cv2.inRange(hsv, light_yellow, dark_yellow)
        green_mask = cv2.inRange(hsv, light_green, dark_green)
        blue_mask = cv2.inRange(hsv, light_blue, dark_blue)
        ink_mask = cv2.inRange(hsv, light_ink, dark_ink)
        # Creates a Black mask for the colored region
        om = cv2.bitwise_not(orange_mask)
        rm = cv2.bitwise_not(red_mask)
        ym = cv2.bitwise_not(yellow_mask)
        gm = cv2.bitwise_not(green_mask)
        bm = cv2.bitwise_not(blue_mask)
        im = cv2.bitwise_not(ink_mask)
        # Extracts a SkinPatch in the color pattern
        skinPatch = cv2.bitwise_and(skin, skin, mask=orange_mask)
        spr = cv2.bitwise_and(skin, skin, mask=red_mask)
        spy = cv2.bitwise_and(skin, skin, mask=yellow_mask)
        spg = cv2.bitwise_and(skin, skin, mask=green_mask)
        spb = cv2.bitwise_and(skin, skin, mask=blue_mask)
        spi = cv2.bitwise_and(skin, skin, mask=ink_mask)

        # replaces the Color pixel with black and then with the skin color
        imnew = cv2.bitwise_and(img, img, mask=om)
        rst = cv2.add(imnew, skinPatch, dtype=cv2.CV_8UC1)

        rst = cv2.bitwise_and(rst, rst, mask=rm)
        rst = cv2.add(rst, spr, dtype=cv2.CV_8UC1)

        rst = cv2.bitwise_and(rst, rst, mask=ym)
        rst = cv2.add(rst, spy, dtype=cv2.CV_8UC1)

        rst = cv2.bitwise_and(rst, rst, mask=gm)
        rst = cv2.add(rst, spg, dtype=cv2.CV_8UC1)

        rst = cv2.bitwise_and(rst, rst, mask=bm)
        rst = cv2.add(rst, spb, dtype=cv2.CV_8UC1)

        rst = cv2.bitwise_and(rst, rst, mask=im)
        rst = cv2.add(rst, spi, dtype=cv2.CV_8UC1)

        rst = rst.astype(np.uint8)

        Z = rst.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 13
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        kmeans_img = res.reshape((rst.shape))

        grayScale = cv2.cvtColor(kmeans_img, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(1, (17, 17))
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

        ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        thresh3 = cv2.adaptiveThreshold(blackhat, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,
                                        2)  # Adaptive gaussian
        thresh4 = cv2.adaptiveThreshold(blackhat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,
                                        2)  # Adaptive gaussian

        # inpaint the original image depending on the mask
        dst = cv2.inpaint(kmeans_img, thresh2, 10, cv2.INPAINT_TELEA)

        return dst


if __name__ == '__main__':
    pre = preprocessing()
