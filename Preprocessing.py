import pandas as pd
import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import hog
from numpy import genfromtxt
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import hsv_to_rgb
from numpy import asarray
from SVM_Classifier import *

class preprocessing:

    def __init__(self):
        train_data = pd.read_csv(r'F:\TUhh\Sem 5\Project\ISIC_2019_Training_GroundTruth.csv', index_col=False)
        im_path = r'F:\TUhh\Sem 5\Project\ISIC_2019_Training_Input\ISIC_2019_Training_Input'
        imageName_set = train_data['image']
        train_data = train_data.drop(['image'], axis=1)
        Y_labels = []
        X_set = []
        for i in range(imageName_set.shape[0]):
            image_path = os.path.join(im_path, imageName_set[i] + '.jpg')
            image = cv2.imread(image_path)
            try:
                #premask = self.apply_discmasking(image)
                dst = self.apply_dullrazor(image)
                med = self.median_filter(dst)
                gray = cv2.cvtColor(med, cv2.COLOR_RGB2GRAY)
                gray = cv2.resize(gray, (620, 620))
                hog_features = hog(gray, block_norm='L2-Hys', pixels_per_cell=(16, 16))
                flat_features = np.hstack(hog_features)
            except:
                continue

            X_set.append(flat_features)
            ind_lab = train_data.iloc[i].values
            Y_labels.append((np.where(ind_lab==1))[0][0])

        self.X_set = np.array(X_set)
        self.Y_labels = np.array(Y_labels)

    def median_filter(self, image):
        median = cv2.medianBlur(image, 5)
        return median

    def segmentation(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        return thresh

    def svm_classifier(self,train,lables):
        SVC_classifier = SVC(C=10.0, gamma=0.05)
        SVC_classifier.fit(train, lables)
        #svc_preds = SVC_classifier.predict(scaled_test)

    def apply_discmasking(self, image):
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

        return rst

    def apply_dullrazor(self, image):
        grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(1, (17, 17))
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

        ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        thresh3 = cv2.adaptiveThreshold(blackhat, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,
                                        2)  # Adaptive gaussian
        thresh4 = cv2.adaptiveThreshold(blackhat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,
                                        2)  # Adaptive gaussian

        # inpaint the original image depending on the mask
        dst = cv2.inpaint(image, thresh2, 10, cv2.INPAINT_TELEA)

        return dst

    def apply_kmeans(self, image):

        Z = image.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 15
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        kmeans_img = res.reshape((image.shape))
        # cv2.imshow("Kmeans img", kmeans_img)
        # cv2.waitKey(0)
        return kmeans_img

    def apply_AHE(self, image):
        clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # cv2.imshow("HSV", hsv)
        # cv2.waitKey(0)

        h, s, v = cv2.split(hsv)

        h1 = clahe.apply(h)
        s1 = clahe.apply(s)
        v1 = clahe.apply(v)

        lab = cv2.merge((h1, s1, v1))
        # cv2.imshow("Lab", lab)
        # cv2.waitKey(0)
        
        enhanced_img = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        # cv2.imshow("Enhanced", enhanced_img)
        # cv2.waitKey(0)
        
        return hsv, lab, enhanced_img

    def grabcut_mask(self, image, enhancedimg):
        hsv_img = cv2.cvtColor(enhancedimg, cv2.COLOR_BGR2HSV)

        lower_green = np.array([50,100,100])
        higher_green = np.array([100,255,255])
        mask = cv2.inRange(hsv_img, lower_green, higher_green)
        # cv2.imshow("Mask", mask)
        # cv2.waitKey(0)

        ret, inv_mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY_INV)
        # cv2.imshow("Inv Mask", inv_mask)
        # cv2.waitKey(0)

        res = cv2.bitwise_and(image, image, mask=mask)
        # cv2.imshow("Res", res)
        # cv2.waitKey(0)

        new_mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        if (np.sum(inv_mask[:]) < 80039400):
            newmask = inv_mask
            new_mask[newmask == 0] = 0
            new_mask[newmask == 255] = 1
            dim = cv2.grabCut(image, new_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            mask2 = np.where((new_mask == 2) | (new_mask == 0), 0, 1).astype('uint8')
            GrabCut_img = image * mask2[:, :, np.newaxis]
            #cv2.imshow("Grab img", GrabCut_img)
            #cv2.waitKey(0)
        else:
            s = (int(image.shape[0] / 10), int(image.shape[1] / 10))
            rect = (s[0], s[1], int(image.shape[0] - (3 / 10) * s[0]), image.shape[1] - s[1])
            cv2.grabCut(enhancedimg, new_mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            GrabCut_img = image * mask2[:, :, np.newaxis]
            #cv2.imshow("Grab img", GrabCut_img)
            #cv2.waitKey(0)

            #plt.imshow(GrabCut_img)
            #plt.colorbar()
            #plt.show()

        imgmask = cv2.medianBlur(GrabCut_img, 5)
        ret, Segmented_mask = cv2.threshold(imgmask, 0, 255, cv2.THRESH_BINARY)
        #cv2.imshow("Seg Image", Segmented_mask)
        #cv2.waitKey(0)

        if (np.sum(inv_mask[:]) < 80039400):
            newmask = inv_mask
            new_mask[newmask == 0] = 0
            new_mask[newmask == 255] = 1
            dim = cv2.grabCut(image, new_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            mask2 = np.where((new_mask == 2) | (new_mask == 0), 0, 1).astype('uint8')
            GrabCut_img2 = image * mask2[:, :, np.newaxis]
            #cv2.imshow("Grab img", GrabCut_img)
            #cv2.waitKey(0)
        else:
            s = (int(image.shape[0] / 10), int(image.shape[1] / 10))
            rect = (s[0], s[1], int(image.shape[0] - (3 / 10) * s[0]), image.shape[1] - s[1])
            cv2.grabCut(enhancedimg, new_mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            GrabCut_img2 = image * mask2[:, :, np.newaxis]
            #cv2.imshow("Grab img", GrabCut_img)
            #cv2.waitKey(0)

        imgmask2 = cv2.medianBlur(GrabCut_img2, 5)
        ret, Segmented_mask2 = cv2.threshold(imgmask2, 0, 255, cv2.THRESH_BINARY)
        #plt.imshow(GrabCut_img2)
        #plt.colorbar()
        #plt.show()

        return GrabCut_img2
    
if __name__ == '__main__':
    trainSetPath = r'F:\TUhh\Sem 5\Project\input.csv'
    labelSetPath = r'F:\TUhh\Sem 5\Project\label.csv'
    modelPath = r'F:\TUhh\Sem 5\Project\svm_model.sav'
    svmTrain = SVM_Classify()
    if os.path.isfile(trainSetPath)==False:
        pre = preprocessing()
        X_train = pre.X_set
        Y_train = pre.Y_labels

        svmTrain.formatInputData(X_train)
        svmTrain.formatLabels(Y_train)
        svmTrain.saveTrainData(trainSetPath, Y_train)

    svmTrain.loadTrainData(trainSetPath, labelSetPath)
    svmTrain.trainModel(modelPath)
