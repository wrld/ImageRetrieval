'''
  @author: Tiejian Zhang
  @email: zhangtj_pku@hotmail.com
  @method: SIFT -> k-means -> SVM
'''
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

TrainSetInfo = {
    "car": 60,
    # "city": 20,
    "dog": 34,
    # "earth": 15,
    # "fireworks": 20,
    "flower": 96,
    # "fruits": 20,
    # "glass": 20,
    # "gold": 15,
    "gun": 40,
    "airplane": 83,
    "sky": 92,
    "apple": 50,
    "banana": 15,
    "watermelon": 45
    # "worldcup": 40
}

TestSetInfo = {
    "car": 60,
    # "city": 20,
    "dog": 34,
    # "earth": 15,
    # "fireworks": 20,
    "flower": 96,
    # "fruits": 20,
    # "glass": 20,
    # "gold": 15,
    "gun": 40,
    "airplane": 83,
    "sky": 92,
    "apple": 50,
    "banana": 15,
    "watermelon": 16
    # "worldcup": 40
}
# training_path = '/home/gjx/visual-struct/dataset/mix'  #训练样本文件夹路径
# training_names = os.listdir(training_path)
# pic_names = [
#     'bmp', 'jpg', 'png', 'tiff', 'gif', 'pcx', 'tga', 'exif', 'fpx', 'svg',
#     'psd', 'cdr', 'pcd', 'dxf', 'ufo', 'eps', 'ai', 'raw', 'WMF'
# ]
# for name in training_names:
#     file_format = name.split('.')[-1]
#     if file_format not in pic_names:
#         training_names.remove(name)
# img_paths = []  # 所有图片路径
# for name in training_names:
#     img_path = os.path.join(training_path, name)
#     img_paths.append(img_path)


def showImg(target_img_path, index, dataset_paths, name):
    '''
    target_img:要搜索的图像
    dataset_paths：图像数据库所有图片的路径
    显示最相似的图片集合
    '''
    # get img path
    paths = []
    for i in range(index):
        paths.append(dataset_paths + str(name) + str(i) + ".jpg")

    plt.figure(figsize=(10, 20))  #  figsize 用来设置图片大小
    plt.subplot(432), plt.imshow(
        plt.imread(target_img_path)), plt.title('target_image')

    for i in range(index):
        plt.subplot(4, 3, i + 4), plt.imshow(plt.imread(paths[i]))
    plt.show()


def calcSiftFeature(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(200)  # max number of SIFT points is 200
    kp, des = sift.detectAndCompute(gray, None)
    return des


def calcFeatVec(features, centers):
    featVec = np.zeros((1, 50))
    for i in range(0, features.shape[0]):
        fi = features[i]
        diffMat = np.tile(fi, (50, 1)) - centers
        sqSum = (diffMat**2).sum(axis=1)
        dist = sqSum**0.5
        sortedIndices = dist.argsort()
        idx = sortedIndices[0]  # index of the nearest center
        featVec[0][idx] += 1
    return featVec


def initFeatureSet():
    for name, count in TrainSetInfo.items():
        dir = "/home/gjx/visual-struct/dataset/" + name + "/"
        featureSet = np.float32([]).reshape(0, 128)
        training_names = os.listdir(dir)
        img_paths = []  # 所有图片路径
        for myname in training_names:
            img_path = os.path.join(dir, myname)
            img_paths.append(img_path)

        # print "Extract features from TrainSet " + name + ":"
        for filename in img_paths:
            # filename = dir + name + " (" + str(i) + ").jpg"
            print(filename)
            img = cv2.imread(filename)
            des = calcSiftFeature(img)
            featureSet = np.append(featureSet, des, axis=0)

        featCnt = featureSet.shape[0]
        # print str(featCnt) + " features in " + str(count) + " images\n"

        # save featureSet to file
        filename = name + ".npy"
        np.save(filename, featureSet)


def learnVocabulary():
    wordCnt = 50
    for name, count in TrainSetInfo.items():
        filename = name + ".npy"
        features = np.load(filename)

        print("Learn vocabulary of ")
        #+ name + "..."
        # use k-means to cluster a bag of features
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20,
                    0.1)
        newflags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(features, wordCnt, None,
                                                  criteria, 20,
                                                  cv2.KMEANS_RANDOM_CENTERS)
        # ret,label,center=cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        # save vocabulary(a tuple of (labels, centers)) to file
        filename = name + ".npy"
        np.save(filename, (labels, centers))
        # print "Done\n"


def trainClassifier():
    trainData = np.float32([]).reshape(0, 50)
    response = np.float32([])

    dictIdx = 0
    for name, count in TrainSetInfo.items():
        dir = "/home/gjx/visual-struct/dataset/" + name + "/"
        filename = name + ".npy"
        labels, centers = np.load(filename)
        training_names = os.listdir(dir)
        img_paths = []  # 所有图片路径
        for myname in training_names:
            img_path = os.path.join(dir, myname)
            img_paths.append(img_path)
        print("Init training data of " + name + "...")
        for newname in img_paths:
            # filename = dir + name + " (" + str(i) + ").jpg"
            img = cv2.imread(newname)
            features = calcSiftFeature(img)
            featVec = calcFeatVec(features, centers)
            trainData = np.append(trainData, featVec, axis=0)
        print(trainData.shape)

        res = np.repeat(np.float32([dictIdx]), count)
        response = np.append(response, res)
        dictIdx += 1
        print("Done" + str(dictIdx))

    print("Now train svm classifier...")
    trainData = np.float32(trainData)
    # trainData = trainData.reshape(-1, 1)
    response = response.reshape(-1, 1)
    response = response.astype(int)
    # svm = cv2.SVM()
    # print(trainData)
    print(trainData.shape)
    print(response.shape)
    svm = cv2.ml.SVM_create()
    svm.train(np.array(trainData), cv2.ml.ROW_SAMPLE,
              np.array(response))  #None, None, None)  # select best params
    svm.save("svm.clf")
    print("Done\n")


def classify():
    svm = cv2.ml.SVM_load("svm.clf")

    total = 0
    correct = 0
    dictIdx = 0
    for name, count in TestSetInfo.items():
        crt = 0
        dir = "/home/gjx/visual-struct/dataset/verify/" + name + "/"
        labels, centers = np.load(name + ".npy")
        training_names = os.listdir(dir)
        img_paths = []  # 所有图片路径
        for myname in training_names:
            img_path = os.path.join(dir, myname)
            img_paths.append(img_path)
        print("Classify on TestSet " + name + ":")
        for filename in img_paths:
            # filename = dir + name + " (" + str(i) + ").jpg"

            img = cv2.imread(filename)
            features = calcSiftFeature(img)
            featVec = calcFeatVec(features, centers)
            case = np.float32(featVec)
            # print(case.shape)
            # print(case)

            np.array(case, np.float32)
            res = svm.predict(case)
            print(res[1][0][0])
            print(dictIdx)
            if (dictIdx == res[1][0][0]):
                crt += 1

        print("Accuracy: " + str(crt) + " / " + str(count) + "\n")
        total += count
        correct += crt
        dictIdx += 1
        # cv2.imshow("new", img)
        # print(name)
        # cv2.waitKey(10000)

    print("Total accuracy: " + str(correct) + " / " + str(total))
    origin_dir = "/home/gjx/visual-struct/dataset/" + name + "/"
    showImg(filename, 9, origin_dir, name)


if __name__ == "__main__":
    # initFeatureSet()
    # learnVocabulary()
    # trainClassifier()
    classify()
