# coding: utf-8

import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA  #加载PCA算法包
from sklearn.datasets import load_iris
from sklearn.neighbors import NearestNeighbors
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


def getImageInput(img_paths, train_file, allname=None):
    '''
    获取聚类中心
    
    img_paths:图像数据中所有图像路径
    dataset_matrix：图像数据的矩阵表示   注：img_paths dataset_matrix这两个参数只需要指定一个
    num_words:聚类中心数
    '''
    all_class = []
    des_list = []  # 特征描述
    addr_list = []
    if feature_method == "orb":
        des_matrix = np.zeros((1, 32))
    elif feature_method == "sift":
        des_matrix = np.zeros((1, 128))
    else:
        raise ValueError('Error input')
    response = np.float32([])
    sumcount = 0
    if allname == None:
        training_names = os.listdir(img_paths)
        print(training_names)
    else:
        training_names = allname
    for i in range(len(training_names)):
        file_path = img_paths + training_names[i] + "/"
        file_paths = os.listdir(file_path)
        count = 0
        for path in file_paths:
            img = cv2.imread(img_paths + training_names[i] + "/" + path)
            if feature_method == "orb":
                kp, des = orb.detectAndCompute(img, None)
            elif feature_method == "sift":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                kp, des = sift_det.detectAndCompute(img, None)
            else:
                raise ValueError('Error input')
            if des.any() != None:
                des_matrix = np.row_stack((des_matrix, des))
            des_list.append(des)
            addr_list.append(img_paths + training_names[i] + "/" + path)
            # response = np.append(response, np.float32([count]))
            count += 1
        res = np.repeat(np.float32([i]), count)
        response = np.append(response, res)
        sumcount += count
        all_class.append(sumcount)
        print("Done" + str(i))

    print(all_class)
    des_matrix = des_matrix[1:, :]  # the des matrix of sift

    response = response.reshape(-1, 1)
    response = response.astype(int)
    np.save(train_file, (response, all_class, addr_list))

    return des_matrix, des_list


def getClusters(des_matrix, num_words, train_file):
    response, all_class, addr_list = np.load(train_file)
    # kmeans = KMeans(n_clusters=num_words, random_state=33)
    # kmeans.fit(des_matrix)
    # centers = kmeans.cluster_centers_  # 视觉聚类中心
    kmeans = KMeans(n_clusters=num_words, n_jobs=4,
                    random_state=None).fit(des_matrix)
    centers = kmeans.cluster_centers_
    np.save(train_file, (response, centers, all_class, addr_list))
    print("Complete kmeans")


# 将特征描述转换为特征向量
def des2feature(des, num_words, centures):
    '''
    des:单幅图像的SIFT特征描述
    num_words:视觉单词数/聚类中心数
    centures:聚类中心坐标   num_words*128
    return: feature vector 1*num_words
    '''
    img_feature_vec = np.zeros((1, num_words), 'float32')
    for i in range(des.shape[0]):
        if feature_method == "orb":
            feature_k_rows = np.ones((num_words, 32), 'float32')
        elif feature_method == "sift":
            feature_k_rows = np.ones((num_words, 128), 'float32')
        else:
            raise ValueError('Error input')
        # print(feature_k_rows.shape)
        feature = des[i]
        # print("1")
        # print(feature)
        # print("2")
        # print(feature_k_rows)
        feature_k_rows = feature_k_rows * feature
        # print(feature_k_rows)
        feature_k_rows = np.sum((feature_k_rows - centures)**2, 1)
        # print(feature_k_rows)
        index = np.argmin(feature_k_rows)
        img_feature_vec[0][index] += 1
    # print(feature)
    # print(feature_k_rows)
    return img_feature_vec


def get_all_features(des_list, num_words, filename,
                     train_centers=np.array([])):
    # 获取所有图片的特征向量

    if train_centers.any():
        centers = train_centers
        response, all_class, addr_list = np.load(filename)
    else:
        response, centers, all_class, addr_list = np.load(filename)
    allvec = np.zeros((len(des_list), num_words), 'float32')
    for i in range(len(des_list)):
        if des_list[i].all() != None:
            allvec[i] = des2feature(centures=centers,
                                    des=des_list[i],
                                    num_words=num_words)

    # svm = cv2.ml.SVM_create()
    # svm.train(np.array(allvec), cv2.ml.ROW_SAMPLE,
    #           np.array(response))  #None, None, None)  # select best params
    # svm.save("svmnew.clf")

    np.save(filename, (response, centers, allvec, all_class, addr_list))
    print("Get all features")
    return allvec


# #### 经过之前的操作，我们已经成功通过词袋表示法将SIFT提取的特征表示出来
# #### 接下来计算待检索图像最近邻图像


def getNearestImg(feature, dataset, num_close):
    '''
    找出目标图像最像的几个
    feature:目标图像特征
    dataset:图像数据库
    num_close:最近个数
    return:最相似的几个图像
    '''
    features = np.ones((dataset.shape[0], len(feature)), 'float32')
    print(feature.shape)
    features = features * feature
    print(features.shape)
    # print(dataset)

    dist = np.sum((features - dataset)**2, 1)
    # print(dist)
    dist_index = np.argsort(dist)
    return dist_index[:num_close]


def showImg(target_img_path, index, filename):
    '''
    target_img:要搜索的图像
    dataset_paths：图像数据库所有图片的路径
    显示最相似的图片集合
    '''
    # get img path

    response, centers, img_dataset, all_class, addr_list = np.load(filename)
    paths = []
    for i in index[0]:
        paths.append(addr_list[i])
        print(i)
    print(paths)
    plt.figure(figsize=(10, 20))  #  figsize 用来设置图片大小
    plt.subplot(432), plt.imshow(
        plt.imread(target_img_path)), plt.title('target_image')

    for i in range(len(index[0])):
        plt.subplot(4, 3, i + 4), plt.imshow(plt.imread(paths[i]))
    plt.show()


def showImg_kmeans(target_img_path, index, class_index, filename):
    '''
    target_img:要搜索的图像
    dataset_paths：图像数据库所有图片的路径
    显示最相似的图片集合
    '''
    # get img path

    response, centers, img_dataset, all_class, addr_list = np.load(filename)
    paths = []
    if class_index > 0:
        for i in index:
            paths.append(addr_list[all_class[class_index - 1] + i])
    else:
        for i in index:
            paths.append(addr_list[i])
    plt.figure(figsize=(10, 20))  #  figsize 用来设置图片大小
    plt.subplot(432), plt.imshow(
        plt.imread(target_img_path)), plt.title('target_image')

    for i in range(len(index)):
        plt.subplot(4, 3, i + 4), plt.imshow(plt.imread(paths[i]))
    plt.show()


def retrieval_svm(img_path, img_paths, filename):
    '''
    检索图像，找出最像的几个
    img:待检索的图像
    img_dataset:图像数据库 matrix
    num_close:显示最近邻的图像数目
    centures:聚类中心
    img_paths:图像数据库所有图像路径
    '''

    response, centers, img_dataset, all_class, addr_list = np.load(filename)
    num_close = 6
    img = cv2.imread(img_path)
    if feature_method == "orb":
        kp, des = orb.detectAndCompute(img, None)
    elif feature_method == "sift":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = sift_det.detectAndCompute(img, None)
    else:
        raise ValueError('Error input')
    feature = des2feature(des=des, centures=centers, num_words=num_words)

    case = np.float32(feature)
    np.array(case, np.float32)
    from sklearn.svm import SVC
    classifier = SVC(C=20)

    classifier.fit(np.array(img_dataset), np.array(response))
    y_pred = classifier.predict(case)
    print(classifier.score(img_dataset, response))
    from sklearn.metrics import classification_report
    print(allname[y_pred[0]])
    result_index = y_pred[0]
    if result_index > 0:
        sorted_index = getNearestImg(
            feature,
            img_dataset[all_class[result_index - 1]:all_class[result_index]],
            num_close)
    else:
        sorted_index = getNearestImg(feature, img_dataset[0:all_class[0]],
                                     num_close)
    showImg_kmeans(img_path, sorted_index, result_index, filename)


def retrieval_rf(img_path, img_paths, filename):
    response, centers, img_dataset, all_class, addr_list = np.load(filename)
    num_close = 6
    img = cv2.imread(img_path)
    if feature_method == "orb":
        kp, des = orb.detectAndCompute(img, None)
    elif feature_method == "sift":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = sift_det.detectAndCompute(img, None)
    else:
        raise ValueError('Error input')
    feature = des2feature(des=des, centures=centers, num_words=num_words)

    case = np.float32(feature)
    np.array(case, np.float32)
    model_rf = RandomForestClassifier(n_estimators=len(all_class),
                                      max_depth=3,
                                      random_state=1234)  # 1234随机初始化的种子

    model_rf.fit(np.array(img_dataset), np.array(response))  # 训练数据集
    y_pred = model_rf.predict(case)
    print(allname[y_pred[0]])
    result_index = y_pred[0]
    if result_index > 0:
        sorted_index = getNearestImg(
            feature,
            img_dataset[all_class[result_index - 1]:all_class[result_index]],
            num_close)
    else:
        sorted_index = getNearestImg(feature, img_dataset[0:all_class[0]],
                                     num_close)
    showImg_kmeans(img_path, sorted_index, result_index, filename)


def getNearestImg_global(feature, dataset, num_close):
    '''
    找出目标图像最像的几个
    feature:目标图像特征
    dataset:图像数据库
    num_close:最近个数
    return:最相似的几个图像
    '''
    features = np.ones((dataset.shape[0], len(feature)), 'float32')
    features = features * feature
    dist = np.sum((features - dataset)**2, 1)
    dist_index = np.argsort(dist)
    return dist_index[:num_close]


def retrieval_global(img_path, filename):
    '''
    检索图像，找出最像的几个
    img:待检索的图像
    img_dataset:图像数据库 matrix
    num_close:显示最近邻的图像数目
    centures:聚类中心
    img_paths:图像数据库所有图像路径
    '''
    response, centers, img_dataset, all_class, addr_list = np.load(filename)
    num_close = 6
    img = cv2.imread(img_path)

    if feature_method == "orb":
        kp, des = orb.detectAndCompute(img, None)
    elif feature_method == "sift":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = sift_det.detectAndCompute(img, None)
    else:
        raise ValueError('Error input')
    feature = des2feature(des=des, centures=centers, num_words=num_words)
    nbrs = NearestNeighbors(n_neighbors=num_close,
                            algorithm='kd_tree').fit(img_dataset)
    distances, indices = nbrs.kneighbors(feature)
    # sorted_index = getNearestImg_global(feature, img_dataset, num_close)
    print(indices)
    showImg(img_path, indices, filename)


def verify(test_path, filename, num_words):
    response, centers, img_dataset, all_class, addr_list = np.load(filename)
    from sklearn.svm import SVC
    classifier = SVC(C=20)
    classifier.fit(np.array(img_dataset), np.array(response))

    des_matrix, des_list = getImageInput(img_paths=test_path,
                                         train_file="test.npy",
                                         allname=allname)
    response, all_class, addr_list = np.load("test.npy")
    # getClusters(des_matrix=des_matrix,
    #             num_words=num_words,
    #             train_file="test.npy")
    img_features = get_all_features(des_list=des_list,
                                    num_words=num_words,
                                    filename="test.npy",
                                    train_centers=centers)

    y_pred = classifier.predict(img_features)
    print(classifier.score(img_features, response))
    from sklearn.metrics import classification_report
    print(classification_report(response, y_pred))
    return img_features


ap = argparse.ArgumentParser()
ap.add_argument("-feature_method",
                required=True,
                help="Method how to extract feature")
ap.add_argument("-retrieval_method",
                required=True,
                help="Method how to search images")
ap.add_argument("-image_path",
                required=False,
                help="Path of image to retrieval")
ap.add_argument("-verify", required=False, help="if verify")
ap.add_argument("-train", required=False, help="if train")
args = vars(ap.parse_args())

feature_method = args["feature_method"]
retrieval_method = args["retrieval_method"]

orb = cv2.ORB_create()
sift_det = cv2.xfeatures2d.SIFT_create()
training_path = '/home/gjx/visual-struct/dataset/train/'  #训练样本文件夹路径
verify_path = '/home/gjx/visual-struct/dataset/verify/'
allname = [
    'airplane', 'elephant', 'face', 'pills', 'car', 'horse', 'gun', 'dragon',
    'dog', 'women', 'bus', 'google_logo', 'poke', 'cat', 'accordion', 'brain'
]
num_words = len(allname)  # 聚类中心数
if args["train"] != None:
    des_matrix, des_list = getImageInput(img_paths=training_path,
                                         train_file="train.npy",
                                         allname=allname)

    getClusters(des_matrix=des_matrix,
                num_words=num_words,
                train_file="train.npy")

    img_features = get_all_features(des_list=des_list,
                                    num_words=num_words,
                                    filename="train.npy")
if args["image_path"] != None:
    path = args["image_path"]
else:
    path = '/home/gjx/visual-struct/dataset/verify/airplane/image_0390.jpg'

if retrieval_method == "svm":
    retrieval_svm(path, training_path, "train.npy")
elif retrieval_method == "kd_tree":
    retrieval_global(path, "train.npy")
elif retrieval_method == "random_forest":
    retrieval_rf(path, training_path, "train.npy")
else:
    raise ValueError('Error input')

if args["verify"] != None:
    verify(verify_path, "train.npy", num_words)
# retrieval_img('/home/gjx/visual-struct/dataset/verify/airplane/airplane20.jpg',
#               training_path)
# retrieval_img('/home/gjx/visual-struct/dataset/train/banana/banana11.jpg',
#               training_path)
# retrieval_img('/home/gjx/visual-struct/dataset/verify/airplane/airplane14.jpg',
#               training_path)
# pic = plt.imread(path)
# plt.figure(figsize=(10, 20))
# plt.imshow(pic)
# plt.show()