# coding: utf-8
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.neighbors import NearestNeighbors
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.datasets.samples_generator import make_classification
import time
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus


def getImageInput(img_paths, train_file, allname=None):
    '''
    输入图像预处理，提取特征，制作标签
    '''
    all_class = []
    des_list = []
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
            #使用不同特征提取算法
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
            count += 1
        res = np.repeat(np.float32([i]), count)
        response = np.append(response, res)
        sumcount += count
        all_class.append(sumcount)
        print("Done" + str(i))

    print(all_class)
    des_matrix = des_matrix[1:, :]

    response = response.reshape(-1, 1)
    response = response.astype(int)
    np.save(train_file, (response, all_class, addr_list))

    return des_matrix, des_list


def getClusters(des_matrix, num_words, train_file):
    '''
    kmeans聚类
    '''
    response, all_class, addr_list = np.load(train_file)
    kmeans = KMeans(n_clusters=num_words, n_jobs=4,
                    random_state=None).fit(des_matrix)
    centers = kmeans.cluster_centers_
    np.save(train_file, (response, centers, all_class, addr_list))
    print("Complete kmeans")


def des2feature(des, num_words, centures):
    '''
    将特征描述转换为词袋模型的特征向量
    '''
    img_feature_vec = np.zeros((1, num_words), 'float32')
    for i in range(des.shape[0]):
        if feature_method == "orb":
            feature_k_rows = np.ones((num_words, 32), 'float32')
        elif feature_method == "sift":
            feature_k_rows = np.ones((num_words, 128), 'float32')
        else:
            raise ValueError('Error input')
        feature = des[i]
        feature_k_rows = feature_k_rows * feature
        feature_k_rows = np.sum((feature_k_rows - centures)**2, 1)
        index = np.argmin(feature_k_rows)
        img_feature_vec[0][index] += 1
    return img_feature_vec


def get_all_features(des_list, num_words, filename,
                     train_centers=np.array([])):
    '''
    获取所有图片的词袋模型特征向量
    '''
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

    np.save(filename, (response, centers, allvec, all_class, addr_list))
    print("Get all features")
    return allvec


def getNearestImg(feature, dataset, num_close):
    '''
    基于欧氏距离，找出目标图像最像的几个
    '''
    features = np.ones((dataset.shape[0], len(feature)), 'float32')
    print(feature.shape)
    features = features * feature
    print(features.shape)

    dist = np.sum((features - dataset)**2, 1)
    dist_index = np.argsort(dist)
    return dist_index[:num_close]


def showImg(target_img_path, index, filename):
    '''
    对于全局搜索算法，显示图像
    '''
    end = time.clock()
    print('Running time: %s Seconds' % (end - start))
    response, centers, img_dataset, all_class, addr_list = np.load(filename)
    paths = []
    for i in index[0]:
        paths.append(addr_list[i])
        print(i)
    print(paths)
    plt.figure(figsize=(10, 20))
    plt.subplot(432), plt.imshow(
        plt.imread(target_img_path)), plt.title('target_image')

    for i in range(len(index[0])):
        plt.subplot(4, 3, i + 4), plt.imshow(plt.imread(paths[i]))
    plt.show()


def showImg_kmeans(target_img_path, index, class_index, filename):
    '''
    对于图像分类算法，显示图像
    '''
    end = time.clock()
    print('Running time: %s Seconds' % (end - start))
    response, centers, img_dataset, all_class, addr_list = np.load(filename)
    paths = []
    if class_index > 0:
        for i in index:
            paths.append(addr_list[all_class[class_index - 1] + i])
    else:
        for i in index:
            paths.append(addr_list[i])
    plt.figure(figsize=(10, 20))
    plt.subplot(432), plt.imshow(
        plt.imread(target_img_path)), plt.title("Detect result: " +
                                                allname[class_index] +
                                                "   running time:" +
                                                str(end - start) + "seconds")
    for i in range(len(index)):
        plt.subplot(4, 3, i + 4), plt.imshow(plt.imread(paths[i]))
    plt.show()


def retrieval(img_path, img_paths, filename):
    '''
    通过分类器分类+检索图像
    方法有svm, random_forest, decision_tree
    '''

    response, centers, img_dataset, all_class, addr_list = np.load(filename)
    num_close = 6
    img = cv2.imread(img_path)
    if feature_method == "orb":
        kp, des = orb.detectAndCompute(img, None)
        # img = cv2.drawKeypoints(img, kp, img, color=(0, 255, 0))
    elif feature_method == "sift":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = sift_det.detectAndCompute(img, None)
        # img = cv2.drawKeypoints(img, kp, img)
    else:
        raise ValueError('Error input')

    feature = des2feature(des=des, centures=centers, num_words=num_words)
    case = np.float32(feature)
    np.array(case, np.float32)
    if retrieval_method == "svm":
        from sklearn.svm import SVC
        classifier = SVC(C=20)

        classifier.fit(np.array(img_dataset), response.ravel())
        y_pred = classifier.predict(case)
        print(classifier.score(img_dataset, response))
        from sklearn.metrics import classification_report
        print(allname[y_pred[0]])
        result_index = y_pred[0]
    elif retrieval_method == "random_forest":
        model_rf = RandomForestClassifier(n_estimators=len(all_class),
                                          max_depth=35,
                                          random_state=1234)  # 1234随机初始化的种子
        model_rf.fit(np.array(img_dataset), response.ravel())  # 训练数据集
        y_pred = model_rf.predict(case)
        print(allname[y_pred[0]])
        result_index = y_pred[0]
    elif retrieval_method == "decision_tree":
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=30)
        # Train Decision Tree Classifer
        clf = clf.fit(img_dataset, response.ravel())
        y_pred = clf.predict(case)
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
    features = np.ones((dataset.shape[0], len(feature)), 'float32')
    features = features * feature
    dist = np.sum((features - dataset)**2, 1)
    dist_index = np.argsort(dist)
    return dist_index[:num_close]


def retrieval_global(img_path, filename):
    '''
    基于kd_tree全局检索图像，找到最相似的几个
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
    showImg(img_path, indices, filename)


def verify(test_path, filename, num_words):
    '''
    测试集验证训练效果
    '''
    start_2 = time.clock()
    response_train, centers, img_dataset, all_class, addr_list = np.load(
        filename)
    des_matrix, des_list = getImageInput(img_paths=test_path,
                                         train_file="test.npy",
                                         allname=allname)
    response_test, all_class, addr_list = np.load("test.npy")
    img_features = get_all_features(des_list=des_list,
                                    num_words=num_words,
                                    filename="test.npy",
                                    train_centers=centers)
    if retrieval_method == "svm":
        from sklearn.svm import SVC
        classifier = SVC(C=20)  #22
        classifier.fit(np.array(img_dataset), np.array(response_train))

        y_pred = classifier.predict(img_features)
        print(classifier.score(img_features, response_test))

    elif retrieval_method == "random_forest":
        model_rf = RandomForestClassifier(n_estimators=len(all_class),
                                          max_depth=35,
                                          random_state=1234)
        model_rf.fit(np.array(img_dataset), np.array(response_train))  # 训练数据集
        y_pred = model_rf.predict(img_features)

    elif retrieval_method == "decision_tree":
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=40)
        clf = clf.fit(np.array(img_dataset), np.array(response_train))
        y_pred = clf.predict(img_features)
        # dot_data = StringIO()
        # export_graphviz(clf,
        #                 out_file=dot_data,
        #                 filled=True,
        #                 rounded=True,
        #                 special_characters=True)
        # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        # graph.write_png('diabetes.png')
        # Image(graph.create_png())
    # draw the result
    end_2 = time.clock()
    print('verify Running time: %s Seconds' % (end_2 - start_2))
    from sklearn.metrics import classification_report
    print(classification_report(response_test, y_pred))
    print("Accuracy:", metrics.accuracy_score(response_test, y_pred))
    from sklearn.metrics import precision_recall_fscore_support as score

    precision, recall, fscore, support = score(response_test, y_pred)

    # print('precision: {}'.format(precision))
    # print('recall: {}'.format(recall))
    # print('fscore: {}'.format(fscore))
    # print('support: {}'.format(support))
    plt.plot(range(0, 15),
             precision,
             color="r",
             linestyle="-",
             marker="^",
             linewidth=1,
             label="precision")
    plt.plot(range(0, 15),
             recall,
             color="b",
             linestyle="-",
             marker="s",
             linewidth=1,
             label="recall")
    plt.plot(range(0, 15),
             fscore,
             color="g",
             linestyle="-",
             marker="*",
             linewidth=1,
             label="fscore")
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.title(retrieval_method)
    plt.xlabel("classes")
    plt.show()
    return img_features


def diminish(features):
    '''
    PCA降维
    '''
    from sklearn.decomposition import PCA
    pca = PCA(n_components=8)
    pca.fit(features)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    features_new = pca.transform(features)
    return features_new


if __name__ == '__main__':
    #对于不同的指令输入参数，相应选择不同的算法或操作
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
    #训练集路径
    training_path = '/home/gjx/visual-struct/dataset/train/'
    #验证集路径
    verify_path = '/home/gjx/visual-struct/dataset/verify/'
    allname = [
        'sunflower', 'elephant', 'face', 'pills', 'car', 'horse', 'dolphin',
        'dragon', 'dog', 'tools', 'bus', 'google_logo', 'poke', 'cat', 'brain'
    ]

    #根据算法选择相应的模型文件
    if feature_method == "sift":
        train_name = "train_sift_50_3.npy"
        num_words = 50  # 聚类中心数
    elif feature_method == "orb":
        train_name = "train_orb_50_3.npy"
        num_words = 25  # 聚类中心数

    if args["train"] != None:
        des_matrix, des_list = getImageInput(img_paths=training_path,
                                             train_file=train_name,
                                             allname=allname)

        getClusters(des_matrix=des_matrix,
                    num_words=num_words,
                    train_file=train_name)

        img_features = get_all_features(des_list=des_list,
                                        num_words=num_words,
                                        filename=train_name)
    if args["image_path"] != None:
        path = args["image_path"]
    else:
        path = '/home/gjx/visual-struct/dataset/verify/poke/39_3911.jpg'
    start = time.clock()
    if retrieval_method == "svm":
        retrieval(path, training_path, train_name)
    elif retrieval_method == "kd_tree":
        retrieval_global(path, train_name)
    elif retrieval_method == "random_forest":
        retrieval(path, training_path, train_name)
    elif retrieval_method == "decision_tree":
        retrieval(path, training_path, train_name)
    else:
        raise ValueError('Error input')

    if args["verify"] != None:
        verify(verify_path, train_name, num_words)
