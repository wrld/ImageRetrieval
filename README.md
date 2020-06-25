# README

郭佳昕

##  通过GUI界面运行

``` shell
python retrieval_gui.py
```

可以打开gui界面，相应选择要检索的图片、检索方法等。

## 通过命令行直接运行

可以查看帮助

``` shell
python retrieval.py -h
```

交互运行代码`

``` shell
python retrieval.py -feature_method sift -retrieval_method svm 
#验证集测试：-verify 1
#指定图片：-image_path xxx
#训练集训练：-train 1
#特征检测算法：sift orb
#检索算法：svm random_forest decision_tree
```



