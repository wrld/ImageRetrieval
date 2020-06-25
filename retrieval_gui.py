from PIL import Image, ImageTk  # 导入图像处理函数库
import tkinter as tk  # 导入GUI界面函数库
from tkinter import filedialog
import retrieval as ret
from tkinter import ttk
# 创建窗口 设定大小并命名
window = tk.Tk()
window.title('图像检索系统')
window.geometry('1000x750')
mycolor = '#FFFFFF'
window.configure(bg=mycolor)
global img_png  # 定义全局变量 图像的
global filename
var = tk.StringVar()  # 这时文字变量储存器


def train():
    import os

    os.system(
        "/usr/bin/python3 /home/gjx/opencv/open/retrieval/retrieval.py -feature_method "
        + combobox.get() + " -retrieval_method " + combobox2.get() +
        " -train 1")


def verify():
    import os

    os.system(
        "/usr/bin/python3 /home/gjx/opencv/open/retrieval/retrieval.py -feature_method "
        + combobox.get() + " -retrieval_method " + combobox2.get() +
        " -verify 1")


def retrieval():
    import os
    global filename

    os.system(
        "/usr/bin/python3 /home/gjx/opencv/open/retrieval/retrieval.py -feature_method "
        + combobox.get() + " -retrieval_method " + combobox2.get() +
        " -image_path " + filename)


def fileDialog():
    global img_png
    global filename
    filename = filedialog.askopenfilename(
        initialdir="/home/gjx/visual-struct/dataset/verify/",
        title="Select A File",
        filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    label = tk.Label(window, text="")
    # label.grid(column=1, row=2)
    label.configure(text=filename)
    Img = Image.open(filename)
    img_png = ImageTk.PhotoImage(Img)
    var.set('已显示')  # 设置标签的文字为 'you hit me'
    label_Img = tk.Label(window, image=img_png)
    label_Img.pack()
    label_Img.place(x=300, y=300, anchor='nw')


Label_Show = tk.Label(window,
                      text='欢迎使用图像检索系统，请选择算法与图像',
                      bg='#FFC0CB',
                      font=('Arial', 12),
                      width=40,
                      height=2)
Label_Show.pack(side="top", padx=0, pady=10)
Label_Show = tk.Label(window,
                      text='请选择特征提取算法：',
                      bg=mycolor,
                      font=('Arial', 10),
                      width=40,
                      height=2)
Label_Show.pack()
Label_Show.place(x=80, y=48, anchor='nw')
Label_Show = tk.Label(window,
                      text='请选择检索算法：',
                      bg=mycolor,
                      font=('Arial', 10),
                      width=40,
                      height=2)
Label_Show.pack()
Label_Show.place(x=80, y=72, anchor='nw')
mynumber = tk.StringVar()
combobox = ttk.Combobox(window, width=15, height=5, textvariable=mynumber)
combobox['values'] = ("sift", "orb")
combobox.pack()
combobox.current(0)

mynumber = tk.StringVar()
combobox2 = ttk.Combobox(window, width=15, textvariable=mynumber)
combobox2['values'] = ("svm", "kd_tree", "decision_tree", "random_forest")
combobox2.pack()
combobox2.current(0)

btn_Open = tk.Button(
    window,
    text='打开图像',  # 显示在按钮上的文字
    font=('Arial', 15),
    bg='#FFC0CB',
    width=15,
    height=2,
    command=fileDialog)  # 点击按钮式执行的命令
btn_Open.pack(ipadx=10, ipady=10, padx=10, pady=10)  # 按钮位置
btn_Open.place(x=250, y=100, anchor='nw')
# 创建显示图像按钮
btn_run = tk.Button(
    window,
    text='检索图像',  # 显示在按钮上的文字
    font=('Arial', 15),
    bg='#FFC0CB',
    width=15,
    height=2,
    command=retrieval)  # 点击按钮式执行的命令
btn_run.pack(ipadx=10, ipady=10, padx=10, pady=20)  # 按钮位置
btn_run.place(x=600, y=100, anchor='nw')

btn_ver = tk.Button(
    window,
    text='训练集训练',  # 显示在按钮上的文字
    font=('Arial', 15),
    bg='#FFC0CB',
    width=15,
    height=2,
    command=verify)  # 点击按钮式执行的命令
btn_ver.pack(ipadx=10, ipady=10, padx=10, pady=20)  # 按钮位置
btn_ver.place(x=250, y=200, anchor='nw')
btn_train = tk.Button(
    window,
    text='验证集测试',  # 显示在按钮上的文字
    font=('Arial', 15),
    bg='#FFC0CB',
    width=15,
    height=2,
    command=verify)  # 点击按钮式执行的命令
btn_train.pack(ipadx=10, ipady=10, padx=10, pady=20)  # 按钮位置
btn_train.place(x=600, y=200, anchor='nw')
window.mainloop()
