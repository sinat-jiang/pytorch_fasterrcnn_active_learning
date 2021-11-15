"""
计算图片的主成分 并 保存主成分相关模型参数。
"""
# from PIL import Image
# from numpy import *
# from pylab import *
# import matplotlib; matplotlib.use('TkAgg')    # 常用的有图形界面显示的终端名称为 Qt5Agg（）
from matplotlib.pylab import plt
import pickle
from . import pca
import sys
sys.path.append("..")
import filetools


# # Pickle 模型保存名称
# # PICKLE_MODEL = "flowers5_pca_modes.pkl"
# # PICKLE_MODEL = "font_pca_modes.pkl"
# PICKLE_MODEL = "military_pca_modes.pkl"

# # 获取图片路径
# # path = "E:\\test\\numbers"
# # path = "E:\\test\\flowers_5"
# path = "E:\\test\\vcupro"
# imlist = file_tools.get_all_file_list(path)


def getimageshape(imglist):
    """
    返回图像列表中的图像尺寸和列表中的图像数量
    :param imglist:
    :return:
    """
    # 获取图像的尺寸
    im = array(Image.open(imglist[0]).convert('L'))
    m, n = im.shape[0:2]     # 获取图像的大小
    print(m, n)
    imnbr = len(imglist)     # 获取图像的数目
    print("length =", imnbr)
    # 返回图像的长、宽、数据集长度
    return m, n, imnbr


def getmaincomp(immatrix):
    """
    通过执行 PCA 操作获取主成分
    :return:
    """
    # 执行 PCA 操作
    # V:投影矩阵；   S:方差；   immean：所有图像的均值；
    V, S, immean = pca.pca(immatrix)
    print(V, V.shape)
    # 返回 投影矩阵V、方差S、图像数据均值immean
    return V, S, immean


# 展示一下主成分选取后，经过降维的图像
def showing(m, n, immean, V):
    # 显示一些图像（均值图像和前 7 个模式）
    plt.figure()
    plt.gray()  # 灰度
    plt.subplot(2, 4, 1)
    plt.imshow(immean.reshape(m, n))    # 平均化的图像
    for i in range(7):
        plt.subplot(2, 4, i+2)
        plt.imshow(V[i].reshape(m, n))  # 投影矩阵
    plt.show()


def savepikmodel(pickle_model, immean, V):
    # 保存均值和主成分数据
    # with open(pickle_model, "wb") as f:
    #     pickle.dump(immean, f, protocol=4)
    #     pickle.dump(V, f, protocol=4)

    # 报错：OverflowError: cannot serialize a bytes object larger than 4 GiB
    # 解决：添加 protocol=4 参数
    f = open(pickle_model, 'wb')
    pickle.dump(immean, f, protocol=4)
    pickle.dump(V, f, protocol=4)
    f.close()

    # 另附加载模型示例代码：
    # 载入均值和主成分数据
    # f = open('font_pca_modes.pkl', 'rb')
    # immean = pickle.load(f)
    # V = pickle.load(f)
    # f.close()


if __name__ == '__main__':
    # Pickle 模型保存名称
    pickle_model = "military_pca_modes3.pkl"

    # 获取图片路径
    path = r"E:\AllDateSets\MilitaryOB_5_class_torch_2th\unlabel_pool\images"
    imlist = filetools.get_all_file_list(path)
    print(len(imlist))

    # 创建矩阵，保存所有压平后的图像数据
    # 可能的报错：ValueError: setting an array element with a sequence.此时说明图片的大小可能不一致，需调整
    # 这里就简单的调整为第一张图片的大小
    m, n = Image.open(imlist[0]).size
    immatrix = array([array(Image.open(im).convert('L').resize((m, n), Image.ANTIALIAS)).flatten() for im in imlist], 'f')
    print(immatrix)

    # 获取图片的长宽和 imlist 长度
    m, n, imnbr = getimageshape(imlist)

    # 主成分分析（PCA），返回 投影矩阵、方差、数据均值
    V, S, immean = getmaincomp(immatrix)

    # 保存参数（主要是保存 数据均值 和 投影矩阵）
    savepikmodel(pickle_model, immean, V)

    # 展示
    showing(m, n, immean, V)

