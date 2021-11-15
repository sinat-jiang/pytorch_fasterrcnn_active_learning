"""
使用 GMM 模型对图像数据进行聚类
"""
import pickle
from PIL import Image
from numpy import *
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
import operator
import sys
sys.path.append("..")
import filetools


M = 720    # 图像的 height
N = 1280   # 图像的 width

K = 5   # 聚类中心数
MAIN_COMP = 25  # 选取的主成分数(注意有多少数据，原始数据矩阵和投影矩阵就有多少行，而主成分数 < 投影矩阵行数)
# 主成分不同，聚类结果可能也有所差别

# PCA 参数模型
# PICKLE_MODEL = 'font_pca_modes.pkl'
PICKLE_MODEL = 'military_pca_modes.pkl'

# 获取 selected-fontimages 文件下图像文件名，并保存在列表中
# path = "E:\\test\\numbers"
path = r"E:\AllDateSets\MilitaryOB_5_class_torch_2th\unlabel_pool\images"
imlist = filetools.get_all_file_list(path)
imnbr = len(imlist)

# ------------------------------------更新 MAIN_COMP，固定为数据量的 60% 左右------------------------------------
MAIN_COMP = int(imnbr * 0.6)
print('MAIN_COMP =', MAIN_COMP)

# 载入模型文件
with open(PICKLE_MODEL, 'rb') as f:
    immean = pickle.load(f)     # 所有样本图像的均值
    V = pickle.load(f)      # 投影矩阵（原图像数据经过该投影矩阵转换到新坐标系中）
# print(V, V.shape)   # 最后一行全是 NAN， shape = (25, 250000)

# 创建矩阵，存储所有拉成一维形式后的图像
# 这里也不要忘了调整图像大小（防止不一致带来的问题）
m, n = Image.open(imlist[0]).size
immatrix = array([array(Image.open(im).convert('L').resize((m, n), Image.ANTIALIAS)).flatten() for im in imlist], 'f')

# 投影到前 MAIN_COMP 个主成分上，后面用这 MAIN_COMP 个主成分进行聚类
# （相当于将原来的 mxn 维数据转为了 MAIN_COMP 维数据）
immean = immean.flatten()
projected = array([dot(V[:MAIN_COMP], immatrix[i]-immean) for i in range(imnbr)])
# print(projected, projected.shape)   # projected.shape = (40, MAIN_COMP)

# 进行 GMM 聚类
gmm = GMM(n_components=K).fit(projected)    # 指定聚类中心个数为 K
labels = gmm.predict(projected)     # 预测标签
# scores = gmm._estimate_weighted_log_prob(projected)     # 返回当前数据属于每个高斯模型的 log-probabilities 权重（用来衡量数据离聚类中心的距离）
# score_samples 直接返回属于当前高斯模型的 log-probabilities，
# 与 _estimate_weighted_log_prob 相比少了属于其他高斯模型的对数概率权重，需要配合 labels 使用来判断当前属于哪个子模型
samples_score = gmm.score_samples(projected)
print(samples_score)
print(labels)
# print(scores)


class Img:
    def __init__(self, score, label, name, onedimdata):
        self.score = score
        self.label = label
        self.name = name
        self.onedimdata = onedimdata


# 按离聚类中心远近排列数据（越近越靠前），并挑选出每一聚类中距离中心最近的 30% 数据
nearest_data = []
for k in range(K):
    li = []
    for i in range(len(labels)):
        if labels[i] == k:
            li.append(Img(samples_score[i], labels[i], imlist[i], immatrix[i]))
            # print(image.score, image.name, image.label)
    # 按 scores 排序
    cmpfun = operator.attrgetter('score')  # 参数为排序依据的属性，可以有多个，这里优先 score，使用时按需求改换参数即可
    li.sort(key=cmpfun)  # 使用时改变列表名即可
    li.reverse()
    # 看看排序结果
    for kk in li:
        print(kk.score, kk.name, kk.label)
    # 划重点#划重点#划重点----排序操作
    print('-----------------------------------')
    nearest_data.append(li)
# print(nearest_data)

print('==========================前30%数据===========================')
# 选取前 30% 个数据
nearest_data30 = []
for li in nearest_data:
    length = len(li)
    nearest30 = int(length * 0.3)
    l30 = []
    for i in range(nearest30):
        l30.append(li[i])
    nearest_data30.append(l30)
# 看看 nearest_data30 是否按从大到小的顺序存储了距离各自聚类中心最近的前 30% 的数据
for li in nearest_data30:
    for li2 in li:
        print(li2.score, li2.label, li2.name)


# 绘制所有数据的聚类簇
for k in range(K):
    plt.figure()
    plt.gray()
    count = 1
    for i in range(len(labels)):
        if labels[i] == k:
            plt.subplot(12, 12, count)
            plt.imshow(immatrix[i].reshape((n, m)))     # 这里的 reshape 中的参数 先是 width，再是 height
            plt.axis('off')
            plt.title(imlist[i].split('\\')[-1])
            count += 1
# plt.show()

# 绘制按距离聚类中心距离从近到远的聚类簇（取30%）
for k in range(K):
    plt.figure()
    plt.gray()
    count = 1
    for i in range(len(nearest_data30[k])):
        plt.subplot(6, 10, count)
        plt.imshow(nearest_data30[k][i].onedimdata.reshape((n, m)))
        plt.axis('off')
        plt.title(nearest_data30[k][i].name.split('\\')[-1])
        count += 1
plt.show()

# if __name__ == '__main__':
#     pass
