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


class Img:
    def __init__(self, score, label, name, onedimdata):
        self.score = score
        self.label = label
        self.name = name
        self.onedimdata = onedimdata


def project(data_path, pickle_model, main_comp=25):
    """
    读取之前 pca 操作获得的数据集参数，并将数据投影到新的高维空间，供 GMM 聚类使用
    :param data_path: 数据集路径
    :param main_comp: 主成分数（默认为25），主成分数应小于数据维度
    :return:
    """
    # 获取数据列表
    imlist = filetools.get_all_file_list(data_path)
    imnbr = len(imlist)

    # ------------------------------------更新 MAIN_COMP，固定为数据量的 60% 左右------------------------------------
    main_comp = int(imnbr * 0.6)
    print('MAIN_COMP =', main_comp)

    # 载入模型文件
    with open(pickle_model, 'rb') as f:
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
    projected = array([dot(V[:main_comp], immatrix[i]-immean) for i in range(imnbr)])
    # print(projected, projected.shape)   # projected.shape = (40, MAIN_COMP)
    return projected, imlist, immatrix


def project_2th(data_list, pickle_model, m, n, main_comp=25):
    """
    读取之前 pca 操作获得的数据集参数，并将数据投影到新的高维空间，供 GMM 聚类使用
    :param data_list: 数据列表
    :param main_comp: 主成分数（默认为25），主成分数应小于数据维度
    :return:
    """
    # 获取数据列表
    imlist = data_list
    imnbr = len(imlist)

    print('MAIN_COMP =', main_comp)

    # 载入模型文件
    with open(pickle_model, 'rb') as f:
        immean = pickle.load(f)     # 所有样本图像的均值
        V = pickle.load(f)      # 投影矩阵（原图像数据经过该投影矩阵转换到新坐标系中）
    # print(V, V.shape)   # 最后一行全是 NAN， shape = (25, 250000)

    # 创建矩阵，存储所有拉成一维形式后的图像
    # 还要记得调整图像大小
    immatrix = array([array(Image.open(im).convert('L').resize((m, n), Image.ANTIALIAS)).flatten() for im in imlist], 'f')

    # 投影到前 MAIN_COMP 个主成分上，后面用这 MAIN_COMP 个主成分进行聚类
    # （相当于将原来的 mxn 维数据转为了 MAIN_COMP 维数据）
    immean = immean.flatten()
    projected = array([dot(V[:main_comp], immatrix[i]-immean) for i in range(imnbr)])
    # print(projected, projected.shape)   # projected.shape = (40, MAIN_COMP)
    return projected, imlist, immatrix


def GMM_Cluster(projected, K):
    """
    GMM 聚类
    :return:
    """
    # 进行 GMM 聚类
    gmm = GMM(n_components=K).fit(projected)    # 指定聚类中心个数为 K
    labels = gmm.predict(projected)     # 预测标签
    # scores = gmm._estimate_weighted_log_prob(projected)     # 返回当前数据属于每个高斯模型的 log-probabilities 权重（用来衡量数据离聚类中心的距离）
    # score_samples 直接返回属于当前高斯模型的 log-probabilities，
    # 与 _estimate_weighted_log_prob 相比少了属于其他高斯模型的对数概率权重，需要配合 labels 使用来判断当前属于哪个子模型
    samples_score = gmm.score_samples(projected)
    # print(samples_score)
    # print(labels)
    # print(scores)
    return labels, samples_score


def GMM_Cluster_and_cal(projected, train_data_projected, K):
    """
    GMM 聚类，并计算新数据在 GMM 模型中距离各聚类中心的距离
    :param train_data_projected: 已经转化到高维空间的 train 集数据
    :param projected: 已经转化到高维空间的新数据
    :return:
    """
    # 进行 GMM 聚类
    gmm = GMM(n_components=K).fit(train_data_projected)    # 指定聚类中心个数为 K
    labels_train = gmm.predict(train_data_projected)
    labels = gmm.predict(projected)     # 预测标签
    # scores = gmm._estimate_weighted_log_prob(projected)     # 返回当前数据属于每个高斯模型的 log-probabilities 权重（用来衡量数据离聚类中心的距离）
    # score_samples 直接返回属于当前高斯模型的 log-probabilities，
    # 与 _estimate_weighted_log_prob 相比少了属于其他高斯模型的对数概率权重，需要配合 labels 使用来判断当前属于哪个子模型
    samples_score = gmm.score_samples(projected)
    # print(samples_score)
    print(labels)
    # print(scores)
    return labels, labels_train, samples_score


def chose_kernal_data(labels, samples_score, imlist, immatrix, K, near_ratio=0.3):
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
        # # 看看排序结果
        # for kk in li:
        #     print(kk.score, kk.name, kk.label)
        # 划重点#划重点#划重点----排序操作
        # print('-----------------------------------')
        nearest_data.append(li)
    # print(nearest_data)

    # print('==========================前30%数据===========================')
    # 选取前 30% 个数据
    nearest_data30 = []
    for li in nearest_data:
        length = len(li)
        nearest30 = int(length * near_ratio)
        l30 = []
        for i in range(nearest30):
            l30.append(li[i])
        nearest_data30.append(l30)
    # 看看 nearest_data30 是否按从大到小的顺序存储了距离各自聚类中心最近的前 30% 的数据
    # for li in nearest_data30:
    #     for li2 in li:
    #         print(li2.score, li2.label, li2.name)
    return nearest_data, nearest_data30


def chose_kernal_data_2th(labels, samples_score, imlist, immatrix, K, near_ratio=0.3):
    # 按离聚类中心远近排列数据（越近越靠前），并挑选出每一聚类中距离中心最近的 30% 数据
    # 注意，相交于之前的函数，这里不再是单独选取每个聚类簇中的前30%，而是仅仅按照距离各自聚类簇的远近来排序来选取整体的前30%
    nearest_data = []
    for i in range(len(labels)):
        nearest_data.append(Img(samples_score[i], labels[i], imlist[i], immatrix[i]))
        # 按 scores 排序
        cmpfun = operator.attrgetter('score')  # 参数为排序依据的属性，可以有多个，这里优先 score，使用时按需求改换参数即可
        nearest_data.sort(key=cmpfun)  # 使用时改变列表名即可
        nearest_data.reverse()
    # 看看排序结果
    for kk in nearest_data:
        print(kk.score, kk.name, kk.label)
    # 划重点#划重点#划重点----排序操作
    print('-----------------------------------')

    print('==========================前30%数据===========================')
    # 选取前 30% 个数据
    nearest_data30 = nearest_data[:int(len(nearest_data) * near_ratio)]
    # 看看 nearest_data30 是否按从大到小的顺序存储了距离各自聚类中心最近的前 30% 的数据
    for li in nearest_data30:
        print(li.score, li.label, li.name)
    return nearest_data, nearest_data30


def chose_kernal_data_3th(labels, samples_score, imlist, immatrix, K, near_M):
    # 按离聚类中心远近排列数据（越近越靠前），并挑选出每一聚类中距离中心最近的 near_M 个数据
    # 注意，相交于之前的函数，这里不再是单独选取每个聚类簇中的前 near_M 个，而是仅仅按照距离各自聚类簇的远近来排序来选取整体的前 near_M 个
    nearest_data = []
    for i in range(len(labels)):
        nearest_data.append(Img(samples_score[i], labels[i], imlist[i], immatrix[i]))
        # 按 scores 排序
        cmpfun = operator.attrgetter('score')  # 参数为排序依据的属性，可以有多个，这里优先 score，使用时按需求改换参数即可
        nearest_data.sort(key=cmpfun)  # 使用时改变列表名即可
        nearest_data.reverse()
    # 看看排序结果
    for kk in nearest_data:
        print(kk.score, kk.name, kk.label)
    # 划重点#划重点#划重点----排序操作
    print('-----------------------------------')

    print('==========================前30%数据===========================')
    # 选取前 30% 个数据
    nearest_data_M = nearest_data[:near_M]
    # 看看 nearest_data30 是否按从大到小的顺序存储了距离各自聚类中心最近的前 30% 的数据
    for li in nearest_data_M:
        print(li.score, li.label, li.name)
    return nearest_data, nearest_data_M


def show(labels, imlist, immatrix, nearest_data30, K):
    m, n = Image.open(imlist[0]).size
    # 绘制所有数据的聚类簇
    for k in range(K):
        plt.figure()
        plt.gray()
        count = 1
        for i in range(len(labels)):
            if labels[i] == k:
                plt.subplot(10, 10, count)
                plt.imshow(immatrix[i].reshape((n, m)))     # 这里的 reshape 中的参数 先是 width，再是 height
                plt.axis('off')
                plt.title(imlist[i].split('\\')[-1])
                count += 1
    # plt.show()

    # 绘制按距离聚类中心距离从近到远的聚类簇
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


def show_2th(labels, imlist, immatrix, nearest_data30, K, m, n):
    # 绘制所有数据的聚类簇
    for k in range(K):
        plt.figure()
        plt.gray()
        count = 1
        for i in range(len(labels)):
            if labels[i] == k:
                plt.subplot(20, 20, count)
                plt.imshow(immatrix[i].reshape((n, m)))     # 这里的 reshape 中的参数 先是 width，再是 height
                plt.axis('off')
                plt.title(imlist[i].split('\\')[-1])
                count += 1
    # plt.show()

    # 绘制按距离聚类中心距离从近到远的聚类簇
    for k in range(K):
        plt.figure()
        plt.gray()
        count = 1
        for i in range(len(nearest_data30)):
            if nearest_data30[i].label == k:
                plt.subplot(10, 10, count)
                plt.imshow(nearest_data30[i].onedimdata.reshape((n, m)))
                plt.axis('off')
                plt.title(nearest_data30[i].name.split('\\')[-1])
                count += 1
    plt.show()


if __name__ == '__main__':
    M = 720  # 图像的 height
    N = 1280  # 图像的 width

    # 加载 pca 参数，并将数据集投影到新的高维空间，得到原始数据的高维投影形式
    data_path = r"E:\AllDateSets\MilitaryOB_5_class_torch_2th\unlabel_pool\images"
    pickle_model = 'military_pca_modes.pkl'     # PCA 参数模型
    projected, imlist, immatrix = project(data_path, pickle_model, main_comp=25)   # 默认 25，会在函数内根据数据量自行调整

    # 聚类，得到数据属于哪个高斯子模型的标签 以及 其在自身的子模型中的置信度分数（可以用来衡量距离中心的远近）
    K = 5  # 聚类中心数
    labels, samples_score = GMM_Cluster(projected, K)

    # 将每一个子模型的数据单独排序，并返回距离各自中心最近的一批数据
    near_ = 0.4     # 靠近中心的 30% 数据
    nearest_data, nearest_data30 = chose_kernal_data(labels, samples_score, imlist, immatrix, K, near_)

    # 展示聚类情况，以及各个子模型中靠近中心的数据
    show(labels, imlist, immatrix, nearest_data30, K)
