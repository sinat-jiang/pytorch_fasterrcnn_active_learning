import sys
sys.path.append("..")
from PIL import Image
from numpy import *
from sklearn.mixture import GaussianMixture as GMM
from . import maincomponentcal
from . import GMMclusting
import filetools


def pca_opt(dataset_path, m, n, if_show, pickle_model="military_pca_modes.pkl"):
    """
    对数据集进行 pca 操作，并将计算得到的参数存成模型
    :param dataset_path: 数据集的路径
    :param if_show: 是否画图展示
    :param pickle_model: 参数模型名称
    :return:
    """
    # 获取图片 list
    imlist = filetools.get_all_file_list(dataset_path)
    print("pca's main comp numbers:", len(imlist))

    # 创建矩阵，保存所有压平后的图像数据

    immatrix = array([array(Image.open(im).convert('L').resize((m, n), Image.ANTIALIAS)).flatten() for im in imlist],
                     'f')
    print(immatrix)

    # 获取图片的长宽和 imlist 长度
    # m_, n_, imnbr = maincomponentcal.getimageshape(imlist)

    # 主成分分析（PCA），返回 投影矩阵、方差、数据均值
    V, S, immean = maincomponentcal.getmaincomp(immatrix)

    # 保存参数（主要是保存 数据均值 和 投影矩阵）
    # pickle_model：模型名称
    maincomponentcal.savepikmodel(pickle_model, immean, V)

    # 展示
    if if_show:
        maincomponentcal.showing(m, n, immean, V)


def GMMClusting_opt(data_path, pickle_model, near_, K, main_comp=25, if_show=False):
    """
    使用高斯混合模型进行聚类，并选取靠近聚类中心的数据
    :param K: 聚类中心数
    :param MAIN_COMP: 选取的主成分数(注意有多少数据，原始数据矩阵和投影矩阵就有多少行，而主成分数 < 投影矩阵行数)
                      主成分不同，聚类结果可能也有所差别
    :param pickle_model: PCA 参数模型
    :param near_: 靠近中心的前百分之多少的数据
    :param dataset_path: 数据集路径
    :param if_show: 是否将聚类结果简单的展示出来
    :return:
    """
    # 加载 pca 参数，并将数据集投影到新的高维空间，得到原始数据的高维投影形式
    projected, imlist, immatrix = GMMclusting.project(data_path, pickle_model, main_comp)  # 默认 25，会在函数内根据数据量自行调整

    # 聚类，得到数据属于哪个高斯子模型的标签 以及 其在自身的子模型中的置信度分数（可以用来衡量距离中心的远近）
    # K = 5  # 聚类中心数
    labels, samples_score = GMMclusting.GMM_Cluster(projected, K)

    # 将每一个子模型的数据单独排序，并返回距离各自中心最近的一批数据
    # near_ ：靠近中心的 xx% 数据
    nearest_data, nearest_data_near = GMMclusting.chose_kernal_data(labels, samples_score, imlist, immatrix, K, near_)

    # 展示聚类情况，以及各个子模型中靠近中心的数据
    if if_show:
        GMMclusting.show(labels, imlist, immatrix, nearest_data_near, K)

    return nearest_data, nearest_data_near


def GMMClusting_opt_2th(data_list, pickle_model, near_, K, train_data_list, main_comp=25, if_show=False):
    """
    使用高斯混合模型进行聚类，并选取靠近聚类中心的数据（依据不确定性策略挑选出的数据）
    这里是对 train 数据集中的数据进行聚类得到 GMM 模型，然后将新数据输入 GMM 模型得到距离各聚类中心的距离
    :param K: 聚类中心数
    :param main_comp: 选取的主成分数(注意有多少数据，原始数据矩阵和投影矩阵就有多少行，而主成分数 < 投影矩阵行数)
                      主成分不同，聚类结果可能也有所差别
    :param pickle_model: PCA 参数模型
    :param near_: 靠近中心的前百分之多少的数据
    :param data_list: 要测试聚类结果的数据列表
    :param if_show: 是否将聚类结果简单的展示出来
    :param train_data_list: 要进行聚类的 train 集数据列表
    :return:
    """
    # 加载 pca 参数，并将数据集投影到新的高维空间，得到原始数据的高维投影形式
    # 这里也不要忘了调整图像大小（防止不一致带来的问题）
    m, n = Image.open(data_list[0]).size
    projected, imlist, immatrix = GMMclusting.project_2th(data_list, pickle_model, m, n, main_comp)
    train_data_projected, train_imlist, train_immatrix = GMMclusting.project_2th(train_data_list, pickle_model, m, n, main_comp)

    # train 集数据聚类，得到新数据属于哪个高斯子模型的标签 以及 其在自身的子模型中的置信度分数（可以用来衡量距离中心的远近）
    # K = 5  # 聚类中心数
    labels, labels_train, samples_score = GMMclusting.GMM_Cluster_and_cal(projected, train_data_projected, K)

    # 排序，并返回距离各自中心最近的一批数据
    # near_ ：靠近中心的 xx% 数据
    nearest_data, nearest_data_near = GMMclusting.chose_kernal_data_2th(labels, samples_score, imlist, immatrix, K, near_)

    # 展示聚类情况，以及各个子模型中靠近中心的数据
    if if_show:
        GMMclusting.show_2th(labels_train, train_imlist, train_immatrix, nearest_data_near, K, m, n)

    return nearest_data, nearest_data_near


def GMMClusting_opt_3th(data_list, pickle_model, M, K, train_data_list, main_comp=25, if_show=False):
    """
    使用高斯混合模型进行聚类，并选取靠近聚类中心的数据（依据不确定性策略挑选出的数据）
    这里是对 train 数据集中的数据进行聚类得到 GMM 模型，然后将新数据输入 GMM 模型得到距离各聚类中心的距离
    > 与上一个函数不同的是，这里的是直接指定图片数量，而不是根据比例计算得出
    :param K: 聚类中心数
    :param main_comp: 选取的主成分数(注意有多少数据，原始数据矩阵和投影矩阵就有多少行，而主成分数 < 投影矩阵行数)
                      主成分不同，聚类结果可能也有所差别
    :param pickle_model: PCA 参数模型
    :param M: 靠近中心的数据数量
    :param data_list: 要测试聚类结果的数据列表
    :param if_show: 是否将聚类结果简单的展示出来
    :param train_data_list: 要进行聚类的 train 集数据列表
    :return:
    """
    # 加载 pca 参数，并将数据集投影到新的高维空间，得到原始数据的高维投影形式
    # 这里也不要忘了调整图像大小（防止不一致带来的问题）
    m, n = Image.open(data_list[0]).size
    projected, imlist, immatrix = GMMclusting.project_2th(data_list, pickle_model, m, n, main_comp)
    train_data_projected, train_imlist, train_immatrix = GMMclusting.project_2th(train_data_list, pickle_model, m, n, main_comp)

    # train 集数据聚类，得到新数据属于哪个高斯子模型的标签 以及 其在自身的子模型中的置信度分数（可以用来衡量距离中心的远近）
    # K = 5  # 聚类中心数
    labels, labels_train, samples_score = GMMclusting.GMM_Cluster_and_cal(projected, train_data_projected, K)

    # 排序，并返回距离各自中心最近的一批数据
    # M ：靠近中心的 M 个数据
    nearest_data, nearest_data_near = GMMclusting.chose_kernal_data_3th(labels, samples_score, imlist, immatrix, K, M)

    # 展示聚类情况，以及各个子模型中靠近中心的数据
    if if_show:
        GMMclusting.show_2th(labels_train, train_imlist, train_immatrix, nearest_data_near, K, m, n)

    return nearest_data, nearest_data_near


if __name__ == '__main__':
    # 1 - 进行 pca 并保存参数
    path = r'E:\AllDateSets\MilitaryOB_5_class_torch_3th\train\JPEGImages'
    list_p = filetools.get_all_file_list(r"E:\AllDateSets\MilitaryOB_5_class_torch_3th\unlabel_pool\images")[50:70]
    m, n = Image.open(list_p[0]).size
    pca_opt(path, m, n, if_show=True, pickle_model='military_pca_modes2.pkl')

    # 2 - GMM Clusting and choose those img near the kernal of class
    data_path = r"E:\AllDateSets\MilitaryOB_5_class_torch_3th\train\JPEGImages"
    pickle_model = 'military_pca_modes2.pkl'  # PCA 参数模型
    K = 5   # 聚类数
    main_comp = 25  # 主成分数（默认为25）
    if_show = True  # 是否将聚类结果画出来
    near_ = 0.4
    data_list = filetools.get_all_file_list(r"E:\AllDateSets\MilitaryOB_5_class_torch_3th\unlabel_pool\images")[50:70]
    data_train_list = filetools.get_all_file_list(data_path)
    main_comp = int(len(data_list) * 0.9)
    nearest_data, nearest_data_near = GMMClusting_opt_2th(data_list, pickle_model, near_, K, data_train_list, main_comp, if_show)
    for i in range(len(nearest_data)):
        for k in nearest_data[i]:
            print(k.name, k.score, k.label)
    print("---------------------------------------------------")
    for i in range(len(nearest_data_near)):
        for k in nearest_data_near[i]:
            print(k.name, k.score, k.label)
    print("---------------------------------------------------")

    data_list = filetools.get_all_file_list(data_path)
    data_train_list = filetools.get_all_file_list(data_path)
    main_comp = int(len(data_list) * 0.9)
    nearest_data, nearest_data_near = GMMClusting_opt_2th(data_list, pickle_model, near_, K, data_train_list, main_comp,
                                                          if_show)
    for i in range(len(nearest_data_near)):
        for k in nearest_data_near[i]:
            print(k.name, k.score, k.label)

