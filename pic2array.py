import numpy as np
from PIL import Image
import torch_NN
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from emsemble import ensemble_predict
import sys
import argparse


def Normalization(dataset):
    # 将数据集中的每个元素减去该数据集的最小值
    temp = dataset - np.tile(dataset.min(), dataset.shape)
    # 找到temp中的最大值，并构建与原数据集相同形状的最大值矩阵
    maxmatrix = np.tile(temp.max(), dataset.shape)
    # 将temp中的每个元素除以最大值，完成归一化
    return temp / maxmatrix


def rgb2gray(rgb):
    # 使用公认的彩色到灰度的转换公式
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def getimage_array(filename):
    # 打开图像文件
    img = Image.open(filename)
    # 将图像转换为数组
    img_array = np.array(img)
    # 如果图像是RGB的，将其转换为灰度图
    if img_array.ndim == 3:
        img_array = rgb2gray(img_array)
    # 将图像数据展平
    img_array = img_array.flatten()
    # 对图像数据进行归一化，并取反，这常用于预处理阶段
    img_array = 1 - Normalization(img_array)
    return img_array


def JudgeEdge(img_array):
    """
    判断图像中非零像素区域的边缘位置。

    参数:
        img_array (np.array): 一个二维的图像数据数组，通常是二值化后的图像。

    返回:
        list: 包含四个整数的列表，分别代表上、下、左、右边缘的索引。
    """
    # 获取图像的高度和宽度
    height = len(img_array)
    width = len(img_array[0])
    # 初始化边缘位置列表
    size = [-1, -1, -1, -1]  # 分别记录上、下、左、右边缘的位置

    # 寻找上下边缘
    for i in range(height):
        high = img_array[i]
        low = img_array[height - 1 - i]
        # 检测上边缘
        if len(high[high > 0]) > 0 and size[0] == -1:
            size[0] = i
        # 检测下边缘
        if len(low[low > 0]) > 0 and size[1] == -1:
            size[1] = height - 1 - i
        # 如果已确定上下边缘，中断循环
        if size[1] != -1 and size[0] != -1:
            break

    # 寻找左右边缘
    for i in range(width):
        left = img_array[:, i]
        right = img_array[:, width - 1 - i]
        # 检测左边缘
        if len(left[left > 0]) > 0 and size[2] == -1:
            size[2] = i
        # 检测右边缘
        if len(right[right > 0]) > 0 and size[3] == -1:
            size[3] = width - i - 1
        # 如果已确定左右边缘，中断循环
        if size[2] != -1 and size[3] != -1:
            break

    # 返回边缘索引
    return size


def JudgeOneNumber(img_array):
    """
    判断图像数组中是否只包含一个数字。

    该函数通过检测图像数组中的空白列来判断是否只有一个数字。如果在数字的边缘之间的列中找到了完全为空的列，
    则认为图像中包含多个数字。

    参数:
        img_array (np.array): 一个二维的图像数据数组，通常是二值化后的图像。

    返回:
        bool: 如果图像只包含一个数字，返回True；如果包含多个数字，返回False。
    """
    # 初始化边缘位置
    edge = [-1, -1]
    width = len(img_array[0])
    # 寻找左右边缘
    for i in range(width):
        left = img_array[:, i]
        right = img_array[:, width - 1 - i]
        if len(left[left > 0]) > 0 and edge[0] == -1:
            edge[0] = i
        if len(right[right > 0]) > 0 and edge[1] == -1:
            edge[1] = width - i - 1
        if edge[0] != -1 and edge[1] != -1:
            break
    # 检查边缘之间是否存在空白列
    for j in range(edge[0], edge[1] + 1):
        border = img_array[:, j]
        if len(border[border > 0]) == 0:
            return False
    return True


def SplitPicture(img_array, img_list):
    """
    递归地分割图像数组，以确保每个子图像数组只包含一个数字。

    这个函数通过寻找分割点，即一列完全包含数字而下一列不包含任何数字的位置，来分割图像。
    分割后，如果子图像数组仍可能包含多个数字，则继续递归分割。

    参数:
        img_array (np.array): 需要分割的图像数据数组，通常是二值化后的图像。
        img_list (list): 存储已分割出的子图像数组的列表。

    返回:
        list: 包含所有分割后的子图像数组的列表，每个数组中只包含一个数字。
    """
    # 判断当前图像是否只包含一个数字
    if JudgeOneNumber(img_array):
        img_list.append(img_array)
        return img_list

    width = len(img_array[0])
    # 寻找分割点
    for i in range(width):
        left_border = img_array[:, i]
        right_border = img_array[:, i + 1]
        if len(left_border[left_border > 0]) > 0 and len(right_border[right_border > 0]) == 0:
            break

    # 根据分割点分割图像，并递归处理剩余部分
    return_array = img_array[:, 0:i + 1]
    img_list.append(return_array)
    new_array = img_array[:, i + 1:]
    return SplitPicture(new_array, img_list)


# 读取图片，包括图片灰度化、剪裁、压缩
def GetCutZip(imagename):
    # 读取图片文件
    img = Image.open(imagename)
    img_array = np.array(img)
    # 如果是RGB图像，转换为灰度图
    if img_array.ndim == 3:
        img_array = rgb2gray(img_array)
    # 归一化，提高对比度
    img_array = Normalization(img_array)
    # 转换图像为黑底白字，便于处理
    arr1 = (img_array >= 0.9)
    arr0 = (img_array <= 0.1)
    if arr1.sum() > arr0.sum():
        img_array = 1 - img_array
    # 噪声消除
    img_array[img_array > 0.7] = 1
    img_array[img_array < 0.4] = 0
    # 分割图像
    img_list = SplitPicture(img_array, [])
    final_list = []
    for img_array in img_list:
        # 获取边缘并剪裁
        edge = JudgeEdge(img_array)
        cut_array = img_array[edge[0]:edge[1] + 1, edge[2]:edge[3] + 1]
        cut_img = Image.fromarray(np.uint8(cut_array * 255))
        # 调整大小
        if cut_img.size[0] <= cut_img.size[1]:
            zip_img = cut_img.resize((20 * cut_img.size[0] // cut_img.size[1], 20), Image.LANCZOS)
        else:
            zip_img = cut_img.resize((20, 20 * cut_img.size[1] // cut_img.size[0]), Image.LANCZOS)
        # 将调整大小后的图像转换为numpy数组
        zip_img_array = np.array(zip_img)
        # 创建一个新的28x28像素的数组，用于放置调整后的图像
        final_array = np.zeros((28, 28))
        height = len(zip_img_array)
        width = len(zip_img_array[0])
        # 计算图像在新数组中的起始位置
        high = (28 - height) // 2
        left = (28 - width) // 2
        # 将图像数据复制到新数组的指定位置
        final_array[high:high + height, left:left + width] = zip_img_array
        # 对图像数据进行归一化处理
        final_array = Normalization(final_array)
        # 将处理完成的图像数组添加到final_list列表中
        final_list.append(final_array)
    return final_list



def printSegmentedImages(image_list):
    # 显示final_list中的图像
    for i, array in enumerate(image_list):
        plt.imshow(array, cmap='gray')
        plt.title(f'Image {i + 1}')
        plt.show()
    return


def SegmentedImages(image_path):
    img_list = GetCutZip(image_path)
    # 遍历数组，为每个 (28, 28) 的子数组创建一个二进制文件
    for i, array in enumerate(img_list):
        filename = f'image_{i}.bin'
        array.tofile(filename)
    return len(img_list)


def recognize(src):

    # nn = torch_NN.CNN().to("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.is_available():
    #     nn.load_state_dict(torch.load('model_epoch_40.pth'))
    # else:
    #     nn.load_state_dict(torch.load('model_epoch_40.pth', map_location=torch.device('cpu')))

    # import joblib
    # loaded_model = joblib.load('mnist_log_reg_model.joblib')

    img_list = GetCutZip(src)
    final_result = ''
    printSegmentedImages(img_list)
    for img_array in img_list:
        # img_array = img_array.flatten().reshape(1, 784)
        # result = loaded_model.predict(img_array)
        # final_result = final_result + str(result[0][0])
        # np.save('saved_array.npy', img_array)

        # img_array = img_array.flatten()
        # result = nn.predict_image_from_array(img_array)
        # final_result = final_result + str(result)

        result = ensemble_predict(img_array)
        final_result = final_result + str(result)
    return final_result


# if __name__ == "__main__":
#     # img_path = input("请输入图片路径:\n")
#     img_path = "to_recognize.png"
#     final_result = recognize(img_path)
#     print("识别的最终结果是:" + final_result)
def main():
    parser = argparse.ArgumentParser(description="处理图片的命令行工具")
    parser.add_argument("image_path", type=str, help="图片的路径")
    parser.add_argument("--recognize", action="store_true", help="识别图片并返回字符串")
    parser.add_argument("--segment", action="store_true", help="将图片分割成多个二进制文件")

    args = parser.parse_args()

    if args.recognize:
        result = recognize(args.image_path)
        print(result)

    if args.segment:
        num = SegmentedImages(args.image_path)
        print(num)


if __name__ == "__main__":
    main()