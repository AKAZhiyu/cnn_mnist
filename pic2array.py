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
    temp = dataset - np.tile(dataset.min(), dataset.shape)
    maxmatrix = np.tile(temp.max(), dataset.shape)
    return temp / maxmatrix


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def getimage_array(filename):
    img = Image.open(filename)
    img_array = np.array(img)
    if img_array.ndim == 3:
        img_array = rgb2gray(img_array)
    img_array = img_array.flatten()
    img_array = 1 - Normalization(img_array)
    return img_array


def JudgeEdge(img_array):
    height = len(img_array)
    width = len(img_array[0])
    size = [-1, -1, -1, -1]
    for i in range(height):
        high = img_array[i]
        low = img_array[height - 1 - i]
        if len(high[high > 0]) > 0 and size[0] == -1:
            size[0] = i
        if len(low[low > 0]) > 0 and size[1] == -1:
            size[1] = height - 1 - i
        if size[1] != -1 and size[0] != -1:
            break
    for i in range(width):
        left = img_array[:, i]
        right = img_array[:, width - 1 - i]
        if len(left[left > 0]) > 0 and size[2] == -1:
            size[2] = i
        if len(right[right > 0]) > 0 and size[3] == -1:
            size[3] = width - i - 1
        if size[2] != -1 and size[3] != -1:
            break
    return size


def JudgeOneNumber(img_array):
    edge = [-1, -1]
    width = len(img_array[0])
    for i in range(width):
        left = img_array[:, i]
        right = img_array[:, width - 1 - i]
        if len(left[left > 0]) > 0 and edge[0] == -1:
            edge[0] = i
        if len(right[right > 0]) > 0 and edge[1] == -1:
            edge[1] = width - i - 1
        if edge[0] != -1 and edge[1] != -1:
            break
    for j in range(edge[0], edge[1] + 1):
        border = img_array[:, j]
        if len(border[border > 0]) == 0:
            return False
    return True


def SplitPicture(img_array, img_list):
    if JudgeOneNumber(img_array):
        img_list.append(img_array)
        return img_list
    width = len(img_array[0])
    for i in range(width):
        left_border = img_array[:, i]
        right_border = img_array[:, i + 1]
        if len(left_border[left_border > 0]) > 0 and len(right_border[right_border > 0]) == 0:
            break
    return_array = img_array[:, 0:i + 1]
    img_list.append(return_array)
    new_array = img_array[:, i + 1:]
    return SplitPicture(new_array, img_list)


# 读取图片，包括图片灰度化、剪裁、压缩
def GetCutZip(imagename):
    img = Image.open(imagename)
    img_array = np.array(img)
    # RGB图灰度化
    if img_array.ndim == 3:
        img_array = rgb2gray(img_array)
    # 归一化，提高数字与背景的对比度
    img_array = Normalization(img_array)
    # 白底黑字图像转化为黑底白字
    arr1 = (img_array >= 0.9)
    arr0 = (img_array <= 0.1)
    if arr1.sum() > arr0.sum():
        img_array = 1 - img_array
    # 消除部分噪音，便于提取数字
    img_array[img_array > 0.7] = 1
    img_array[img_array < 0.4] = 0
    img_list = SplitPicture(img_array, [])
    final_list = []
    for img_array in img_list:
        edge = JudgeEdge(img_array)
        cut_array = img_array[edge[0]:edge[1] + 1, edge[2]:edge[3] + 1]
        cut_img = Image.fromarray(np.uint8(cut_array * 255))
        if cut_img.size[0] <= cut_img.size[1]:
            zip_img = cut_img.resize((20 * cut_img.size[0] // cut_img.size[1], 20), Image.LANCZOS)
        else:
            zip_img = cut_img.resize((20, 20 * cut_img.size[1] // cut_img.size[0]), Image.LANCZOS)
        zip_img_array = np.array(zip_img)
        final_array = np.zeros((28, 28))
        height = len(zip_img_array)
        width = len(zip_img_array[0])
        high = (28 - height) // 2
        left = (28 - width) // 2
        final_array[high:high + height, left:left + width] = zip_img_array
        final_array = Normalization(final_array)
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
    # printSegmentedImages(img_list)
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