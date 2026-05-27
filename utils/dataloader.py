import numpy as np
import imageio
import os
from PIL import Image, ImageEnhance, ImageOps, ImageFile
from torchvision import datasets, transforms
import torch
from scipy.io import loadmat
import random
from PIL import ImageFilter
from torchvision import utils as vutils
import imageio

# trsfm = transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class DataAugmentation:
    """
    包含数据增强的八种方式
    """

    def __init__(self):
        pass

    @staticmethod
    def openImage(image,label,size=128):
        image = Image.open(image, mode="r")
        if label is not None:
            label=Image.open(label, mode="r")
        return image,label

    @staticmethod
    def resizeImage(image, label,size=128):
        size=int(1.05*size)
        image=image.resize((size, size), Image.ANTIALIAS)
        if label is not None:
            label=label.resize((size,size), Image.ANTIALIAS)
        return image,label

    @staticmethod
    def randomRotation(image, label, mode=Image.BICUBIC,size=128):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        random_angle = np.random.randint(1, 360)
        if label is not None:
            return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST)
        else:
            return image.rotate(random_angle, mode), label

    # 暂时未使用这个函数
    @staticmethod
    def randomCrop(image, label,size=128):
        """
        对图像随意剪切,考虑到图像大小范围(68,68),使用一个一个大于(36*36)的窗口进行截图
        :param image: PIL的图像image
        :return: 剪切之后的图像
        """
        image_width = image.size[0]
        image_height = image.size[1]
        crop_win_size = np.random.randint(size-18, size)
        random_region = (
            (image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,
            (image_height + crop_win_size) >> 1)
        image = image.crop(random_region)
        if label is not None:
            label=label.crop(random_region)
        return image, label

    @staticmethod
    def randomColor(image, label,size=128):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor), label  # 调整图像锐度

    @staticmethod
    def randomGaussian(image, label, mean=0.2, sigma=0.3,size=None):
        """
         对图像进行高斯噪声处理
        :param image:
        :return:
        """

        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            """
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
            :param sigma: 标准差
            :return:
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        # 将图像转化成数组
        img = np.asarray(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img)), label

    @staticmethod
    def array2tensor(image,label,size=128):
        image=Image.fromarray(np.uint8(image))
        if label is not None:
            label=Image.fromarray(np.uint8(label))
        return image,label

    @staticmethod
    def saveImage(image, path):
        image.save(path)



class DATASET():
    #CAR and AIR dataset
    def __init__(self,dataset,path,root, input_size=128,is_train=True):
        self.dataset=dataset
        self.input_size = input_size
        self.path=path
        self.is_train=is_train
        self.data=self.readPath(path)
        self.root=root
        self.seed=0
        self.Aug=DataAugmentation
        self.transform_val = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()])
        if dataset=="COVID":
            self.label_key = {
                "COVID":0,
                "Normal":1,
                "Lung_Opacity":2,
                "Viral Pneumonia":3
            }
        else:
            self.label_key={
                "adenosis":0,
                "fibroadenoma":1,
                "phyllodes_tumor":2,
                "tubular_adenoma":3,
                "ductal_carcinoma":4,
                "lobular_carcinoma":5,
                "mucinous_carcinoma":6,
                "papillary_carcinoma":7
             }


    def readPath(self,path):
        data=[]
        for line in open(path,encoding='UTF'):
            line=line.replace('\n','')
            data.append(line)
        return data

    def traindataAug(self,image,label,size):
        image,label=self.Aug.openImage(image,label,size=size)
        image,label=self.Aug.resizeImage(image,label,size=size)
        image,label=self.Aug.randomRotation(image,label,size=size)
        image,label=self.Aug.randomCrop(image,label,size=size)
        image,label=self.Aug.array2tensor(image,label,size=size)
        return image,label

    def testdataAug(self,image,label,size):
        image, label = self.Aug.openImage(image, label,size=size)
        image, label = self.Aug.resizeImage(image, label,size=size)
        image, label = self.Aug.array2tensor(image, label,size=size)
        return image,label

    def train_transforms(self,img, input_size):
        img = transforms.Resize((int(1.05*input_size), int(1.05*input_size)), Image.BILINEAR)(img)
        img = transforms.RandomHorizontalFlip()(img)
        img = transforms.CenterCrop(input_size)(img)
        img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img

    def test_transforms(self,img,input_size):
        img = transforms.Resize((input_size, input_size), Image.BILINEAR)(img)
        # img = transforms.Resize((688, 688), Image.BILINEAR)(img)
        img = transforms.CenterCrop(self.input_size)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img


    def __getitem__(self, index):
        img,target=(self.data[index]).split('[token]')

        target=int(target)
        imgData=self.root+'/'+img
        name=imgData
        # if self.dataset=='breast':
        #     imgData.replace('\ufeff','')
        mask=None
        if self.is_train==True:
            imgData, mask = self.traindataAug(imgData, mask, size=self.input_size)
            imgData = self.transform_val(imgData)
            # mask=self.transform_val(mask)
            if imgData.shape[0] > 3:
                imgData = torch.mean(imgData, dim=0, keepdim=True)
            if imgData.shape[0] < 3:
                imgData = imgData.repeat(3, 1, 1)
        else:
            imgData, mask = self.testdataAug(imgData, mask, size=self.input_size)
            imgData = self.transform_val(imgData)
            # mask=self.transform_val(mask)
            if imgData.shape[0] > 3:
                imgData = torch.mean(imgData, dim=0, keepdim=True)
            if imgData.shape[0] < 3:
                imgData = imgData.repeat(3, 1, 1)
        return index,imgData,target,name

    def __len__(self):
        return len(self.data)


# class Xray_DATASET():
#     def __init__(self,root,data_list, input_size=224,is_train=True):
#         self.root=root
#         self.input_size = input_size
#         self.data=data_list
#         self.is_train=is_train
#         self.seed=0
#         self.transform_val = transforms.Compose([
#             transforms.Resize(input_size),
#             transforms.ToTensor()])
#
#     def train_transforms(self,img, input_size):
#         img = transforms.Resize((int(1.0*input_size), int(1.0*input_size)))(img)
#         img = transforms.CenterCrop(input_size)(img)
#         img = transforms.ToTensor()(img)
#         img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
#         return img
#
#     def train_strong_transforms(self,img, input_size):
#         img = transforms.Resize((int(1.05*input_size), int(1.05*input_size)), Image.BILINEAR)(img)
#         img = transforms.RandomHorizontalFlip()(img)
#         img = transforms.CenterCrop(input_size)(img)
#         img = transforms.RandomGrayscale(p=0.2)(img)
#         img = transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5)(img)
#         img = transforms.ToTensor()(img)
#         img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
#         return img
#
#     def test_transforms(self,img,input_size):
#         img = transforms.Resize((int(1.05*input_size), int(1.05*input_size)))(img)
#         # img = transforms.Resize((688, 688), Image.BILINEAR)(img)
#         img = transforms.CenterCrop(self.input_size)(img)
#         img = transforms.ToTensor()(img)
#         img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
#         return img
#
#     def __getitem__(self, index):
#         data=(self.data[index]).replace('\n', '').split(' ')
#         img=data[0]
#         target=data[1:]
#         target_int=[]
#         target_sum = 0
#         for i in range(len(target)):
#             target_int.append(int(target[i]))
#             target_sum=target_sum+int(target[i])
#         if target_sum>0:
#             target_int.append(0)
#         else:
#             target_int.append(1)
#         target=np.array(target_int)
#
#         imgData=self.root+'/'+img
#         imgData=Image.open(imgData).convert('RGB')
#         if self.is_train==True:
#             imgData=self.train_transforms(imgData,self.input_size)
#         else:
#             imgData = self.test_transforms(imgData, self.input_size)
#         return index,imgData,target
#
#     def __len__(self):
#         return len(self.data)

class MIMIC_DATASET():
    def __init__(self,root,data_list, input_size=224,is_train=True):
        self.root=root
        self.input_size = input_size
        self.data=data_list
        self.is_train=is_train
        self.seed=0
        self.transform_val = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()])

    def train_transforms(self,img, input_size):
        img = transforms.Resize((int(1.0*input_size), int(1.0*input_size)))(img)
        img = transforms.CenterCrop(input_size)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img

    def train_strong_transforms(self,img, input_size):
        img = transforms.Resize((int(1.05*input_size), int(1.05*input_size)), Image.BILINEAR)(img)
        img = transforms.RandomHorizontalFlip()(img)
        img = transforms.CenterCrop(input_size)(img)
        img = transforms.RandomGrayscale(p=0.2)(img)
        img = transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img

    def test_transforms(self,img,input_size):
        img = transforms.Resize((int(1.05*input_size), int(1.05*input_size)))(img)
        # img = transforms.Resize((688, 688), Image.BILINEAR)(img)
        img = transforms.CenterCrop(self.input_size)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img

    def __getitem__(self, index):
        dicom_id, subject_id, study_id, VP, target, states = (self.data[index]).split('[splittoken]')
        img='p'+subject_id[0:2]+'/p'+subject_id+'/s'+study_id+'/'+dicom_id+'.jpg'

        target_int=[]
        target_sum=0
        for i in range(len(target)):
            cur_target=int(target[i])
            if cur_target>1:
                cur_target=0
            target_int.append(cur_target)
            target_sum=target_sum+cur_target
        if target_sum>0:
            target_int.append(0)
        else:
            target_int.append(1)
        target=np.array(target_int)

        imgData=self.root+'/'+img
        imgData=Image.open(imgData).convert('RGB')
        if self.is_train==True:
            imgData=self.train_transforms(imgData,self.input_size)
        else:
            imgData = self.test_transforms(imgData, self.input_size)
        return index,imgData,target

    def __len__(self):
        return len(self.data)


class HE_DATASET():
    def __init__(self,root,data_list, input_size=224,is_train=True,is_HE=None):
        self.root=root
        self.input_size = input_size
        self.data=data_list
        self.is_train=is_train
        self.seed=0
        self.transform_val = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()])
        self.is_HE=is_HE

    def train_transforms(self,img, input_size):
        img = transforms.Resize((int(1.05*input_size), int(1.05*input_size)), Image.BILINEAR)(img)
        img = transforms.RandomHorizontalFlip()(img)
        img = transforms.CenterCrop(input_size)(img)
        img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img

    def train_strong_transforms(self,img, input_size):
        img = transforms.Resize((int(1.05*input_size), int(1.05*input_size)), Image.BILINEAR)(img)
        img = transforms.RandomHorizontalFlip()(img)
        img = transforms.CenterCrop(input_size)(img)
        img = transforms.RandomGrayscale(p=0.2)(img)
        img = transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img

    def test_transforms(self,img,input_size):
        img = transforms.Resize((int(1.05*input_size), int(1.05*input_size)), Image.BILINEAR)(img)
        # img = transforms.Resize((688, 688), Image.BILINEAR)(img)
        img = transforms.CenterCrop(self.input_size)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img

    def __getitem__(self, index):
        # print(self.data[index])
        img,target=(self.data[index]).split('[splittoken]')
        target=int(target)
        target=target
        imgData=self.root+'/'+img
        name=imgData
        # print(imgData)
        imgData = imgData.replace('.tif', '.jpg').replace('regular-fundus-training/',
                                                          'regular-fundus-training/Images/')
        imgData = imgData.replace('regular-fundus-validation/', 'regular-fundus-validation/Images/')
        if self.is_HE is None:
            imgData = imgData.replace('.png', '_process.png').replace('.jpg', '_process.jpg')
        # print(imgData)
        imgData=Image.open(imgData).convert('RGB')
        if self.is_train==True:
            imgData=self.train_transforms(imgData,self.input_size)
        else:
            imgData = self.test_transforms(imgData, self.input_size)
        return index,imgData,target    #,name

    def __len__(self):
        return len(self.data)


class CUB_DATASET():
    def __init__(self,root,data_list, input_size=224,is_train=True,is_HE=None):
        self.root=root
        self.input_size = input_size
        self.data=data_list
        self.is_train=is_train
        self.seed=0
        self.transform_val = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()])
        self.is_HE=is_HE

    def train_transforms(self,img, input_size):
        img = transforms.Resize((int(1.05*input_size), int(1.05*input_size)), Image.BILINEAR)(img)
        img = transforms.RandomHorizontalFlip()(img)
        img = transforms.CenterCrop(input_size)(img)
        img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img

    def train_strong_transforms(self,img, input_size):
        img = transforms.Resize((int(1.05*input_size), int(1.05*input_size)), Image.BILINEAR)(img)
        img = transforms.RandomHorizontalFlip()(img)
        img = transforms.CenterCrop(input_size)(img)
        img = transforms.RandomGrayscale(p=0.2)(img)
        img = transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img

    def test_transforms(self,img,input_size):
        img = transforms.Resize((int(1.05*input_size), int(1.05*input_size)), Image.BILINEAR)(img)
        # img = transforms.Resize((688, 688), Image.BILINEAR)(img)
        img = transforms.CenterCrop(self.input_size)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img

    def __getitem__(self, index):
        img,target=(self.data[index]).split('[splittoken]')
        target=int(target)
        if self.is_HE is not None:
            target=target%100
        imgData=self.root+'/'+img
        # print(imgData)
        imgData=Image.open(imgData).convert('RGB')
        if self.is_train==True:
            imgData=self.train_transforms(imgData,self.input_size)
        else:
            imgData = self.test_transforms(imgData, self.input_size)
        return index,imgData,target

    def __len__(self):
        return len(self.data)


class ISIC_DATASET():
    def __init__(self,root,data_list, input_size=224,is_train=True):
        self.root=root
        self.input_size = input_size
        self.data=data_list
        self.is_train=is_train
        self.seed=0
        self.transform_val = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()])

    def train_transforms(self,img, input_size):
        img = transforms.Resize((int(1.15*input_size), int(1.15*input_size)), Image.BILINEAR)(img)
        img = transforms.RandomHorizontalFlip()(img)
        img = transforms.CenterCrop(input_size)(img)
        img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img

    def train_strong_transforms(self,img, input_size):
        img = transforms.Resize((int(1.05*input_size), int(1.05*input_size)), Image.BILINEAR)(img)
        img = transforms.RandomHorizontalFlip()(img)
        img = transforms.CenterCrop(input_size)(img)
        img = transforms.RandomGrayscale(p=0.2)(img)
        img = transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        # img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)
        return img

    def test_transforms(self,img,input_size):
        img = transforms.Resize((int(1.05*input_size), int(1.05*input_size)), Image.BILINEAR)(img)
        # img = transforms.Resize((688, 688), Image.BILINEAR)(img)
        img = transforms.CenterCrop(self.input_size)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        # img = transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])(img)
        return img

    def __getitem__(self, index):
        img,target=(self.data[index]).split('[splittoken]')
        target=int(target)
        imgData=self.root+'/'+img+'.jpg'

        imgData=Image.open(imgData).convert('RGB')
        if self.is_train==True:
            imgData=self.train_transforms(imgData,self.input_size)
        else:
            imgData = self.test_transforms(imgData, self.input_size)
        return index,imgData,target

    def __len__(self):
        return len(self.data)


class NIH_DATASET():
    #CAR and AIR dataset
    def __init__(self,root,data_list, input_size=256,is_train=True):
        self.root=root
        self.input_size = input_size
        self.data=data_list
        # self.label_list=transfer_label(data_list)
        self.is_train=is_train
        self.seed=0
        self.transform_val = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()])

    def train_transforms(self,img, input_size):
        img = transforms.Resize((int(1.05*input_size), int(1.05*input_size)), Image.BILINEAR)(img)
        img = transforms.RandomHorizontalFlip()(img)
        img = transforms.CenterCrop(input_size)(img)
        img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img

    def test_transforms(self,img,input_size):
        img = transforms.Resize((input_size, input_size), Image.BILINEAR)(img)
        # img = transforms.Resize((688, 688), Image.BILINEAR)(img)
        img = transforms.CenterCrop(self.input_size)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img

    def __getitem__(self, index):
        data=(self.data[index]).replace('\n','')
        img,label=data.split(' ')[0], data.split(' ')[1:]

        label_int=[int(i) for i in label]
        label_int=np.array(label_int)

        remain_index=[0,2,3,4,5,7,8]
        other_index=[1,6,9,10,11,12,13]
        remain_label=label_int[remain_index]
        other_label=label_int[other_index]
        other_label=other_label.sum()

        label_int=np.zeros(8)
        label_int[0:7]=remain_label

        if other_label>0:
            label_int[7]=1
        # label_int=list(remain_label)+list(other_label)
        # label_int=np.array(label_int)

        imgData=self.root+'/'+img
        imgData=Image.open(imgData).convert('RGB')
        if self.is_train==True:
            imgData=self.train_transforms(imgData,self.input_size)
        else:
            imgData = self.test_transforms(imgData, self.input_size)
        return index,imgData,label_int

    def __len__(self):
        return len(self.data)


class Xray_DATASET():
    def __init__(self,root,data_list, input_size=224,is_train=True,is_HE=None):
        self.root=root
        self.input_size = input_size
        self.data=data_list
        self.is_train=is_train
        self.seed=0
        self.transform_val = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()])
        self.is_HE=is_HE

    def train_transforms(self,img, input_size):
        img = transforms.Resize((int(1.05*input_size), int(1.05*input_size)), Image.BILINEAR)(img)
        img = transforms.RandomHorizontalFlip()(img)
        img = transforms.CenterCrop(input_size)(img)
        # img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img

    def train_strong_transforms(self,img, input_size):
        img = transforms.Resize((int(1.05*input_size), int(1.05*input_size)), Image.BILINEAR)(img)
        img = transforms.RandomHorizontalFlip()(img)
        img = transforms.CenterCrop(input_size)(img)
        # img = transforms.RandomGrayscale(p=0.2)(img)
        # img = transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img

    def test_transforms(self,img,input_size):
        img = transforms.Resize((int(1.05*input_size), int(1.05*input_size)), Image.BILINEAR)(img)
        # img = transforms.Resize((688, 688), Image.BILINEAR)(img)
        img = transforms.CenterCrop(self.input_size)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img

    def __getitem__(self, index):
        img,target=(self.data[index]).split('[splittoken]')
        target=target.replace('\n','')
        target=int(target)
        # img=img.replace('data1/images/cropedimages/','images/')
        # if self.is_HE is not None:
        #     target=target%100
        imgData=self.root+'/'+img
        name=imgData
        # print(imgData)
        imgData=Image.open(imgData).convert('RGB')
        if self.is_train==True:
            imgData=self.train_transforms(imgData,self.input_size)
        else:
            imgData = self.test_transforms(imgData, self.input_size)
        return index,imgData,target

    def __len__(self):
        return len(self.data)
# if __name__ == '__main__':
#     dataset = 'breast'
#     path = 'E:/PycharmProjects/Fujian/data/breast_cancer/train.txt'
#     root = 'E:/dataset/福建省立医院乳腺癌分析/福建省立医院乳腺癌图片'
#
#     # dataset = 'BreakHis'
#     # path = 'C:/Users/Administrator/PycharmProjects/MedicalIMGPro/breast/train8class400X'
#     # root = 'C:/Users/Administrator/Documents/papers/medicalAI/retrieval/dataset/archive-BreaKHis_v1/breast'
#     Dataset = DATASET(dataset, path, root)
#     dataloader = torch.utils.data.DataLoader(Dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
#     for i, (index, img,target) in enumerate(dataloader):
#         a=img.numpy()
#         # b=name[0].split('/')[-1]
        # save_path='mask'+b
        # c=mask.max()
        # mask[mask==c]=0
        # c = mask.max()
        # mask[mask == c] = 0
        # c = mask.max()
        # mask[mask!=c]=0
        # # vutils.save_image(mask[0]*255,b)
        # img[img>0]=1
        # img=1-img
        # print(img.sum()/img.shape[0]/img.shape[1]/img.shape[2]/img.shape[3])
        # print()





