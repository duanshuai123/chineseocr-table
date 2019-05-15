#coding:utf-8
'''
表格生成线条坐标
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
import json
import sys
import subprocess
import os

import os
from PIL import Image
import json
import time
import uuid
import base64
import model
from config import DETECTANGLE
from apphelper.image import union_rbox
from application import trainTicket,idcard

import pytesseract


# 图像切割模块
class cutImage(object):
    def __init__(self,img, bin_threshold, kernel, iterations, areaRange, filename, border=10, show=True, write=True,):
        '''
        :param img: 输入图像
        :param bin_threshold: 二值化的阈值大小
        :param kernel: 形态学kernel
        :param iterations: 迭代次数
        :param areaRange: 面积范围
        :param filename:保留json数据的文件名称
        :param border: 留边大小
        :param show: 是否显示结果图，默认是显示
        :param write: 是否把结果写到文件，默认是写入
        '''
        self.img = img
        self.bin_threshold = bin_threshold
        self.kernel = kernel
        self.iterations = iterations
        self.areaRange = areaRange
        self.border = border
        self.show = show
        self.write = write
        self.filename = filename

    def getRes(self):
        # 获取格网线图和格网点图,都是像素级别的
        img_erode, joint = detectTable(self.img).run()
        cv2.imshow("src", img_erode)
        image, contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        area_coord_roi = []
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if area >self.areaRange[0] and area <self.areaRange[1]:
                x, y, w, h = cv2.boundingRect(cnt)
                roi = self.img[(y+self.border):((y+h)-self.border), (x+self.border):((x+w)-self.border)]
                area_coord_roi.append((area,(x,y,w,h),roi))

        #最大面积
        max_area = max([info[0] for info in area_coord_roi])

        for info in area_coord_roi:
            if info[0]==max_area:
                max_rect = info[1]
                max_info = info

        bScript = cutImage.SkipMaxArea(self,max_info,area_coord_roi)

        for each in area_coord_roi:
            x,y,w,h = each[1]
            #if 1 == 0:  #x>max_rect[0] and y>max_rect[1] and (x+w)<(max_rect[0]+max_rect[2]) and (y+h) <(max_rect[1]+max_rect[3]):
            if bScript and each[0] == max_area: #跳过最大面积的表格
                pass
            else:
                billModel = '通用OCR'
                dst_dir = 'clip'
                if not os.path.exists(dst_dir):
                    os.mkdir(dst_dir)
                name = '%s/%d_%d_%d_%d.jpg' % (dst_dir, x, y, x + w, y + h)
                cv2.imwrite(name, each[2])
                img = Image.open(name).convert("RGB") #图片临时保存再来

                # 定义图像拼接函数
                def image_compose():
                    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
                    # 循环遍历，把每张图片按顺序粘贴到对应位置上
                    for y in range(1, IMAGE_ROW + 1):
                        for x in range(1, IMAGE_COLUMN + 1):
                            from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                                (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
                            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
                    return to_image.save(IMAGE_SAVE_PATH)  # 保存新图



                code = pytesseract.image_to_string(img, lang='chi_sim')
                print(code)
                print(name)

                '''
                img = Image.open(name).convert("RGB")
                W, H = img.size
                timeTake = time.time()
                _, result, angle = model.model(img,
                                               detectAngle=DETECTANGLE,  ##是否进行文字方向检测
                                               config=dict(MAX_HORIZONTAL_GAP=100,  ##字符之间的最大间隔，用于文本行的合并
                                                           MIN_V_OVERLAPS=0.7,
                                                           MIN_SIZE_SIM=0.7,
                                                           TEXT_PROPOSALS_MIN_SCORE=0.1,
                                                           TEXT_PROPOSALS_NMS_THRESH=0.3,
                                                           TEXT_LINE_NMS_THRESH=0.99,  ##文本行之间测iou值
                                                           MIN_RATIO=1.0,
                                                           LINE_MIN_SCORE=0.2,
                                                           TEXT_PROPOSALS_WIDTH=0,
                                                           MIN_NUM_PROPOSALS=0,
                                                           ),
                                               leftAdjust=True,  ##对检测的文本行进行向左延伸
                                               rightAdjust=True,  ##对检测的文本行进行向右延伸
                                               alph=0.2,  ##对检测的文本行进行向右、左延伸的倍数
                                               ifadjustDegree=False  ##是否先小角度调整文字倾斜角度
                                               )

                result = union_rbox(result, 0.2)
                res = [{'text': x['text'], 'name': str(i)} for i, x in enumerate(result)]
                print(result)
                #os.remove(name)
                '''
        return




    def SkipMaxArea(self,maxinfo,area_coord_roi):
        mx,my,mw,mh = maxinfo[1]
        for info in area_coord_roi:
            if info[0]==maxinfo[0]:
                pass
            else:
                x,y,w,h = info[1]
                if x>mx and y>my and x+w<mx+mw and y+h<my+mh:
                    return 1
        return 0

    def SkipMaxArea(self, maxinfo, area_coord_roi):
        IMAGES_PATH = 'D:\Mafengwo\photo\五月坦桑的暖风，非洲原野的呼唤\\'  # 图片集地址
        IMAGES_FORMAT = ['.jpg', '.JPG']  # 图片格式
        IMAGE_SIZE = 256  # 每张小图片的大小
        IMAGE_ROW = 5  # 图片间隔，也就是合并成一张图后，一共有几行
        IMAGE_COLUMN = 4  # 图片间隔，也就是合并成一张图后，一共有几列
        IMAGE_SAVE_PATH = 'final.jpg'  # 图片转换后的地址

        # 获取图片集地址下的所有图片名称
        image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
                    os.path.splitext(name)[1] == item]

        # 简单的对于参数的设定和实际图片集的大小进行数量判断
        if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
            raise ValueError("合成图片的参数和要求的数量不能匹配！")

        # 定义图像拼接函数
        def image_compose():
            to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
            # 循环遍历，把每张图片按顺序粘贴到对应位置上
            for y in range(1, IMAGE_ROW + 1):
                for x in range(1, IMAGE_COLUMN + 1):
                    from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                        (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
                    to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
            return to_image.save(IMAGE_SAVE_PATH)  # 保存新图

        image_compose()  # 调用函数



    def getRes_By_Cell(self):
        #获取格网线图和格网点图
        img_line, img_joint = detectTable(self.img).run()
        cv2.imshow("line", img_line)
        cv2.imshow("point", img_joint)

        #获取格网图边界 x,y 最大最小值，兼容表格最外边没有边界的情况
        contours, hierarchy = cv2.findContours(img_line, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        xmin , xmax, ymin, ymax = img_line.shape[1], 0, img_line.shape[0], 0
        for i in range(len(contours)):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            xmin = min(xmin, x)
            xmax = max(xmax, x + w)
            ymin = min(ymin, y)
            ymax = max(ymax, y + h)

        #通过格网点及格网线图找出每一个单元格并裁切
        idx = np.argwhere(img_joint > 1)
        idx_unirow = np.unique(idx[:, 0])
        idx_unicol = np.unique(idx[0, :]);
        return


# 检测表格，使用形态学
# 返回是表格图以及表格中交叉点的图
class detectTable(object):
    def __init__(self, src_img):
        self.src_img = src_img

    def run(self):
        #转换为灰度图
        if len(self.src_img.shape) == 2:  # 灰度图
            gray_img = self.src_img
        elif len(self.src_img.shape) ==3:
            gray_img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2GRAY)

        #二值化
        thresh_img = cv2.adaptiveThreshold(~gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
        h_img = thresh_img.copy()
        v_img = thresh_img.copy()
        scale = 15
        h_size = int(h_img.shape[1]/scale)

        #提取水平方向格网线（通过腐蚀膨胀）
        h_structure = cv2.getStructuringElement(cv2.MORPH_RECT,(h_size,1)) # 形态学因子
        h_erode_img = cv2.erode(h_img,h_structure, 1) #腐蚀
        #cv2.imshow("1",h_erode_img)
        h_dilate_img = cv2.dilate(h_erode_img,h_structure, 1)  #膨胀
        #cv2.imshow("2",h_dilate_img)
        v_size = int(v_img.shape[0] / scale)

        # 提取竖直方向格网线（通过腐蚀膨胀）
        v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))  # 形态学因子
        v_erode_img = cv2.erode(v_img, v_structure, 1) #腐蚀
        #cv2.imshow("3", v_erode_img)
        v_dilate_img = cv2.dilate(v_erode_img, v_structure, 1) #膨胀
        #cv2.imshow("4", v_dilate_img)

        #合并格网线（水平+竖直）
        mask_img = h_dilate_img+v_dilate_img
        #cv2.imshow("5", mask_img)

        #计算格网线交叉点
        joints_img = cv2.bitwise_and(h_dilate_img,v_dilate_img)
        cv2.imshow("joints",joints_img)
        cv2.imshow("mask",mask_img)

        return mask_img, joints_img


if __name__=='__main__':
    img = cv2.imread('test/p27.png')
    win = cv2.namedWindow("img", flags=2) 
    cv2.imshow("img", img)
    #mask,joint = detectTable(img).run()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    area = img.shape[0]*img.shape[1]
    cutImage(img, 127, kernel, 2, (1, area), 'Json.json', 1).getRes()

    cv2.waitKey()




