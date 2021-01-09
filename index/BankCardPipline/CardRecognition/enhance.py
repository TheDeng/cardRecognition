import cv2 as cv
import numpy as np


# 直方图正规化
def normalization_change(path=None,img=None):
    if img is None:
        img = cv.imread(path, 0)
    # 计算原图中出现的最小灰度级和最大灰度级
    # 使用函数计算
    Imin, Imax = cv.minMaxLoc(img)[:2]
    # 使用numpy计算
    # Imax = np.max(img)
    # Imin = np.min(img)
    Omin, Omax = 0, 255
    # 计算a和b的值
    a = float(Omax - Omin) / (Imax - Imin)
    b = Omin - a * Imin
    out = a * img + b
    out = out.astype(np.uint8)
    return out

# 伽马变换
def gamma_change(path=None,img=None):
    if img is None:
        img = cv.imread(path, 0)
    # 图像归一化
    fi = img / 255.0
    # gamma<1 变亮，1时图片不变，>1时变暗
    gamma = 0.4
    out = np.power(fi, gamma)
    return out


# 直方图均衡化
def equal_hist(path=None,img=None):
    if img is None:
        img = cv.imread(path, 0)
    # 灰度图像矩阵的高、宽
    h, w = img.shape
    # 第一步：计算灰度直方图
    grayHist = calcGrayHist(img)
    # 第二步：计算累加灰度直方图
    zeroCumuMoment = np.zeros([256], np.uint32)
    for p in range(256):
        if p == 0:
            zeroCumuMoment[p] = grayHist[0]
        else:
            zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
    # 第三步：根据累加灰度直方图得到输入灰度级和输出灰度级之间的映射关系
    outPut_q = np.zeros([256], np.uint8)
    cofficient = 256.0 / (h * w)
    for p in range(256):
        q = cofficient * float(zeroCumuMoment[p]) - 1
        if q >= 0:
            outPut_q[p] = np.floor(q)
        else:
            outPut_q[p] = 0
    # 第四步：得到直方图均衡化后的图像
    out = np.zeros(img.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            out[i][j] = outPut_q[img[i][j]]

    return out


# 计算灰度直方图
def calcGrayHist(I):
    h, w = I.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            grayHist[I[i][j]] += 1
    return grayHist

# 限制对比度的自适应直方图均衡化
def equal_hist_with_opt(path=None,img=None):
    if img is None:
        img = cv.imread(path, 0)
    else:
        #cv.imshow('before_enhance', img)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #cv.imshow('after_gray',img)
    # 创建CLAHE对象
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值均衡化
    dst = clahe.apply(img)
    return dst

# 拉普拉斯算子
def laplace(path=None,img=None):
    if img is None:
        img = cv.imread(path, 0)
    dst = cv.Laplacian(img, cv.CV_32F)
    lpls = cv.convertScaleAbs(dst)
    return lpls

# sobel算子
def sobal(path=None,img=None):
    if img is None:
        img = cv.imread(path, 0)
    grad_x = cv.Sobel(img, cv.CV_32F, 1, 0)   #对x求一阶导
    grad_y = cv.Sobel(img, cv.CV_32F, 0, 1)   #对y求一阶导
    gradx = cv.convertScaleAbs(grad_x)  #用convertScaleAbs()函数将其转回原来的uint8形式
    grady = cv.convertScaleAbs(grad_y)

    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0) #图片融合
    return gradxy

# Scharr算子
def scharr(path=None,img=None):
    if img is None:
        img = cv.imread(path, 0)
    grad_x = cv.Scharr(img, cv.CV_32F, 1, 0)  # 对x求一阶导
    grad_y = cv.Scharr(img, cv.CV_32F, 0, 1)  # 对y求一阶导
    gradx = cv.convertScaleAbs(grad_x)  # 用convertScaleAbs()函数将其转回原来的uint8形式
    grady = cv.convertScaleAbs(grad_y)
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    return gradxy

if __name__ == '__main__':
    path = 'new_train_16/6216388888888888.jpg'
    img = None
    # hist_gray_change(path)
    #gamma_change(path)
    #equal_hist(path)
    #img = equal_hist_with_opt(path)
    # img = laplace(path)
    #sobal(path)
    img =  equal_hist_with_opt(path)
    cv.imshow('img', img)
    cv.waitKey()