import cv2 as cv
import numpy as np

def nothing():
    """
    空函数，用于OpenCV滑动条的回调函数。
    当滑动条的值发生变化时，调用此函数，但不执行任何操作。
    """
    pass

def initTrackbars(window_info):
    """
    初始化多个窗口和对应的滑动条。

    :param window_info: 一个列表，每个元素是一个字典，包含窗口名称和滑动条信息。
                        滑动条信息是一个列表，每个元素是一个元组，包含滑动条名称、初始值、最大值。
    """
    for info in window_info:
        window_name = info["window_name"]
        trackbars = info["trackbars"]
        # 创建窗口
        cv.namedWindow(window_name)
        # 调整窗口大小
        cv.resizeWindow(window_name, 360, 240)
        for trackbar_name, initial_value, max_value in trackbars:
            # 创建滑动条，回调函数为nothing
            cv.createTrackbar(trackbar_name, window_name, initial_value, max_value, nothing)

def valueTrackbars(window_info):
    """
    从多个窗口中获取滑动条的值。

    :param window_info: 一个列表，每个元素是一个字典，包含窗口名称和滑动条信息。
                        滑动条信息是一个列表，每个元素是一个字符串，表示滑动条名称。
    :return: 一个字典，键为窗口名称，值为该窗口中滑动条的值组成的列表。
    """
    result = {}
    for info in window_info:
        window_name = info["window_name"]
        trackbars = info["trackbars"]
        values = []
        for trackbar_name in trackbars:
            # 获取滑动条的当前值
            value = cv.getTrackbarPos(trackbar_name, window_name)
            values.append(value)
        result[window_name] = values
    return result

def biggestContour(contours):
    """
    从给定的轮廓列表中找到最大的四边形轮廓。

    :param contours: 轮廓列表，每个轮廓是一个由点组成的数组。
    :return: 最大四边形轮廓的点数组和其面积。
    """
    max_area = 0
    biggest = np.array([])
    for cnt in contours:
        # 计算当前轮廓的面积
        area = cv.contourArea(cnt)
        # 降低面积阈值以适应倾斜情况
        if area > 3000:  # 从5000调整为3000
            # 计算轮廓的周长
            peri = cv.arcLength(cnt, True)
            # 调整多边形逼近的精度参数
            approx = cv.approxPolyDP(cnt, 0.03 * peri, True)  # 从0.02调整为0.03
            # 放宽四边形检测条件
            if area > max_area and (len(approx) == 4 or len(approx) == 5):  # 增加对五边形的容忍
                # 更新最大轮廓和最大面积
                biggest = approx
                max_area = area
    
    # 如果是五边形，尝试转换为四边形
    if len(biggest) == 5:
        biggest = cv.approxPolyDP(biggest, 0.02 * cv.arcLength(biggest, True), True)
        # 如果转换后还不是四边形，则返回空
        if len(biggest) != 4:
            return np.array([]), 0
            
    return biggest, max_area

def drawRectangle(img, biggest, thickness):
    """
    在图像上绘制一个由四个点定义的矩形。

    :param img: 要绘制矩形的图像。
    :param biggest: 矩形的四个顶点，形状为 (4, 1, 2) 的数组。
    :param thickness: 矩形边框的厚度。
    :return: 绘制了矩形的图像。
    """
    # 绘制矩形的四条边
    cv.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    return img

def reorder(myPoints):
    """
    对四个点进行重新排序，使其顺序为左上角、右上角、左下角、右下角。

    :param myPoints: 四个点的数组，形状为 (4, 2)。
    :return: 重新排序后的点数组，形状为 (4, 1, 2)。
    """
    # 调整点数组的形状
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    # 计算每个点的坐标和
    add = myPoints.sum(1)
    # 左上角的点坐标和最小
    myPointsNew[0] = myPoints[np.argmin(add)]
    # 右下角的点坐标和最大
    myPointsNew[3] = myPoints[np.argmax(add)]
    # 计算每个点的坐标差
    diff = np.diff(myPoints, axis=1)
    # 右上角的点坐标差最小
    myPointsNew[1] = myPoints[np.argmin(diff)]
    # 左下角的点坐标差最大
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def stackImages(scale, imgArray, labels=[]):
    """
    将多个图像堆叠成一个大图像。

    :param scale: 图像缩放比例。
    :param imgArray: 图像数组，可以是二维列表或一维列表。
    :param labels: 每个图像的标签列表，可选参数。
    :return: 堆叠后的大图像。
    """
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    # 缩放图像
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    # 调整图像大小
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    # 将灰度图像转换为彩色图像
                    imgArray[x][y] = cv.cvtColor(imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            # 水平堆叠图像
            hor[x] = np.hstack(imgArray[x])
            # 水平连接图像
            hor_con[x] = np.concatenate(imgArray[x])
        # 垂直堆叠图像
        ver = np.vstack(hor)
        # 垂直连接图像
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                # 缩放图像
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                # 调整图像大小
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                # 将灰度图像转换为彩色图像
                imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        # 水平堆叠图像
        hor = np.hstack(imgArray)
        # 水平连接图像
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(labels) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        for d in range(0, rows):
            for c in range(0, cols):
                # 绘制标签背景
                cv.rectangle(ver, (c*eachImgWidth, eachImgHeight*d), (c*eachImgWidth+len(labels[d][c])*13+27, 30+eachImgHeight*d), (255, 255, 255), cv.FILLED) 
                cv.putText(ver, labels[d][c], (eachImgWidth*c+10, eachImgHeight*d+20), cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver

def spiltboxs(img, ques=7, choice=5):
    """
    将输入的图像分割成多个小方块。

    :param img: 输入的图像
    :param ques: 问题的数量，默认为7
    :param choice: 每个问题的选项数量，默认为5
    :return: 分割后的小方块列表
    """
    # 垂直分割图像为ques行
    rows = np.vsplit(img, ques)
    # 初始化小方块列表
    boxes = []
    # 遍历每一行
    for r in rows:
        # 水平分割当前行图像为choice列
        cols = np.hsplit(r, choice)
        # 遍历每一列
        for box in cols:
            # 将每个小方块添加到列表中
            boxes.append(box)
    return boxes

def displayAnswers(img, myIndex, grading, ans, ques=7, choice=5):
    """
    在图像上显示答案和得分情况。

    :param img: 要显示答案的图像
    :param myIndex: 用户选择的答案索引列表
    :param grading: 每个问题的得分情况列表
    :param ans: 正确答案索引列表
    :param ques: 问题的数量，默认为7
    :param choice: 每个问题的选项数量，默认为5
    :return: 显示了答案和得分情况的图像
    """
    # 计算每个选项的宽度
    sectionWidth = int(img.shape[1]/choice)
    # 计算每个问题的高度
    sectionHeight = int(img.shape[0]/ques)
    # 遍历每个问题
    for x in range(0, ques):
        # 获取用户当前问题的选择
        myAns = myIndex[x]
        # 计算用户选择的选项的中心点坐标
        cx, cy  = myAns * sectionWidth + sectionWidth//2, (x+1) * sectionHeight - sectionHeight//2
        # 判断用户当前问题是否回答正确
        if grading[x] == 1:
            # 回答正确，设置颜色为绿色
            myColor = (0, 255, 0)
            # 在用户选择的选项中心绘制绿色填充圆
            cv.circle(img, (cx, cy), 25, myColor, cv.FILLED)
        else:
            # 回答错误，设置颜色为红色
            myColor = (0, 0, 255)
            # 获取正确答案
            rightAns = ans[x]
            # 在用户选择的选项中心绘制红色填充圆
            cv.circle(img, (cx, cy), 25, myColor, cv.FILLED)
            # 在正确答案的选项中心绘制绿色填充圆
            cv.circle(img, ((rightAns * sectionWidth + sectionWidth//2), (x+1) * sectionHeight - sectionHeight//2), 10, (0, 255, 0), cv.FILLED)
    return img

def displayResult(img, img_original, myIndex, ques=7, choice=5, detect_w=30, detect_h=30, debug=False, 
                  myPixelVal=None, matrix=None):
    """
    在图像上显示棋子的位置和颜色。

    :param img: 要显示答案的图像
    :param myIndex: 用户选择的答案索引列表
    :param grading: 每个问题的得分情况列表
    :param ans: 正确答案索引列表
    :param ques: 问题的数量，默认为7
    :param choice: 每个问题的选项数量，默认为5
    :return: 显示了答案和得分情况的图像
    """
    # 计算每个选项的宽度
    def originalcenter(x, y, matrix):
        """
        计算原始中心点坐标，添加比例调整
        """
        if matrix is None or not isinstance(matrix, np.ndarray):
            return (0, 0)
            
        try:
            # 获取原始图像和变换后图像的尺寸比例
            original_h, original_w = img_original.shape[:2]
            warped_h, warped_w = img.shape[:2]
            
            # 调整坐标比例
            x_scaled = x * (original_w / warped_w)
            y_scaled = y * (original_h / warped_h)
            
            point = np.array([[[x_scaled, y_scaled]]], dtype=np.float32)
            original_point = cv.perspectiveTransform(point, np.linalg.inv(matrix))
            #print((int(original_point[0][0][0]), int(original_point[0][0][1])))
            return (int(original_point[0][0][0]), int(original_point[0][0][1]))
        
        except Exception as e:
            print(f"坐标转换错误: {e}")
            return (0, 0)
    # 计算每个选项的宽度   
    sectionWidth = int(img.shape[1]/choice)
    # 计算每个问题的高度
    sectionHeight = int(img.shape[0]/ques)
    # 遍历每个问题
    for y in range(0, choice):
        for x in range(0, ques):
            cx, cy  = x * sectionWidth + sectionWidth//2, (y+1) * sectionHeight - sectionHeight//2
            cv.circle(img, (cx, cy), 2, (0, 0, 255), cv.FILLED)
            cv.circle(img_original, originalcenter(cx, cy, matrix), 2, (0, 0, 255), cv.FILLED)
            cv.rectangle(img, (cx - detect_w // 2, cy - detect_h // 2), (cx + detect_w // 2, cy + detect_h // 2), (255, 0, 0), 1)
            cv.putText(img, str(x + y + 1), (cx - 10, cy + 10), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            if debug:
                cv.putText(img, str(myPixelVal[y][x]), (cx - 10, cy + 30), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            if myIndex[y][x] == 1: 
                cv.putText(img, "black", (cx - 10, cy + 10), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                if debug:
                    cv.circle(img_original, originalcenter(cx, cy, matrix), 10, (0, 0, 0), cv.FILLED)
            elif myIndex[y][x] == 2:
                cv.putText(img, "white", (cx - 10, cy + 10), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
                if debug:
                    cv.circle(img_original, originalcenter(cx, cy, matrix), 10, (255, 255, 255), cv.FILLED)
    return img, img_original