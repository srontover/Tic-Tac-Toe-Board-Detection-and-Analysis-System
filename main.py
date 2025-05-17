# 导入OpenCV库，使用别名cv
import cv2 as cv
# 导入NumPy库，用于数值计算
import numpy as np
# 导入自定义的扫描功能模块
import scanFunctions as sf
import time

################################
# 定义图像文件的路径
path = "test.jpg"
# 定义是否使用摄像头的标志
Webcam = True
# 定义图像的高度
height = 640
# 定义图像的宽度
width = 480
# 初始化摄像头对象，使用默认摄像头（索引为0）
cap = cv.VideoCapture(0)
# 设置摄像头的宽度
cap.set(3, width)
# 设置摄像头的高度
cap.set(4, height)
# 设置摄像头的亮度
cap.set(10, 150)

# 定义窗口信息列表，每个元素是一个字典，包含窗口名称和滑动条信息
window_info = [
    {
        # 窗口名称
        "window_name": "threshold1",
        # 滑动条信息，每个滑动条是一个元组，包含名称、初始值和最大值
        "trackbars": [("thres_min1", 0, 255), ("thres_max1", 255, 255)]
    },
    # {
    #     # 窗口名称
    #     "window_name": "threshold2",
    #     # 滑动条信息，每个滑动条是一个元组，包含名称、初始值和最大值
    #     "trackbars": [("thres_min2", 0, 255), ("thres_max2", 255, 255)]
    # }
]

# 定义窗口信息结果列表，用于获取滑动条的值
window_info_result = [
    {
        # 窗口名称
        "window_name": "threshold1",
        # 滑动条名称列表
        "trackbars": ["thres_min1", "thres_max1"]
    },
    # {
    #     # 窗口名称
    #     "window_name": "threshold2",
    #     # 滑动条名称列表
    #     "trackbars": ["thres_min2", "thres_max2"]
    # }
]

# 定义问题数量
ROWS_BROAD = 3
# 定义每个问题的选项数量
COLUMNS_BROAD = 3

# 定义检测区域的宽度和高度
DETECT_WIDTH = 30
DETECT_HEIGHT = 30
# 定义稳定时间
STABLE_TIME = 1

# 定义答案像素值的阈值
ans_limit = 4000
# 定义变化检测的阈值
CHANGE_THRESHOLD = 1
# 定义正确答案列表
ans = [1,1,1]
# 定义答案像素值的阈值
ans_limit = 4000
#################################

#################################
myIndex_last = np.zeros((ROWS_BROAD, COLUMNS_BROAD))
last_computer_count = 0
last_person_count = 0
quantity_change_start = None  
position_change_start = None  
initial_position = None  # 新增：记录位置变化前的初始状态
# 初始化滑动条
sf.initTrackbars(window_info)
#################################

# 初始化计数器
count = 0

# 人执的棋子颜色
PERSON_COLOR = 2
# 机器执的棋子颜色
COMPUTER_COLOR = 1
# 进入主循环
while True:
    # 创建一个空白图像
    img_blank = np.zeros((width, height, 3), np.uint8)

    # 判断是否使用摄像头
    if Webcam:
        # 从摄像头读取图像
        success, img = cap.read()
    else:
        # 从文件读取图像
        img = cv.imread(path)
    # 调整图像大小
    img = cv.resize(img, (width, height))
    # 复制原始图像
    img_copy = img.copy()
    # 将图像转换为灰度图
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 对灰度图进行高斯模糊
    img_blur = cv.GaussianBlur(img_gray, (5, 5), 1)
    # 获取滑动条的值
    result = sf.valueTrackbars(window_info_result)
    # 获取threshold1窗口的滑动条值
    thres = result["threshold1"]

    # 使用Canny边缘检测算法
    img_thres = cv.Canny(img_blur, thres[0], thres[1])
    # 创建一个5x5的卷积核
    kernel = np.ones((5, 5))
    # 对边缘图像进行膨胀操作
    img_dilate = cv.dilate(img_thres, kernel, iterations=2)
    # 对膨胀后的图像进行腐蚀操作
    img_erode = cv.erode(img_dilate, kernel, iterations=1)

    # 复制原始图像用于绘制轮廓
    img_contours = img.copy()
    # 复制原始图像用于绘制最大轮廓
    img_big_contour = img.copy()
    # 查找图像中的轮廓
    contours, hierarchy = cv.findContours(img_erode, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # 在图像上绘制所有轮廓
    cv.drawContours(img_contours, contours, -1, (0, 255, 0), 10)

    # 找到最大的轮廓
    biggest, maxArea = sf.biggestContour(contours)
    # 判断是否找到最大轮廓
    if biggest.size != 0:
        # 对最大轮廓的点进行重新排序
        biggest = sf.reorder(biggest)
        # 在图像上绘制最大轮廓
        cv.drawContours(img_big_contour, biggest, -1, (0, 255, 0), 20)
        # 在图像上绘制最大轮廓的矩形
        img_big_contour = sf.drawRectangle(img_big_contour, biggest, 2)
        # 定义透视变换的源点
        pts1 = np.float32(biggest)
        # 定义透视变换的目标点
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        # 计算透视变换矩阵
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        # 应用透视变换
        img_warpColored = cv.warpPerspective(img, matrix, (width, height))
        # 裁剪透视变换后的图像
        img_warpColored = img_warpColored[5:img_warpColored.shape[0] - 5, 5:img_warpColored.shape[1] - 5]
        # 调整裁剪后图像的大小
        img_warpColored = cv.resize(img_warpColored, (width, height))

        # 裁剪透视变换后图像的感兴趣区域用来标记
        img_warp_rio = img_warpColored
        # 调整感兴趣区域的大小
        img_warp_rio = cv.resize(img_warp_rio, (300,300))
        # 将透视图转换为灰度图
        img_warpGray = cv.cvtColor(img_warpColored, cv.COLOR_BGR2GRAY)
        #使用自适应直方图均衡化增强对比度
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img_warpGray = clahe.apply(img_warpGray)
        # 裁剪灰度图的感兴趣区域
        img_warpGray_roi = cv.resize(img_warpGray, (300,300))
        # # 获取threshold2窗口的滑动条值
        # thres_warp = result["threshold2"]
        # # 对灰度图进行二值化处理
        # img_warpthres = cv.threshold(img_warpGray, thres_warp[0], 255, cv.THRESH_BINARY_INV)[1]

        # # 裁剪二值化后图像的感兴趣区域
        # img_warp_thres_rio = img_warpthres
        # # 调整感兴趣区域的大小
        # img_warp_thres_rio = cv.resize(img_warp_thres_rio, (300,300))
        # # 显示裁剪后的二值化图像
        # cv.imshow("img_warp_thres_rio", img_warp_thres_rio)

        # 将二值化图像分割成多个小方块
        boxs = sf.spiltboxs(img_warpGray_roi, ROWS_BROAD, COLUMNS_BROAD)
        
        if len(boxs) != ROWS_BROAD * COLUMNS_BROAD:
            print(f"棋盘分割异常，期待{ROWS_BROAD*COLUMNS_BROAD}格，实际{len(boxs)}格")
            continue

       # 初始化像素值数组
        myPixelVal = np.zeros((ROWS_BROAD, COLUMNS_BROAD))

        # 初始化行和列索引
        rows = 0
        column = 0
        # 遍历每个小方块
        for box in boxs:
            # # 计算小方块中非零像素的数量
            # totalPixels = cv.countNonZero(box)
            # # 将非零像素数量存储到像素值数组中
            # myPixelVal[rows][column] = totalPixels
            cx, cy = box.shape[1]//2, box.shape[0]//2
            
            # 提取ROI区域
            roi = box[max(0, cy-DETECT_HEIGHT//2):min(box.shape[0], cy+DETECT_HEIGHT//2),
                     max(0, cx-DETECT_WIDTH//2):min(box.shape[1], cx+DETECT_WIDTH//2)]
            
            # # 直方图均衡化(替代histequal)
            # roi = cv.equalizeHist(roi)
            
            # 计算中值(替代a_median())
            # CLAHE局部对比度增强
            # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            # box_clahe = clahe.apply(box)
            values = np.median(roi)
            myPixelVal[rows][column] = values
            # 列索引加1
            column += 1
            # 判断是否到达一行的末尾
            if column == COLUMNS_BROAD:
                # 行索引加1
                rows += 1
                # 列索引重置为0
                column = 0
            

               # 初始化选择索引列表
        myIndex = np.zeros((ROWS_BROAD, COLUMNS_BROAD))
        
        # 使用np.where替代双重循环
        myIndex = np.where(myPixelVal > 155, 2, 
                  np.where(myPixelVal < 90, 1, 0))
        
        # 计算当前白棋数量
        current_person_count = np.sum(myIndex == PERSON_COLOR)
        current_computer_count = np.sum(myIndex == COMPUTER_COLOR)  # 新增黑棋数量统计
        
        # 1. 检测person数量变化（需要持续STABLE_TIME）
        if abs(current_person_count - last_person_count) >= CHANGE_THRESHOLD:
            if quantity_change_start is None:
                quantity_change_start = time.time()  # 记录变化开始时间
                print(f"检测到person数量变化开始，计时中...")
            else:
                if time.time() - quantity_change_start > STABLE_TIME:
                    # 达到稳定时间后输出
                    print(f"检测到持续{STABLE_TIME}秒的person数量变化: 上次{last_person_count}, 当前{current_person_count}")
                    last_person_count = current_person_count  # 确认变化后更新last值
                    last_computer_count = current_computer_count
                    quantity_change_start = None  # 重置计时
        else:
            quantity_change_start = None  # 无变化时重置计时

        # 2. 数量稳定时检测位置变化（需要持续STABLE_TIME）
        if quantity_change_start is None:  
            # 首次进入时记录初始位置
            if initial_position is None:
                initial_position = myIndex.copy()
                position_change_start = None  
                stable_counter = 0  # 新增：稳定帧数计数器
            else:
                # 比较当前状态与初始状态（允许1处差异容错）
                all_changes = np.where(myIndex != initial_position)
                change_count = len(all_changes[0])
                
                if change_count > 0:  # 新增：至少2处变化才认为有效
                    if position_change_start is None:
                        position_change_start = time.time()
                        print(f"检测到位置变化开始（{change_count}处），计时中...")
                        stable_counter = 0  # 重置稳定计数
                    else:
                        # 检查是否达到稳定时间（允许1帧抖动）
                        if time.time() - position_change_start > STABLE_TIME:
                            # 新增：连续2帧确认变化（避免单帧误判）
                            if stable_counter >= 2:
                                print(f"检测到持续{STABLE_TIME}秒的位置变化: {list(zip(all_changes[0], all_changes[1]))}")
                                initial_position = myIndex.copy()  
                                position_change_start = None  
                                stable_counter = 0
                            else:
                                stable_counter += 1
                else:
                    # 差异小于2处，视为噪声，重置记录
                    # initial_position = None
                    position_change_start = None
                    stable_counter = 0

        # 更新上一帧棋盘状态（必须保留）
        myIndex_last = myIndex.copy()

        # 移除原错误的持续更新代码（原last_person_count更新会导致检测失效）
        # last_person_count = current_person_count  
        # last_computer_count = current_computer_count

        #     # 获取当前问题的像素值数组
        #     arr = myPixelVal[x]
        #     # 找到像素值数组中的最大值的索引
        #     myIndexVal = np.where(arr == np.amax(arr))
        #     # 判断最大值是否大于阈值
        #     if arr[myIndexVal[0][0]] > ans_limit:
        #         # 将最大值的索引添加到选择索引列表中
        #         myIndex.append(myIndexVal[0][0])
        #     else:
        #         # 如果最大值小于阈值，添加-1表示未选择
        #         myIndex.append(-1)
        # print(myIndex)
        
                                
        # 打印每个问题的得分情况，可用于调试
        # print(grade)
        # 计算总分，总分 = 所有得分之和 / 问题总数 * 100
        # 打印最终得分，可用于调试
        # print(score)
        # 在最大轮廓图像上绘制得分信息
        # 显示特定图像，这里注释掉了，可根据需要取消注释
        # cv.imshow("test", img_warp_thres_rio)
        # cv.imshow("test", boxs[0])
        
        # 复制感兴趣区域图像，用于显示结果
        img_result = img_warp_rio.copy()
        # 调用自定义函数，在图像上显示答案和得分情况
        img_result, img_original_center = sf.displayResult(img_result, img_big_contour, myIndex, 
                                                           ROWS_BROAD, COLUMNS_BROAD, 
                                                           debug=True, myPixelVal=myPixelVal, matrix=matrix)
        # 显示结果图像
        cv.imshow("result", img_result)
        
        # 定义要堆叠显示的图像矩阵
        imgs = ([img, img_gray, img_thres, img_contours],
            [img_big_contour, img_warpColored, img_warpGray, img_blank],
            [img_result, img_original_center, img_blank, img_blank])    
           
    else:
        # 如果没有找到最大轮廓，使用空白图像填充
        imgs = ([img, img_gray, img_thres, img_contours],
            [img_blank, img_blank, img_blank, img_blank], 
            [img_blank, img_blank, img_blank, img_blank])
    # 定义每个图像对应的标签
    labels = [["oringinal", "gray", "thres", "contours"], 
            ["biggest", "warp colored", "warp gray", "warp thres"],
            ["result", "img_original_center", "blank", "blank"]]
    # 调用自定义函数，将多个图像堆叠显示
    stackedImages = sf.stackImages(0.5, imgs, labels)
    
    # 显示堆叠后的图像
    cv.imshow("stacked images", stackedImages)
    
    # 等待1毫秒，处理用户输入
    cv.waitKey(1)
