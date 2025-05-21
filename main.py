# 导入OpenCV库，使用别名cv
import cv2 as cv
# 导入NumPy库，用于数值计算
import numpy as np
# 导入自定义的扫描功能模块
import scanFunctions as sf
import time
# import serial
# 导入自定义的串口通信模块
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
# 设置摄像头帧率（添加在亮度设置后）
cap.set(cv.CAP_PROP_FPS, 30)  # 限制最大30帧/秒
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))  # 设置编码格式为MJPG

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

STABLE_TIME_POINT = 5

# 定义答案像素值的阈值
ans_limit = 4000
# 定义变化检测的阈值
CHANGE_THRESHOLD = 1
# 定义正确答案列表
ans = [1,1,1]
# 定义答案像素值的阈值
ans_limit = 4000

point_h_thres = 7
point_v_thres = 7
#################################

#################################
# 初始化计数器
last_computer_count = 0
last_person_count = 0
quantity_change_start = None  
position_change_start = None  
initial_position = None  # 记录位置变化前的初始状态
 # 在变量初始化部分添加
last_sent_coord = None
last_sent_time = 0
# 初始化滑动条
sf.initTrackbars(window_info)
#################################

# 初始化计算fps
prev_time = time.time()
# 人执的棋子颜色
PERSON_COLOR = 2
# 机器执的棋子颜色
COMPUTER_COLOR = 1

# 初始化串口通信
# ser = serial.Serial('COM3', 9600, timeout=1)  # 替换为你的串口号和波特率
# 检查串口是否打开
# if ser.is_open:
#     print("串口已打开")

# buffer = bytearray()  # 添加缓冲区用于存储接收数据
# while True:
#     data = ser.read(ser.in_waiting or 1)  # 读取所有可用字节
#     if data:
#         buffer.extend(data)
#         # 检查完整帧结构 0x00 0x00 [有效字节] 0xff
#         while len(buffer) >= 4:  # 至少需要4字节判断结构
#             # 查找帧头
#             start = buffer.find(b'\x00\x00')
#             if start != -1 and len(buffer) >= start + 3:
#                 # 找到有效数据字节（第三个字节）
#                 if buffer[start + 3] == 0xff:  # 验证帧尾
#                     middle_byte = buffer[start + 2]
#                     print(f"接收到中间字节: 0x{middle_byte:02x}")
#                     PERSON_COLOR = middle_byte
#                     COMPUTER_COLOR = 2 if PERSON_COLOR == 1 else 1
#                     del buffer[:start+4]  # 移除已处理数据
#                     break
#                 else:
#                     del buffer[:start+1]  # 无效帧头，跳过第一个00
#             else:
#                 break  # 没有有效帧头继续接收
#     else:
#         time.sleep(0.01)  # 添加短暂延时防止CPU占满
#     if PERSON_COLOR in [1, 2]:  # 收到有效数据后退出循环
#         break

    
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
    # 计算fps
    fps, prev_time = sf.calculate_fps(prev_time)
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
        # 将二值化图像分割成多个小方块
        boxs = sf.spiltboxs(img_warpGray_roi, ROWS_BROAD, COLUMNS_BROAD)
        
        if len(boxs) != ROWS_BROAD * COLUMNS_BROAD:
            print(f"棋盘分割异常，期待{ROWS_BROAD*COLUMNS_BROAD}格，实际{len(boxs)}格")
            continue

       # 初始化像素值数组
        myPixelVal = np.zeros((ROWS_BROAD, COLUMNS_BROAD))

        if len(boxs) == ROWS_BROAD * COLUMNS_BROAD:
        # 将列表转为三维数组 (ROWS_BROAD, COLUMNS_BROAD, H, W)
            box_array = np.array(boxs).reshape(ROWS_BROAD, COLUMNS_BROAD, *boxs[0].shape)
        
            # 向量化计算中心区域（比循环快10倍以上）
            cy, cx = box_array.shape[2]//2, box_array.shape[3]//2
            rois = box_array[:, :, 
                            cy-DETECT_HEIGHT//2:cy+DETECT_HEIGHT//2,
                            cx-DETECT_WIDTH//2:cx+DETECT_WIDTH//2]
            myPixelVal = np.median(rois, axis=(2,3))
            
            # 使用向量化条件判断
            myIndex = np.select(
                [myPixelVal > 155, myPixelVal < 90],
                [2, 1],
                default=0
            )

        
        # 计算当前棋子数量、位置变化
        current_person_count = np.sum(myIndex == PERSON_COLOR) # 人执
        current_computer_count = np.sum(myIndex == COMPUTER_COLOR) # 机器执
        
        # 1. 检测person数量变化（需要持续STABLE_TIME）
        if abs(current_person_count - last_person_count) >= CHANGE_THRESHOLD:
            if quantity_change_start is None:
                quantity_change_start = time.time()  # 记录变化开始时间
                print(f"检测到person数量变化开始，计时中...")
            else:
                if time.time() - quantity_change_start > STABLE_TIME:
                    # 找到新增的人执棋子位置
                    new_positions = np.where((myIndex == PERSON_COLOR) & (initial_position != PERSON_COLOR))
                    new_positions = (new_positions[0].astype(int), new_positions[1].astype(int))
                    
                    # 输出每个新棋子的位置
                    for row, col in zip(new_positions[0], new_positions[1]):
                        print(f"人下在: 行{row+1} 列{col+1}")
                    
                    
                    last_person_count = current_person_count  # 确认变化后更新last值
                    last_computer_count = current_computer_count
                    quantity_change_start = None  # 重置计时
                    initial_position = myIndex.copy()  # 更新初始状态
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
                # 比较当前状态与初始状态
                # 计算变化的位置数量（不包括0）
                all_changes = np.where(myIndex != initial_position)
                all_changes = (all_changes[0].astype(int), all_changes[1].astype(int))
                change_count = len(all_changes[0])
                
                if change_count % 2 == 0 and change_count > 1:  
                    if position_change_start is None:
                        position_change_start = time.time()
                        print(f"检测到位置变化开始（{change_count}处），计时中...")
                        stable_counter = 0  
                    else:
                        if time.time() - position_change_start > STABLE_TIME:
                            if stable_counter >= 2:
                                # 只输出新位置（状态非0的坐标）
                                new_positions = np.where(myIndex != 0)
                                for row, col in zip(new_positions[0], new_positions[1]):
                                    if myIndex[row][col] != initial_position[row][col]:
                                        print(f"新位置: 行{row+1} 列{col+1} 颜色{myIndex[row][col]}")
                                
                                initial_position = myIndex.copy()
                                position_change_start = None
                                stable_counter = 0
                            else:
                                stable_counter += 1
                else:
                    # 位置未变化，重置计数器
                    # initial_position = None
                    position_change_start = None
                    stable_counter = 0

        
        # 复制感兴趣区域图像，用于显示结果
        img_result = img_warp_rio.copy()
        # 调用自定义函数，在图像上显示答案和得分情况
        img_result, img_original_center, orignal_point = sf.displayResult(img_result, img_big_contour, myIndex, 
                                                           ROWS_BROAD, COLUMNS_BROAD, 
                                                           debug=True, myPixelVal=myPixelVal, matrix=matrix)
        # 修改坐标检测部分（替换原有复杂逻辑）
        current_time = time.time()
        if orignal_point and all(p != (0, 0) for p in orignal_point):
            # 首次发送或满足稳定条件
            if last_sent_coord is None or \
            (abs(orignal_point[0][0] - last_sent_coord[0][0]) >= point_h_thres and
                abs(orignal_point[0][1] - last_sent_coord[0][1]) >= point_v_thres and
                current_time - last_sent_time > STABLE_TIME_POINT):
                
                # 构造数据包（小端序）
                for i in range(ROWS_BROAD*COLUMNS_BROAD):
                    data = b'\x00' + \
                        orignal_point[i][0].to_bytes(2, 'little') + \
                        orignal_point[i][1].to_bytes(2, 'little') + \
                        b'\xff'
                    
                    # ser.write(data)
                last_sent_coord = orignal_point
                last_sent_time = current_time
                print(f"坐标已发送: {orignal_point}")
            

        # 在结果图像上显示FPS
        cv.putText(img_original_center, f"FPS: {int(fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # 显示结果图像
        cv.imshow("ori_ctr", img_original_center)
        
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
