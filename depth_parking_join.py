import depthai as dai
import cv2
import numpy as np
import pickle
import cvzone

# Load position data
with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

width, height = 107, 48

def checkParkingSpace(imgPro):
    spaceCounter = 0

    for pos in posList:
        x, y = pos

        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)

        if count < 900:
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 2

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1,
                           thickness=2, offset=0, colorR=color)

    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                       thickness=5, offset=20, colorR=(0, 200, 0))

# OAK-D 파이프라인 생성
pipeline = dai.Pipeline()

# 카메라와 디코더 노드 설정
mono_left = pipeline.createMonoCamera()
mono_right = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()
xout_depth = pipeline.createXLinkOut()

# 카메라 파라미터 설정
mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)

# 스테레오 디코더 설정
initialConfig = stereo.initialConfig.setConfidenceThreshold(255)
initialConfig = stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_3x3)

# 링크 설정
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)
stereo.depth.link(xout_depth.input)

xout_depth.setStreamName("depth")

min_depth = 700  # 최소 깊이 (mm)
max_depth = 8000  # 최대 깊이 (mm)

# 디바이스와 파이프라인 연결
with dai.Device(pipeline) as device:
    depth_queue = device.getOutputQueue("depth", 8, False)

    while True:
        depth_frame = depth_queue.get()
        depth_frame_data = np.frombuffer(depth_frame.getData(), dtype=np.uint16).reshape((depth_frame.getHeight(), depth_frame.getWidth()))

        # 스케일링된 깊이 데이터 계산
        scaled_depth_data = np.clip(depth_frame_data, min_depth, max_depth)

        # OpenCV를 사용하여 깊이 맵을 표시
        depth_frame_colored = cv2.normalize(scaled_depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 빨간색에서 파란색으로 변환하는 컬러 맵 생성
        depth_frame_colored = cv2.applyColorMap(255 - depth_frame_colored, cv2.COLORMAP_JET)

        # 수정된 부분: min_depth 밑의 값을 흰색, max_depth 밖의 값을 회색으로 표시하는 마스크 생성
        too_close_mask = depth_frame_data < min_depth
        too_far_mask = depth_frame_data > max_depth

        # 마스크를 깊이 맵에 적용
        depth_frame_colored[too_close_mask] = [255, 255, 255]  # 흰색
        depth_frame_colored[too_far_mask] = [128, 128, 128]    # 회색

        img = depth_frame_colored

        # 빨간색 영역의 마스크 생성
        lower_red = np.array([0, 0, 200])
        upper_red = np.array([80, 80, 255])
        red_mask = cv2.inRange(img, lower_red, upper_red)

        # 빨간색 영역에 주차 공간 정보 표시
        checkParkingSpace(red_mask)

        cv2.imshow("Depth", depth_frame_colored)

        if cv2.waitKey(1) == ord("q"):
            break

