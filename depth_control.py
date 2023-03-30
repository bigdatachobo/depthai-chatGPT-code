import depthai as dai
import cv2
import numpy as np

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
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

# 스테레오 디코더 설정
initialConfig = stereo.initialConfig.setConfidenceThreshold(200)
initialConfig = stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)

# 링크 설정
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)
stereo.depth.link(xout_depth.input)

xout_depth.setStreamName("depth")

min_depth = 2000  # 최소 깊이 (mm)
max_depth = 5000  # 최대 깊이 (mm)

# 디바이스와 파이프라인 연결
with dai.Device(pipeline) as device:
    depth_queue = device.getOutputQueue("depth", 8, False)

    while True:
        depth_frame = depth_queue.get()
        depth_frame_data = np.frombuffer(depth_frame.getData(), dtype=np.uint16).reshape((depth_frame.getHeight(), depth_frame.getWidth()))

        # 스케일링된 깊이 데이터 계산
        scaled_depth_data = np.interp(depth_frame_data, (depth_frame_data.min(), depth_frame_data.max()), (min_depth, max_depth))

        # OpenCV를 사용하여 깊이 맵을 표시
        depth_frame_colored = cv2.normalize(scaled_depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_frame_colored = cv2.applyColorMap(depth_frame_colored, cv2.COLORMAP_JET)

        cv2.imshow("Depth", depth_frame_colored)

        if cv2.waitKey(1) == ord("q"):
            break
