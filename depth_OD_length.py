import depthai as dai
import numpy as np
import cv2

# 파이프라인 생성
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

# 모바일넷 SSD 생성 및 설정
mobile_net_ssd = pipeline.createMobileNetDetectionNetwork()
mobile_net_ssd.setConfidenceThreshold(0.5)
mobile_net_ssd.setBlobPath("mobilenet-ssd_6_shaves.blob")
mobile_net_ssd.input.setBlocking(False)

# Spatial Location Calculator 설정
spatial_location_calculator = pipeline.createSpatialLocationCalculator()
# Spatial Location Calculator 설정
configData = dai.SpatialLocationCalculatorConfigData()
roi = dai.Rect(0.4,0.4,0.2,0.2)
configData.roi = roi
configData.confidenceThreshold = 0.5
configData.mediaType = dai.SensorMediaType.RGB888
configData.waitingForConfigInput = False
spatial_location_calculator.initialConfig = configData
spatial_location_calculator.out.link(xout_spatial_data.input)

# 디바이스와 파이프라인 연결
with dai.Device(pipeline) as device:
    depth_queue = device.getOutputQueue("depth", 8, False)
    spatial_data_queue = device.getOutputQueue("spatialData", 8, False)

    while True:
        depth_frame = depth_queue.get()
        depth_frame_data = np.frombuffer(depth_frame.getData(), dtype=np.uint16).reshape((depth_frame.getHeight(), depth_frame.getWidth()))

        # 스케일링된 깊이 데이터 계산
        min_depth = 100
        max_depth = 7000
        scaled_depth_data = np.clip(depth_frame_data, min_depth, max_depth)

        # OpenCV를 사용하여 깊이 맵을 표시
        depth_frame_colored = cv2.normalize(scaled_depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_frame_colored = cv2.applyColorMap(depth_frame_colored, cv2.COLORMAP_JET)

        too_close_mask = depth_frame_data < min_depth
        too_far_mask = depth_frame_data > max_depth

        depth_frame_colored[too_close_mask] = [255, 255, 255]  # 흰색
        depth_frame_colored[too_far_mask] = [128, 128, 128]    # 회색

        # 객체 인식 및 거리 표시
        detections = mobile_net_ssd.getOutputQueue().tryGetAll()
        if detections is not None:
            detections = detections[0]
            depth_data = depth_queue.get().getData()

            if depth_data is None:
                continue

            depth_data = np.array(depth_data).reshape((3, 320, 544)).astype(np.uint16)

            # Iterate through all the bounding boxes detected
            for detection in detections.detections:
                # Get bounding box coordinates and clip values that are outside the image
                bbox = detection.bbox
                xmin = int(max(0, bbox.xmin * 544))
                ymin = int(max(0, bbox.ymin * 320))
                xmax = int(min(544, bbox.xmax * 544))
                ymax = int(min(320, bbox.ymax * 320))

                # Calculate the center of the bounding box
                center_x = int((xmin + xmax) / 2)
                center_y = int((ymin + ymax) / 2)

                # Get the depth value at the center of the bounding box
                depth = depth_data[0][center_y][center_x]

                # Draw a rectangle around the object and display its label and depth
                cv2.rectangle(depth_frame_colored, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label = mobile_net_ssd.labels[int(detection.label)]
                cv2.putText(depth_frame_colored, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(depth_frame_colored, f"{depth/10} cm", (xmin, ymin + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Depth", depth_frame_colored)

            if cv2.waitKey(1) == ord("q"):
                break
