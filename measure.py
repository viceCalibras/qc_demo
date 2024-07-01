import cv2
import depthai as dai
import numpy as np

def isolate_cubes(input_frame: np.ndarray):
    # Resize & convert to greyscale to speed up processing.
    frame = cv2.resize(input_frame,(640,480))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frame


# Create pipeline.
pipeline = dai.Pipeline()

# Define sources and output nodes.
cam_rgb = pipeline.create(dai.node.ColorCamera)
xout_rgb = pipeline.create(dai.node.XLinkOut)

# Set node properties.
xout_rgb.setStreamName("rgb")
cam_rgb.setPreviewSize(1200, 640)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P) # THE_4_K is also an option.

# Link the nodes.
cam_rgb.preview.link(xout_rgb.input)

# Connect to device and start pipeline.
with dai.Device(pipeline) as device:
    # Get the content of output queue & visualize the image.
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        in_rgb = q_rgb.tryGet()  # Non-blocking call, will return a new data that has arrived or None otherwise.
        if in_rgb is not None:
            frame = in_rgb.getCvFrame() 
            processed_frame = isolate_cubes(frame)
            cv2.imshow("rgb", processed_frame)

        if cv2.waitKey(1) == ord('q'):
            break