import cv2
import depthai as dai
import numpy as np

def isolate_bricks(input_frame: np.ndarray):
    """Isolate LEGO bricks from the background.

    Args:
        input_frame (np.ndarray): Input frame of the camera.

    Returns:
        Frame with isolated LEGO bricks.
    """
    # Convert to greyscale to speed up processing.
    frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)

    # Blur the images to get rid of sharp edges/outlines (further simplification).
    frame = cv2.GaussianBlur(frame, (21, 21), 0)

    # Normalize frame to the 0-255 range & change the data type
    # (some of the following operations are expecting greyscale image with uint8 pixels):
    frame_max = np.max(frame)
    frame_min = np.min(frame)
    norm_frame = (frame - frame_min) * 255 / (frame_max - frame_min)
    norm_frame = norm_frame.astype(np.uint8)

    # Apply Otsu's threshold to separate the foreground from the background.
    thresh = cv2.threshold(norm_frame, 0, 1, cv2.THRESH_OTSU)[1]
    background_mask = np.zeros(frame.shape, dtype=np.uint8)
    background_mask[thresh == 0] = 1  # Used to filter out the background.
    background_mask.astype(np.uint8)
    # Remove background from the frame:
    frame[background_mask == 1] = 0

    # Convert into a binary image to clean the noise.
    binary = cv2.threshold(frame, 25, 90, cv2.THRESH_BINARY)[1]
    binary = binary.astype(np.uint8)

    # Do a morphological closing operation to better isolate outer contours of the bricks.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    cubes = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=6)
    
    return cubes

def measure_and_label(input_frame: np.ndarray, frame: np.ndarray):
    """Isolates the LEGO bricks contours & outputs their basic dimensions.

    Args:
        input_frame (np.ndarray): Input frame. Has to be greyscale!
        frame (np.ndarray): Input RGB frame. 

    Returns:
        RGB frame with dimensions.
    """
    # Find contours or continuous white clusters(blobs) in the image
    # print(input_frame)
    contours, _ = cv2.findContours(input_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print("HI!")
    # print(contours)
    # Draw a bounding box around the cubes.
    measured_frame = input_frame
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        measured_frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        measured_frame = cv2.putText(measured_frame , "Defect", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) 
        # rect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(rect)
        # box = box.astype(np.int8)
        # measured_frame = cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
    if measured_frame is not None:
        return measured_frame

if __name__== "__main__":
    # Create pipeline.
    pipeline = dai.Pipeline()

    # Define sources and output nodes.
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    xout_rgb = pipeline.create(dai.node.XLinkOut)

    # Set node properties.
    xout_rgb.setStreamName("rgb")
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P) # THE_4_K is also an option.

    # Link the nodes.
    cam_rgb.preview.link(xout_rgb.input)

    # Connect to device and start pipeline.
    with dai.Device(pipeline) as device:
        # Get the content of output queue.
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        while True:
            in_rgb = q_rgb.tryGet()  # Non-blocking call, will return a new data that has arrived or None otherwise.
            if in_rgb is not None:
                # Extract the frame, process, measure and visualize.
                frame = in_rgb.getCvFrame() 
                processed_frame = isolate_bricks(frame)
                if processed_frame is not None:
                    # contours, _ = cv2.findContours(processed_frame , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # print(contours)
                    measured_frame = measure_and_label(processed_frame, frame)
                    cv2.imshow("processing", processed_frame)
                    # cv2.imshow("raw", frame)
                    cv2.imshow("measurement", measured_frame)

            if cv2.waitKey(1) == ord('q'):
                break