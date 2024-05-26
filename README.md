# YOLOv8 Projects

This repository contains various projects using the YOLOv8 model for tasks such as object detection, segmentation, tracking, pose estimation, counting, and customer detection. Each project is designed to demonstrate the capabilities of YOLOv8 in real-world applications.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Projects](#projects)
  - [Object Detection](#object-detection)
  - [Object Segmentation](#object-segmentation)
  - [Object Tracking](#object-tracking)
  - [Pose Estimation](#pose-estimation)
  - [Object Counting](#object-counting)
  - [Customer Detection](#customer-detection)
- [Contributing](#contributing)
- [License](#license)

## Introduction

YOLO (You Only Look Once) is a state-of-the-art model for object detection and segmentation. This repository provides implementations of various YOLOv8 functionalities using the Ultralytics YOLO library and OpenCV.

## Installation

To get started, you need to install the required libraries. You can do this using `pip`:

```bash
pip install ultralytics opencv-python
```

Ensure you have Python installed on your machine. This project is compatible with Python 3.7 and above.

## Projects

### Object Detection

This project demonstrates real-time object detection using YOLOv8.

**Script**: `object_detection.py`

```python
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize the video capture object
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform inference on the frame
    results = model(frame)

    # Annotate the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Object Segmentation

This project demonstrates real-time object segmentation using YOLOv8.

**Script**: `object_segmentation.py`

```python
import cv2
from ultralytics import YOLO

# Load the YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')

# Initialize the video capture object
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform inference on the frame
    results = model(frame)

    # Annotate the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('YOLOv8 Segmentation', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Object Tracking

This project demonstrates real-time object tracking using YOLOv8.

**Script**: `object_tracking.py`

```python
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize the video capture object
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

tracker = cv2.TrackerKCF_create()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform inference on the frame
    results = model(frame)

    # Annotate the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('YOLOv8 Tracking', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Pose Estimation

This project demonstrates real-time pose estimation using YOLOv8.

**Script**: `pose_estimation.py`

```python
import cv2
from ultralytics import YOLO

# Load the YOLOv8 pose estimation model
model = YOLO('yolov8n-pose.pt')

# Initialize the video capture object
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform inference on the frame
    results = model(frame)

    # Annotate the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('YOLOv8 Pose Estimation', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Object Counting

This project demonstrates real-time object counting using YOLOv8.

**Script**: `object_counting.py`

```python
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize the video capture object
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform inference on the frame
    results = model(frame)

    # Count the number of detected objects
    num_objects = len(results[0].boxes)

    # Annotate the frame with the count
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, f'Count: {num_objects}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the annotated frame
    cv2.imshow('YOLOv8 Object Counting', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Customer Detection

This project demonstrates real-time customer detection in a retail environment using YOLOv8.

**Script**: `customer_detection.py`

```python
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model trained for customer detection
model = YOLO('yolov8n-custom.pt')

# Initialize the video capture object
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform inference on the frame
    results = model(frame)

    # Annotate the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('YOLOv8 Customer Detection', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`.
3. Make your changes and commit them: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature-branch`.
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
