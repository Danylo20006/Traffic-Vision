"""
Traffic Vision: Multi-Object Tracking & Vehicle Counting Pipeline
This module processes video streams, tracks vehicles across frames using ByteTrack, 
and implements state management logic to count objects crossing a virtual line.
"""

import cv2
from ultralytics import YOLO

# Load our custom YOLO model fine-tuned specifically on traffic data
model = YOLO('models/best.pt')

# Initialize the video stream. 
# Explicitly using the FFmpeg backend ensures stable and robust video decoding.
cap = cv2.VideoCapture(
    'data/train/video/video3.mp4',
    cv2.CAP_FFMPEG
)

# Log video metadata for debugging and pipeline verification
print("Video opened:", cap.isOpened())
print("Video FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Total frames:", cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize counting variables.
# Using a Set data structure guarantees O(1) lookup time for IDs and prevents double-counting.
count = 0
counted_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("End of video stream.")
        break

    height, width, channels = frame.shape

    # Define the spatial threshold (a horizontal virtual line exactly in the middle of the frame)
    line_y = int(height / 2)

    # Run inference and tracking.
    # persist=True instructs the tracker to maintain object identities (Track IDs) across consecutive frames.
    results = model.track(frame, persist=True, tracker='bytetrack.yaml')

    # Robustness check: Ensure the tracker has successfully assigned IDs to detected objects in the current frame.
    # This prevents pipeline crashes (e.g., 'NoneType object is not iterable' exceptions).
    if results[0].boxes.id is not None:

        # Extract bounding box coordinates and their corresponding unique tracking IDs
        boxes = results[0].boxes.xyxy
        ids = results[0].boxes.id

        for box, object_id in zip(boxes, ids):            
            
            x_min, y_min, x_max, y_max = box
            object_id = int(object_id)
            
            # Calculate the Y-coordinate of the object's center for precise crossing detection
            y_center = (y_min + y_max) / 2

            # Crossing Logic (State Management):
            # If the object's center crosses the line AND the object's ID hasn't been counted yet
            if y_center > line_y and object_id not in counted_ids:
                count += 1
                counted_ids.add(object_id)

    # Generate a frame with YOLO bounding boxes, labels, and tracking IDs drawn
    annotated_frame = results[0].plot()

    # Draw the virtual counting line (Red)
    cv2.line(
        annotated_frame,
        (0, line_y), 
        (width, line_y), 
        (0, 0, 255), 
        2
    )
    
    # Display the real-time vehicle counter on the screen (Green text)
    cv2.putText(
        annotated_frame, 
        f'Count: {count}', 
        (20, 40), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0), 
        2
    )

    cv2.imshow("Vehicle Counting", annotated_frame)

    # Exit condition: Pressing 'q' safely terminates the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release hardware resources and close GUI windows
cap.release()
cv2.destroyAllWindows()