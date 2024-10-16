import cv2
import queue
import threading
from tracker import EuclideanDistTracker

# Reduce queue size to minimize memory usage
frame_queue = queue.Queue(maxsize=2)

# Function to receive frames
def receive_frames():
    cap = cv2.VideoCapture('road.mp4')
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Frame could not be read. Exiting...")
            break
        # Resize frame for faster processing on OlinXino
        frame = cv2.resize(frame, (480, 320))
        frame_queue.put(frame)
    cap.release()

# Function to process and display frames
def process_and_display():
    tracker = EuclideanDistTracker()
    object_detector = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=20)
    limits = [300, 500, 550, 500]  # Adjust based on resized frame
    totalcount = []

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Calculate ROI with reduced size
            roi = calculate_center_roi(frame, roi_width=300, roi_height=200, shift_left=150)

            if roi.size == 0:
                continue

            # Apply background subtraction with lower thresholds
            mask = object_detector.apply(roi)
            _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)

            # Find contours with a smaller minimum area
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            detection = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 1000:  # Adjust minimum area based on object size
                    x, y, w, h = cv2.boundingRect(cnt)
                    detection.append([x, y, w, h])

            # Update tracker and draw detections
            boxes_id = tracker.update(detection)
            for box_id in boxes_id:
                x, y, w, h, id = box_id
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cx, cy = (x + w // 2), (y + h // 2)
                if limits[0] < cx + 300 < limits[2] and limits[1] - 10 < cy + 200 < limits[1] + 10:
                    if id not in totalcount:
                        totalcount.append(id)
                        cv2.line(roi, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 2)

            # Display count on the frame
            cv2.putText(frame, f'Count: {len(totalcount)}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.line(frame, (limits[0] // 3, limits[1] // 3), (limits[2] // 3, limits[3] // 3), (255, 0, 255), 2)

            cv2.imshow('ROI', roi)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

    cv2.destroyAllWindows()

# Function to calculate ROI
def calculate_center_roi(frame, roi_width=300, roi_height=200, shift_left=150):
    frame_height, frame_width, _ = frame.shape
    center_x = frame_width // 2 - shift_left
    center_y = frame_height // 2
    top_left_x = max(center_x - roi_width // 2, 0)
    top_left_y = max(center_y - roi_height // 2, 0)
    bottom_right_x = min(top_left_x + roi_width, frame_width)
    bottom_right_y = min(top_left_y + roi_height, frame_height)
    return frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

if __name__ == '__main__':
    # Start the threads
    receive_thread = threading.Thread(target=receive_frames)
    process_thread = threading.Thread(target=process_and_display)

    receive_thread.start()
    process_thread.start()

    receive_thread.join()
    process_thread.join()