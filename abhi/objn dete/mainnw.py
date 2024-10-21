import cv2 as cv
import queue
import threading
from tracker import EuclideanDistTracker

# Initialize a queue to hold frames
frame_queue = queue.Queue(maxsize=10)

# Function to receive frames
def receive_frames():
    cap = cv.VideoCapture('road.mp4')
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Frame could not be read. Exiting...")
            break
        if frame_queue.full():
            frame_queue.get()  # Remove the oldest frame to make room for the new one
        frame_queue.put(frame)
    cap.release()

# Function to process and display frames
def process_and_display():
    tracker = EuclideanDistTracker()
    object_detector = cv.createBackgroundSubtractorMOG2(history=50, varThreshold=30)
    limits = [600, 1000, 1100, 1000]
    totalcount = []

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Resize the frame to a slightly larger size
            frame = cv.resize(frame, (780, 500))

            # Calculate the ROI
            roi = calculate_center_roi(frame)

            if roi.size == 0:
                continue

            # Apply background subtraction
            mask = object_detector.apply(roi)
            _, mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)

            # Find contours
            contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            detection = []
            for cnt in contours:
                area = cv.contourArea(cnt)
                if area > 2000:
                    x, y, w, h = cv.boundingRect(cnt)
                    detection.append([x, y, w, h])

            # Update tracker and draw detections
            boxes_id = tracker.update(detection)
            for box_id in boxes_id:
                x, y, w, h, id = box_id
                cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cx, cy = (x + w // 2), (y + h // 2)
                if limits[0] < cx + 500 < limits[2] and limits[1] - 20 < cy + 600 < limits[1] + 20:
                    if id not in totalcount:
                        totalcount.append(id)
                        cv.line(roi, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 2)

            # Display count on the frame
            cv.putText(frame, f'Count: {len(totalcount)}', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv.line(frame, (limits[0] // 3, limits[1] // 3), (limits[2] // 3, limits[3] // 3), (255, 0, 255), 2)

            cv.imshow('Frame', roi)
            if cv.waitKey(0) & 0xFF == ord('q'):  # Press 'q' to quit
                break

    cv.destroyAllWindows()

# Function to calculate ROI
def calculate_center_roi(frame, roi_width=500, roi_height=400, shift_left=250):
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
