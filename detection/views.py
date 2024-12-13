from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from collections import defaultdict

video_path = "detection/cars2.mp4"  
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    raise IOError("Error: Could not open video file.")

vehicle_counts = defaultdict(int)
frame_skip = 5
frame_counter = 0
movement_threshold = 75
tracked_vehicles = {}
next_vehicle_id = 1

# Track if video processing is complete
video_processing_complete = False  



# calculating centroid to make sure same vehicle is not counted more than once
def get_centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)



def calculate_distance(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

def generate_frames():
    global frame_counter, next_vehicle_id, tracked_vehicles, vehicle_counts, video_processing_complete

    while True:
        ret, frame = video.read()
        if not ret:
            video_processing_complete = True
            break

        frame_counter += 1
        if frame_counter % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (640, 360))

        # Crop the bottom part of the frame
        height, width, _ = frame.shape
        # Keep the bottom 50% of the frame
        cropped_frame = frame[int(height * 0.5):, :]  

        # Detect objects in the cropped frame
        bbox, label, conf = cv.detect_common_objects(cropped_frame) 

        # Adjust bounding boxes to match the original frame coordinates
        adjusted_bbox = [
            [x1, y1 + int(height * 0.5), x2, y2 + int(height * 0.5)] for x1, y1, x2, y2 in bbox
        ]

        # Draw bounding boxes on the original frame
        output_image = draw_bbox(frame, adjusted_bbox, label, conf)

        # Tracking centroids of current frame
        current_centroids = []
        for i, detected_label in enumerate(label):
            # Filter vehicle labels
            if detected_label in ["car", "truck", "bus", "motorcycle"]:  
                detected_box = adjusted_bbox[i]
                centroid = get_centroid(detected_box)
                current_centroids.append((centroid, detected_label))

        # Update tracked vehicles and count new ones
        updated_tracked_vehicles = {}
        for centroid, detected_label in current_centroids:
            matched = False
            for vehicle_id, (prev_centroid, label) in tracked_vehicles.items():
                if calculate_distance(centroid, prev_centroid) < movement_threshold and label == detected_label:
                    updated_tracked_vehicles[vehicle_id] = (centroid, detected_label)
                    matched = True
                    break
            if not matched:
                # New vehicle detected
                updated_tracked_vehicles[next_vehicle_id] = (centroid, detected_label)
                vehicle_counts[detected_label] += 1
                next_vehicle_id += 1

        # Update the tracked vehicles
        tracked_vehicles = updated_tracked_vehicles

        ret, buffer = cv2.imencode('.jpg', output_image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def index(request):
    return render(request, 'detection/index.html', {'vehicle_counts': dict(vehicle_counts)})

def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def get_vehicle_counts(request):
    global vehicle_counts, video_processing_complete
    return JsonResponse({'vehicle_counts': dict(vehicle_counts), 'complete': video_processing_complete})
