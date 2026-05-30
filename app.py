from flask import Flask, render_template, Response, request, send_file, jsonify
import cv2
import os
import csv
import math
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from sort import Sort

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
LOG_FOLDER = "logs"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ------------------ LOAD CLASSES ------------------
with open("classes.txt", "r") as f:
    classNames = f.read().splitlines()

# ------------------ MODEL ------------------
model = YOLO("yolov8n.pt")

VEHICLE_CLASSES = [2, 3, 5, 7]

# ------------------ TRACKER ------------------
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# ------------------ GLOBALS ------------------
video_path = None

vehicle_ids = set()
counts = {"car": 0, "bus": 0, "truck": 0, "bike": 0}

prev_positions = {}
speed_dict = {}

frame_counts = {}

# -------- NEW (LINE CROSSING) --------
LINE_Y = 300
crossed_ids = set()

# -------- SETTINGS --------
MIN_FRAMES = 10
MIN_AREA = 5000
PIXEL_TO_METER = 0.05

# ------------------ RESET ------------------
def reset_all():
    global vehicle_ids, counts, prev_positions, speed_dict, frame_counts, crossed_ids

    vehicle_ids.clear()
    prev_positions.clear()
    speed_dict.clear()
    frame_counts.clear()
    crossed_ids.clear()

    counts = {"car": 0, "bus": 0, "truck": 0, "bike": 0}

    open("logs/vehicle_count.csv", "w").close()
    open("logs/vehicle_speed.csv", "w").close()

# ------------------ ROUTES ------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global video_path

    file = request.files["file"]
    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)

    reset_all()

    return "Uploaded"

# ------------------ DETECTION ------------------
def detect(frame):
    results = model(frame, stream=True)
    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            if cls in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                detections.append([x1, y1, x2, y2, conf, cls])

    return detections

# ------------------ FRAME GENERATOR ------------------
def generate_frames():
    global video_path

    if not video_path or not os.path.exists(video_path):
        while True:
            frame = 255 * np.ones((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Upload a video first", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return

    cap = cv2.VideoCapture(video_path)

    # -------- REAL FPS --------
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    while True:
        success, frame = cap.read()
        if not success:
            break

        detections = detect(frame)

        dets = []
        for d in detections:
            dets.append(d[:5])

        tracks = tracker.update(np.array(dets))

        # -------- DRAW LINE --------
        cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0,0,255), 2)

        for t in tracks:
            x1, y1, x2, y2, track_id = map(int, t)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # -------- ORIGINAL CLASS MATCH --------
            cls = 2
            for d in detections:
                dx1, dy1, dx2, dy2, _, dcls = d
                if x1 >= dx1 and y1 >= dy1 and x2 <= dx2 and y2 <= dy2:
                    cls = dcls
                    break

            vehicle_type = classNames[cls]

            if vehicle_type == "motorbike":
                vehicle_type = "bike"

            # -------- FRAME COUNT --------
            if track_id not in frame_counts:
                frame_counts[track_id] = 1
            else:
                frame_counts[track_id] += 1

            # -------- AREA --------
            area = (x2 - x1) * (y2 - y1)

            # -------- DELAYED LOGGING --------
            if (
                track_id not in vehicle_ids and
                frame_counts[track_id] >= MIN_FRAMES and
                area >= MIN_AREA
            ):
                vehicle_ids.add(track_id)

                if vehicle_type in counts:
                    counts[vehicle_type] += 1

                with open("logs/vehicle_count.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([track_id, vehicle_type, datetime.now()])

            # -------- SPEED (LINE CROSSING ONLY) --------
            if track_id in prev_positions:
                px, py = prev_positions[track_id]

                distance = math.hypot(cx - px, cy - py)
                current_speed = distance * PIXEL_TO_METER * fps * 3.6

                # Detect crossing
                if track_id not in crossed_ids:
                    if py < LINE_Y and cy >= LINE_Y:
                        crossed_ids.add(track_id)
                        speed_dict[track_id] = current_speed

            prev_positions[track_id] = (cx, cy)

            speed = speed_dict.get(track_id, 0)

            # -------- DRAW --------
            label = f"ID {track_id} | {vehicle_type} | {int(speed)} km/h"

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # -------- SAVE SPEED LOG --------
    with open("logs/vehicle_speed.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Max Speed", "Timestamp"])

        for vid, spd in speed_dict.items():
            writer.writerow([vid, round(spd,2), datetime.now()])

# ------------------ ROUTES ------------------
@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/stats")
def stats():
    return jsonify({
        "total": sum(counts.values()),
        **counts
    })

@app.route("/speed_data")
def speed_data():
    data = []

    for vid, spd in speed_dict.items():
        data.append({
            "id": vid,
            "speed": round(spd, 2),
            "overspeed": spd > 40   
        })

    return jsonify(data)

@app.route("/download_count")
def download_count():
    return send_file("logs/vehicle_count.csv", as_attachment=True)

@app.route("/download_speed")
def download_speed():
    return send_file("logs/vehicle_speed.csv", as_attachment=True)

# ------------------
if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    app.run(debug=True)