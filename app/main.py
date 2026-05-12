import cv2
import numpy as np
import base64
import json
import os
import sqlite3
import threading
import requests
import asyncio
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fall_logic import FallDetector

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

LOG_DIR = "detection_logs"
VIDEO_DIR = "uploaded_videos"
OUTPUT_DIR = "output_videos"

for directory in [LOG_DIR, VIDEO_DIR, OUTPUT_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

app.mount("/logs", StaticFiles(directory=LOG_DIR), name="logs")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

db_conn = sqlite3.connect('fall_detection_logs.db', check_same_thread=False)
cursor = db_conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS fall_events 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, confidence REAL, path TEXT)''')
db_conn.commit()

TELEGRAM_TOKEN = "8363272626:AAEc3L0N1dh8ryGLusdhsAv9PxSejbYbexk"
TELEGRAM_CHAT_ID = "6516145491"

cpu_executor = ThreadPoolExecutor(max_workers=4)


def alert_event(frame_bytes, prob, is_tele_on):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = f"fall_{datetime.now().strftime('%H%M%S')}.jpg"
    path = os.path.join(LOG_DIR, filename)

    with open(path, "wb") as f:
        f.write(frame_bytes)

    cursor.execute("INSERT INTO fall_events (timestamp, confidence, path) VALUES (?,?,?)", (timestamp, prob, filename))
    db_conn.commit()

    if is_tele_on:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                files={'photo': ("alert.jpg", frame_bytes, "image/jpeg")},
                data={'chat_id': TELEGRAM_CHAT_ID, 'caption': f"🚨 CẢNH BÁO NGÃ!\n⏰ {timestamp}\n📊 Prob: {prob:.2f}"},
                timeout=5
            )
        except Exception:
            pass


@app.get("/")
async def index():
    with open(os.path.join("templates", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/history")
def get_history():
    cursor.execute("SELECT * FROM fall_events ORDER BY id DESC LIMIT 50")
    rows = cursor.fetchall()
    return JSONResponse(content=[{"id": r[0], "time": r[1], "prob": r[2], "img": f"/logs/{r[3]}"} for r in rows])


# --- WEBSOCKET CAMERA LIVE ---
@app.websocket("/ws/live")
async def live_stream(websocket: WebSocket):
    await websocket.accept()
    detector = FallDetector()
    last_alert_time = 0
    loop = asyncio.get_event_loop()

    try:
        while True:
            raw_data = await websocket.receive_text()
            data = json.loads(raw_data)

            _, encoded = data["image"].split(",", 1)
            img_bytes = base64.b64decode(encoded)
            frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

            if frame is None: continue

            processed_frame, is_fall, prob, is_fallen_state = await loop.run_in_executor(
                cpu_executor, detector.process_frame, frame, data.get("privacy", False)
            )

            _, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])

            curr_time = datetime.now().timestamp()
            if is_fall and (curr_time - last_alert_time > 10):
                last_alert_time = curr_time
                threading.Thread(target=alert_event, args=(buffer.tobytes(), prob, data["telegram"])).start()

            await websocket.send_json({
                "image": "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8'),
                "is_fallen_state": is_fallen_state,
                "prob": prob
            })
    except WebSocketDisconnect:
        pass


# --- API UPLOAD VIDEO OFFLINE ---
@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(VIDEO_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename}


@app.websocket("/ws/process_video/{filename}")
async def process_video_ws(websocket: WebSocket, filename: str):
    await websocket.accept()
    input_path = os.path.join(VIDEO_DIR, filename)
    cap = cv2.VideoCapture(input_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps: fps = 25

    out_filename = f"processed_{filename.split('.')[0]}.mp4"
    output_path = os.path.join(OUTPUT_DIR, out_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

    detector = FallDetector()
    total_falls = 0
    current_frame = 0
    fall_probs = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.resize(frame, (640, 480))
            current_frame += 1

            processed_frame, is_fall, prob, is_fallen_state = detector.process_frame(frame, False)

            cv2.putText(processed_frame, f"Fall Prob: {prob:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),
                        2)
            if is_fallen_state:
                cv2.putText(processed_frame, "FALL DETECTED!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            out_video.write(processed_frame)

            if is_fall:
                total_falls += 1
            if is_fallen_state:
                fall_probs.append(prob)

            if current_frame % 2 == 0:
                _, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                await websocket.send_json({
                    "image": "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8'),
                    "progress": int(current_frame / total_frames * 100),
                    "falls": total_falls,
                    "prob": prob,
                    "is_fallen_state": is_fallen_state
                })
                await websocket.receive_text()

        # Tính MAX PROB thay vì AVG để show hội đồng
        max_prob = (max(fall_probs) * 100) if len(fall_probs) > 0 else 0.0

        await websocket.send_json({
            "done": True,
            "falls": total_falls,
            "max_prob": round(max_prob, 2),
            "video_url": f"/outputs/{out_filename}"
        })
    except WebSocketDisconnect:
        pass
    finally:
        cap.release()
        out_video.release()


@app.delete("/api/delete_record/{record_id}")
def delete_record(record_id: int):
    # Lấy tên file ảnh từ database trước khi xóa
    cursor.execute("SELECT path FROM fall_events WHERE id = ?", (record_id,))
    row = cursor.fetchone()

    if row:
        filename = row[0]
        file_path = os.path.join(LOG_DIR, filename)

        # Xóa file ảnh trong thư mục (nếu nó tồn tại)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

        # Xóa dữ liệu trong database
        cursor.execute("DELETE FROM fall_events WHERE id = ?", (record_id,))
        db_conn.commit()

        return {"status": "success", "message": f"Đã xóa bản ghi #{record_id}"}

    return JSONResponse(status_code=404, content={"message": "Không tìm thấy dữ liệu"})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)