import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import joblib
import time
import requests
import threading
import pygame
from collections import deque
import os
import glob
import sqlite3
from datetime import datetime
import customtkinter as ctk
from PIL import Image

# ==========================================
# CẤU HÌNH & HẰNG SỐ HỆ THỐNG
# ==========================================
SCALER_PATH = "scaler.pkl"
TELEGRAM_TOKEN = "8363272626:AAEc3L0N1dh8ryGLusdhsAv9PxSejbYbexk"
TELEGRAM_CHAT_ID = "6516145491"
ALARM_SOUND = "siren.wav"
LOG_DIR = "detection_logs"

# AI Hyperparameters
SEQ_LENGTH = 60
CONF_THRESHOLD = 0.3
PREDICT_INTERVAL = 3
SMOOTH_WINDOW = 5
COOLDOWN_TIME = 10
FALL_CONFIRM_FRAMES = 5  # Số frame liên tiếp để xác nhận ngã

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

pygame.mixer.init()
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class FallDetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Hệ Thống Cảnh Báo Té Ngã AI - v2.0 Pro")
        self.geometry("1280x800")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Kết nối Database
        self.db_conn = sqlite3.connect('fall_detection_logs.db', check_same_thread=False)
        self.init_db()

        # AI State & Logic
        self.ort_session = None
        self.input_name = None
        try:
            self.scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            self.after(100, lambda: ctk.messagebox.showerror("Lỗi", f"Không tìm thấy scaler.pkl!\n{e}"))
            self.scaler = None

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5,  model_complexity=1)

        self.keypoint_dict = {0: 0, 1: 2, 2: 5, 3: 7, 4: 8, 5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16, 11: 23, 12: 24,
                              13: 25, 14: 26, 15: 27, 16: 28}

        self.is_running = False
        self.cap = None
        self.current_frame = None
        self.lock = threading.Lock()

        # Biến Video Offline
        self.is_processing_video = False
        self.video_path = None
        self.video_cap = None

        # Buffers
        self.frame_buffer = deque(maxlen=SEQ_LENGTH)
        self.pred_buffer = deque(maxlen=SMOOTH_WINDOW)
        self.prev_base_data = None
        self.last_known_feat = None  # Giữ feature cuối cùng để đệm nếu mất dấu pose
        self.last_alert_time = 0
        self.frame_count = 0
        self.smooth_prediction = 0.0

        # State machine
        self.is_fallen_state = False
        self.fall_frames_count = 0

        self.fps = 0
        self.onnx_latency_ms = 0.0

        # Flags UI
        self.sound_enabled = ctk.BooleanVar(value=True)
        self.telegram_enabled = ctk.BooleanVar(value=True)
        self.privacy_mode = ctk.BooleanVar(value=False)

        self.build_ui()
        self.load_available_models()

    def init_db(self):
        cursor = self.db_conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS fall_events 
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, confidence REAL, path TEXT)''')
        self.db_conn.commit()

    def build_ui(self):
        # UI Setup (Giữ nguyên như cũ của bạn)
        self.tabview = ctk.CTkTabview(self, width=1200, height=750)
        self.tabview.pack(padx=20, pady=20, fill="both", expand=True)

        self.tab_live = self.tabview.add("🎥 GIÁM SÁT TRỰC TUYẾN")
        self.tab_video = self.tabview.add("🎞️ PHÂN TÍCH VIDEO")
        self.tab_history = self.tabview.add("📜 LỊCH SỬ CẢNH BÁO")

        # --- TAB LIVE ---
        self.tab_live.columnconfigure(1, weight=1)
        sidebar = ctk.CTkFrame(self.tab_live, width=280)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(sidebar, text="CẤU HÌNH HỆ THỐNG", font=("Arial", 16, "bold")).pack(pady=20)
        ctk.CTkLabel(sidebar, text="AI Model:").pack(anchor="w", padx=15)
        self.model_combo = ctk.CTkComboBox(sidebar, values=[], command=self.change_model)
        self.model_combo.pack(pady=5, padx=15, fill="x")

        ctk.CTkLabel(sidebar, text="Nguồn Camera:").pack(anchor="w", padx=15, pady=(10, 0))
        self.ip_entry = ctk.CTkEntry(sidebar, placeholder_text="Ví dụ: 0 hoặc http://...")
        self.ip_entry.pack(pady=5, padx=15, fill="x")

        self.btn_run = ctk.CTkButton(sidebar, text="▶ BẮT ĐẦU", fg_color="#0052cc", command=self.run_system)
        self.btn_run.pack(pady=15, padx=15, fill="x")

        self.btn_stop = ctk.CTkButton(sidebar, text="⏹ DỪNG LẠI", fg_color="#cc0000", command=self.stop_system,
                                      state="disabled")
        self.btn_stop.pack(pady=5, padx=15, fill="x")

        settings_frame = ctk.CTkFrame(sidebar, fg_color="#2b2b2b")
        settings_frame.pack(fill="x", padx=15, pady=20)
        ctk.CTkLabel(settings_frame, text="Tùy chọn bổ sung", font=("Arial", 12, "bold")).pack(pady=5)
        ctk.CTkSwitch(settings_frame, text="Còi hú tại chỗ", variable=self.sound_enabled,
                      command=self.toggle_sound).pack(pady=10, anchor="w", padx=15)
        ctk.CTkSwitch(settings_frame, text="Gửi tin Telegram", variable=self.telegram_enabled).pack(pady=10, anchor="w",
                                                                                                    padx=15)
        ctk.CTkSwitch(settings_frame, text="Chế độ riêng tư (Blur)", variable=self.privacy_mode).pack(pady=10,
                                                                                                      anchor="w",
                                                                                                      padx=15)

        self.video_container = ctk.CTkFrame(self.tab_live, fg_color="black", corner_radius=15)
        self.video_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.video_container.grid_rowconfigure(0, weight=1)
        self.video_container.grid_columnconfigure(0, weight=1)
        self.video_label = ctk.CTkLabel(self.video_container, text="CHƯA CÓ TÍN HIỆU VIDEO", font=("Arial", 16))
        self.video_label.grid(row=0, column=0)

        # --- TAB VIDEO ---
        self.tab_video.columnconfigure(0, weight=1)
        self.tab_video.rowconfigure(1, weight=1)

        vid_ctrl_frame = ctk.CTkFrame(self.tab_video)
        vid_ctrl_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        self.btn_select_vid = ctk.CTkButton(vid_ctrl_frame, text="📁 Chọn Video", command=self.select_video)
        self.btn_select_vid.pack(side="left", padx=10, pady=10)
        self.lbl_vid_name = ctk.CTkLabel(vid_ctrl_frame, text="Chưa chọn file...", text_color="gray")
        self.lbl_vid_name.pack(side="left", padx=10, pady=10)

        self.btn_start_vid = ctk.CTkButton(vid_ctrl_frame, text="🚀 Phân Tích", fg_color="green",
                                           command=self.start_video_analysis, state="disabled")
        self.btn_start_vid.pack(side="right", padx=10, pady=10)
        self.btn_stop_vid = ctk.CTkButton(vid_ctrl_frame, text="⏹ Dừng", fg_color="#cc0000",
                                          command=self.stop_video_analysis, state="disabled")
        self.btn_stop_vid.pack(side="right", padx=10, pady=10)

        self.vid_player_container = ctk.CTkFrame(self.tab_video, fg_color="black", corner_radius=10)
        self.vid_player_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.vid_player_container.grid_rowconfigure(0, weight=1)
        self.vid_player_container.grid_columnconfigure(0, weight=1)
        self.vid_player_label = ctk.CTkLabel(self.vid_player_container, text="KHUNG HIỂN THỊ VIDEO", font=("Arial", 16))
        self.vid_player_label.grid(row=0, column=0)

        vid_res_frame = ctk.CTkFrame(self.tab_video, corner_radius=10)
        vid_res_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        self.vid_progress = ctk.CTkProgressBar(vid_res_frame)
        self.vid_progress.pack(fill="x", padx=20, pady=(15, 5))
        self.vid_progress.set(0)

        info_frame = ctk.CTkFrame(vid_res_frame, fg_color="transparent")
        info_frame.pack(fill="x", padx=20, pady=5)
        self.lbl_vid_status = ctk.CTkLabel(info_frame, text="Trạng thái: Sẵn sàng", font=("Arial", 14))
        self.lbl_vid_status.pack(side="left")
        self.lbl_vid_falls = ctk.CTkLabel(info_frame, text="Tổng số lần ngã: 0", font=("Arial", 18, "bold"),
                                          text_color="#ff4d4d")
        self.lbl_vid_falls.pack(side="right")

        # --- TAB LỊCH SỬ ---
        self.setup_history_tab()

    def toggle_sound(self):
        if not self.sound_enabled.get():
            try:
                pygame.mixer.music.stop()
            except:
                pass

    def setup_history_tab(self):
        btn_refresh = ctk.CTkButton(self.tab_history, text="Làm mới dữ liệu", command=self.load_history_to_table)
        btn_refresh.pack(pady=10)
        self.history_frame = ctk.CTkScrollableFrame(self.tab_history, width=1100, height=600)
        self.history_frame.pack(padx=20, pady=10, fill="both", expand=True)
        self.load_history_to_table()

    def load_history_to_table(self):
        for widget in self.history_frame.winfo_children(): widget.destroy()
        headers = ["ID", "Thời gian", "Độ tin cậy", "Hành động"]
        for i, h in enumerate(headers):
            ctk.CTkLabel(self.history_frame, text=h, font=("Arial", 14, "bold"), text_color="#00a8ff").grid(row=0,
                                                                                                            column=i,
                                                                                                            padx=30,
                                                                                                            pady=10,
                                                                                                            sticky="w")

        cursor = self.db_conn.cursor()
        cursor.execute("SELECT * FROM fall_events ORDER BY id DESC LIMIT 50")
        rows = cursor.fetchall()
        for idx, row in enumerate(rows):
            ctk.CTkLabel(self.history_frame, text=row[0]).grid(row=idx + 1, column=0, padx=30, pady=10, sticky="w")
            ctk.CTkLabel(self.history_frame, text=row[1]).grid(row=idx + 1, column=1, padx=30, pady=10, sticky="w")
            ctk.CTkLabel(self.history_frame, text=f"{row[2]:.2f}").grid(row=idx + 1, column=2, padx=30, pady=10,
                                                                        sticky="w")
            btn_view = ctk.CTkButton(self.history_frame, text="Xem ảnh", width=100,
                                     command=lambda p=row[3]: self.show_image_popup(p))
            btn_view.grid(row=idx + 1, column=3, padx=30, pady=10)

    def show_image_popup(self, img_path):
        if not os.path.exists(img_path):
            ctk.messagebox.showerror("Lỗi", "Ảnh không còn tồn tại trên ổ cứng!")
            return
        top = ctk.CTkToplevel(self)
        top.title("Bằng chứng phát hiện ngã")
        top.geometry("680x540")
        top.attributes("-topmost", True)
        img = Image.open(img_path)
        img_ctk = ctk.CTkImage(img, size=(640, 480))
        ctk.CTkLabel(top, image=img_ctk, text="").pack(pady=10)

    # ==========================================
    # LOGIC CHỨC NĂNG UPLOAD VIDEO (ĐÃ SỬA LỖI)
    # ==========================================
    def select_video(self):
        file_path = ctk.filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")])
        if file_path:
            self.video_path = file_path
            self.lbl_vid_name.configure(text=os.path.basename(file_path), text_color="white")
            self.btn_start_vid.configure(state="normal")
            self.vid_progress.set(0)
            self.lbl_vid_falls.configure(text="Tổng số lần ngã: 0")

    def start_video_analysis(self):
        if not self.ort_session or not self.scaler:
            ctk.messagebox.showerror("Lỗi", "Mô hình AI chưa sẵn sàng!")
            return

        self.is_processing_video = True
        self.btn_start_vid.configure(state="disabled")
        self.btn_select_vid.configure(state="disabled")
        self.btn_stop_vid.configure(state="normal")
        self.lbl_vid_status.configure(text="⏳ Đang phân tích...", text_color="yellow")

        threading.Thread(target=self.process_video_task, daemon=True).start()

    def stop_video_analysis(self):
        self.is_processing_video = False
        self.lbl_vid_status.configure(text="⏹ Đã dừng phân tích", text_color="orange")
        self.btn_start_vid.configure(state="normal")
        self.btn_select_vid.configure(state="normal")
        self.btn_stop_vid.configure(state="disabled")

    def process_video_task(self):
        self.video_cap = cv2.VideoCapture(self.video_path)
        total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0: video_fps = 30

        v_frame_buffer = deque(maxlen=SEQ_LENGTH)
        v_pred_buffer = deque(maxlen=SMOOTH_WINDOW)
        v_prev_base = None
        v_last_feat = None

        # State machine
        v_is_fallen = False
        v_fall_frames = 0
        total_falls = 0
        v_frame_count = 0

        while self.is_processing_video and self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if not ret: break

            v_frame_count += 1
            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)

            display_frame = rgb.copy()
            status_t = f"BUFFERING ({len(v_frame_buffer)}/{SEQ_LENGTH})"
            status_c, box_c = (0, 255, 255), (0, 255, 0)
            current_prob = 0.0

            if results.pose_landmarks:
                h, w, _ = frame.shape
                landmarks = results.pose_landmarks.landmark
                data = []

                for i in range(17):
                    lm = landmarks[self.keypoint_dict[i]]
                    if lm.visibility < CONF_THRESHOLD and v_prev_base is not None:
                        data.extend([v_prev_base[i * 2], v_prev_base[i * 2 + 1]])
                    else:
                        data.extend([lm.x * w, lm.y * h])

                data = np.array(data)
                vel = (data - v_prev_base) if v_prev_base is not None else np.zeros(34)
                v_prev_base = data.copy()

                m_hip_y = (data[23] + data[25]) / 2
                neck_y = (data[11] + data[13]) / 2
                neck_x = (data[10] + data[12]) / 2
                m_hip_x = (data[22] + data[24]) / 2
                ang = np.array([np.arctan2(neck_y - m_hip_y, neck_x - m_hip_x)])

                feat = np.concatenate([data, vel, [m_hip_y], ang])
                v_last_feat = feat
                v_frame_buffer.append(feat)

                # Vẽ bộ khung xương
                l_spec = self.mp_drawing.DrawingSpec(color=status_c[::-1], thickness=3, circle_radius=3)
                c_spec = self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                self.mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                               l_spec, c_spec)

            else:
                # Nếu mất dấu khung xương, tái sử dụng dữ liệu cũ để tránh gãy buffer
                if v_last_feat is not None:
                    v_frame_buffer.append(v_last_feat)

            # Dự đoán nếu buffer đã đầy
            if len(v_frame_buffer) == SEQ_LENGTH:
                if v_frame_count % PREDICT_INTERVAL == 0:
                    try:
                        inp = np.expand_dims(self.scaler.transform(np.array(v_frame_buffer)), 0).astype(np.float32)
                        out = self.ort_session.run(None, {self.input_name: inp})[0][0]
                        v_pred_buffer.append(float(out[1] if len(out) > 1 else out[0]))
                    except Exception as e:
                        print(f"Lỗi Inference: {e}")

                current_prob = sum(v_pred_buffer) / len(v_pred_buffer) if v_pred_buffer else 0

                # Logic nhận diện bằng State Machine (Máy trạng thái)
                if current_prob > 0.6:
                    v_fall_frames += 1
                elif current_prob < 0.3:
                    v_fall_frames = 0  # Reset nếu prob thấp rõ rệt
                    v_is_fallen = False  # Đã đứng dậy an toàn

                if v_fall_frames >= FALL_CONFIRM_FRAMES and not v_is_fallen:
                    # Bắt đầu rơi vào trạng thái ngã -> Ghi nhận 1 lần
                    v_is_fallen = True
                    total_falls += 1

                # Hiển thị text theo trạng thái
                if v_is_fallen:
                    status_t, status_c, box_c = "FALL DETECTED!", (255, 0, 0), (255, 0, 0)
                else:
                    status_t, status_c, box_c = "SAFE", (0, 255, 0), (0, 255, 0)

            # Vẽ UI lên frame
            cv2.rectangle(display_frame, (0, 0), (640, 60), (0, 0, 0), -1)
            cv2.putText(display_frame, status_t, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, status_c, 2)
            cv2.putText(display_frame, f"Prob: {current_prob:.2f}", (450, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                        (255, 255, 255), 1)

            # Cập nhật UI từ Thread
            if v_frame_count % 2 == 0:  # Chỉnh thành 2: Hiện ảnh nhiều hơn để mắt thấy mượt
                # Resize ảnh nhỏ xuống một xíu trước khi nhét vào UI để giảm lag đồ họa
                ui_frame = cv2.resize(display_frame, (480, 360))
                img = Image.fromarray(ui_frame)
                imgtk = ctk.CTkImage(img, size=(480, 360))
                self.after(0, self.update_vid_ui, imgtk, v_frame_count, total_frames, total_falls)


            # time.sleep(1 / video_fps * 0.4)  # Control playback speed

        if self.video_cap: self.video_cap.release()
        self.is_processing_video = False

        if v_frame_count >= total_frames - 10:
            self.after(0, self.finish_vid_ui, total_falls)

    def update_vid_ui(self, imgtk, current_frame, total_frames, total_falls):
        self.vid_player_label.configure(image=imgtk, text="")
        if total_frames > 0:
            self.vid_progress.set(current_frame / total_frames)
        self.lbl_vid_falls.configure(text=f"Tổng số lần ngã: {total_falls}")

    def finish_vid_ui(self, total_falls):
        self.lbl_vid_status.configure(text="✅ Hoàn thành phân tích", text_color="green")
        self.lbl_vid_falls.configure(text=f"Tổng số lần ngã: {total_falls}")
        self.btn_start_vid.configure(state="normal")
        self.btn_select_vid.configure(state="normal")
        self.btn_stop_vid.configure(state="disabled")
        self.vid_progress.set(1.0)

    # ==========================================
    # LOGIC CAMERA LIVE (CŨNG ĐÃ UPDATE STATE MACHINE)
    # ==========================================
    def load_available_models(self):
        models = glob.glob("*.onnx")
        if models:
            self.model_combo.configure(values=models)
            self.model_combo.set(models[0])
            self.change_model(models[0])

    def change_model(self, model_path):
        try:
            self.ort_session = ort.InferenceSession(model_path)
            self.input_name = self.ort_session.get_inputs()[0].name
        except Exception as e:
            print(f"Lỗi load model: {e}")

    def run_system(self):
        if not self.ort_session or not self.scaler:
            ctk.messagebox.showerror("Lỗi", "Kiểm tra lại Model và Scaler!")
            return

        src = self.ip_entry.get().strip()
        self.cap = cv2.VideoCapture(int(src) if src.isdigit() else (src if src else 0))

        if not self.cap.isOpened():
            ctk.messagebox.showerror("Lỗi", "Không thể kết nối Camera!")
            return

        self.is_running = True
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")

        # Reset buffers
        self.frame_buffer.clear()
        self.pred_buffer.clear()
        self.fall_frames_count = 0
        self.frame_count = 0
        self.is_fallen_state = False

        threading.Thread(target=self.process_thread, daemon=True).start()
        self.update_gui_loop()

    def process_thread(self):
        prev_t = time.time()
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: continue

            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            display_frame = rgb.copy()
            if self.privacy_mode.get():
                display_frame = cv2.GaussianBlur(display_frame, (99, 99), 30)

            results = self.pose.process(rgb)
            status_t = f"BUFFERING ({len(self.frame_buffer)}/{SEQ_LENGTH})"
            status_c, box_c = (0, 255, 255), (0, 255, 0)

            if results.pose_landmarks:
                h, w, _ = frame.shape
                landmarks = results.pose_landmarks.landmark
                data = []

                for i in range(17):
                    lm = landmarks[self.keypoint_dict[i]]
                    if lm.visibility < CONF_THRESHOLD and self.prev_base_data is not None:
                        data.extend([self.prev_base_data[i * 2], self.prev_base_data[i * 2 + 1]])
                    else:
                        data.extend([lm.x * w, lm.y * h])

                data = np.array(data)
                vel = (data - self.prev_base_data) if self.prev_base_data is not None else np.zeros(34)
                self.prev_base_data = data.copy()

                m_hip_y = (data[23] + data[25]) / 2
                neck_y = (data[11] + data[13]) / 2
                neck_x = (data[10] + data[12]) / 2
                m_hip_x = (data[22] + data[24]) / 2
                ang = np.array([np.arctan2(neck_y - m_hip_y, neck_x - m_hip_x)])

                feat = np.concatenate([data, vel, [m_hip_y], ang])
                self.last_known_feat = feat
                self.frame_buffer.append(feat)

                l_spec = self.mp_drawing.DrawingSpec(color=status_c[::-1], thickness=3, circle_radius=3)
                c_spec = self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                self.mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                               l_spec, c_spec)

            else:
                if self.last_known_feat is not None:
                    self.frame_buffer.append(self.last_known_feat)

            if len(self.frame_buffer) == SEQ_LENGTH:
                if self.frame_count % PREDICT_INTERVAL == 0:
                    t_onnx = time.time()
                    try:
                        inp = np.expand_dims(self.scaler.transform(np.array(self.frame_buffer)), 0).astype(np.float32)
                        out = self.ort_session.run(None, {self.input_name: inp})[0][0]
                        self.onnx_latency_ms = (time.time() - t_onnx) * 1000
                        self.pred_buffer.append(float(out[1] if len(out) > 1 else out[0]))
                    except Exception as e:
                        print(f"Lỗi Inference: {e}")

                self.smooth_prediction = sum(self.pred_buffer) / len(self.pred_buffer) if self.pred_buffer else 0

                # Logic Live State Machine
                if self.smooth_prediction > 0.6:
                    self.fall_frames_count += 1
                elif self.smooth_prediction < 0.3:
                    self.fall_frames_count = 0
                    self.is_fallen_state = False

                if self.fall_frames_count >= FALL_CONFIRM_FRAMES and not self.is_fallen_state:
                    self.is_fallen_state = True
                    if time.time() - self.last_alert_time > COOLDOWN_TIME:
                        self.last_alert_time = time.time()
                        sound_status = self.sound_enabled.get()
                        tele_status = self.telegram_enabled.get()
                        threading.Thread(target=self.alert_event, args=(
                        cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR), self.smooth_prediction, sound_status,
                        tele_status), daemon=True).start()

                if self.is_fallen_state:
                    status_t, status_c, box_c = "DANGER: FALL DETECTED!", (255, 0, 0), (255, 0, 0)
                else:
                    status_t, status_c, box_c = "SAFE", (0, 255, 0), (0, 255, 0)

            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)

            self.fps = 1 / (time.time() - prev_t) if (time.time() - prev_t) > 0 else 0
            prev_t = time.time()
            self.frame_count += 1

            cv2.putText(display_frame, status_t, (30, 45), cv2.FONT_HERSHEY_DUPLEX, 0.9, status_c, 2)
            cv2.putText(display_frame,
                        f"Prob: {self.smooth_prediction:.2f} | FPS: {int(self.fps)} | AI: {self.onnx_latency_ms:.1f}ms",
                        (30, 80), 2, 0.6, (255, 255, 255), 1)

            with self.lock:
                self.current_frame = display_frame.copy()

    def alert_event(self, frame, prob, is_sound_on, is_tele_on):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        path = os.path.join(LOG_DIR, f"fall_{datetime.now().strftime('%H%M%S')}.jpg")
        cv2.imwrite(path, frame)

        self.db_conn.execute("INSERT INTO fall_events (timestamp, confidence, path) VALUES (?,?,?)",
                             (timestamp, prob, path))
        self.db_conn.commit()

        if is_sound_on:
            try:
                if os.path.exists(ALARM_SOUND):
                    pygame.mixer.music.load(ALARM_SOUND)
                    pygame.mixer.music.play(loops=-1)
                else:
                    print(f"[LỖI AUDIO] KHÔNG TÌM THẤY FILE '{ALARM_SOUND}'!")
            except Exception as e:
                print(f"[LỖI AUDIO] {e}")

        if is_tele_on:
            try:
                with open(path, 'rb') as photo_file:
                    res = requests.post(
                        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                        files={'photo': photo_file},
                        data={'chat_id': TELEGRAM_CHAT_ID,
                              'caption': f"🚨 CẢNH BÁO NGÃ!\n⏰ Thời gian: {timestamp}\n📊 Độ tin cậy: {prob:.2f}"},
                        timeout=10
                    )
            except Exception as e:
                print(f"[LỖI TELEGRAM] {e}")

    def update_gui_loop(self):
        if self.is_running:
            with self.lock:
                if self.current_frame is not None:
                    img = Image.fromarray(self.current_frame)
                    imgtk = ctk.CTkImage(img, img, size=(800, 600))
                    self.video_label.configure(image=imgtk, text="")
            self.after(30, self.update_gui_loop)

    def stop_system(self):
        self.is_running = False
        if self.cap: self.cap.release()
        self.btn_run.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.video_label.configure(image="", text="HỆ THỐNG TẮT")
        try:
            pygame.mixer.music.stop()
        except:
            pass

    def on_closing(self):
        self.is_running = False
        self.is_processing_video = False
        self.db_conn.close()
        try:
            pygame.mixer.music.stop()
        except:
            pass
        self.destroy()


if __name__ == "__main__":
    app = FallDetectionApp()
    app.mainloop()