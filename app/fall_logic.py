import cv2
import numpy as np
import onnxruntime as ort
import joblib
from collections import deque
from ultralytics import YOLO


class FallDetector:
    def __init__(self):
        self.scaler = joblib.load("scaler.pkl")
        self.ort_session = ort.InferenceSession("best_bilstm_model.onnx")
        self.input_name = self.ort_session.get_inputs()[0].name
        self.yolo = YOLO('yolo26n-pose.pt')

        self.SEQ_LENGTH = 60
        self.CONF_THRESHOLD = 0.3
        self.PREDICT_INTERVAL = 3
        self.SMOOTH_WINDOW = 5
        self.FALL_CONFIRM_FRAMES = 3

        self.frame_buffer = deque(maxlen=self.SEQ_LENGTH)
        self.pred_buffer = deque(maxlen=self.SMOOTH_WINDOW)
        self.prev_base_data = None
        self.last_known_feat = None

        self.frame_count = 0
        self.fall_frames_count = 0
        self.is_fallen_state = False
        self.smooth_prediction = 0.0

        self.skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                         [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                         [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

    def process_frame(self, frame, is_privacy=False):
        self.frame_count += 1
        display_frame = frame.copy()

        if is_privacy:
            display_frame = cv2.GaussianBlur(display_frame, (99, 99), 30)

        is_fall_detected = False
        valid_pose = False
        draw_data = None
        draw_confs = None

        # ==========================================
        # BƯỚC 1: LẤY DỮ LIỆU TỪ YOLO
        # ==========================================
        results = self.yolo.predict(frame, verbose=False)

        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0 and len(
                results[0].keypoints.xy[0]) > 0:
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            confs = results[0].keypoints.conf[0].cpu().numpy()

            data = []
            current_confs = []

            for i in range(17):
                x, y = keypoints[i]
                conf = confs[i]
                current_confs.append(conf)

                # Logic nhồi data cho AI Bi-LSTM
                if conf < self.CONF_THRESHOLD and self.prev_base_data is not None:
                    data.extend([self.prev_base_data[i * 2], self.prev_base_data[i * 2 + 1]])
                else:
                    data.extend([x, y])

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

            # Lưu lại dữ liệu để lát nữa vẽ (sau khi AI dự đoán xong)
            valid_pose = True
            draw_data = data
            draw_confs = current_confs

        else:
            if self.last_known_feat is not None:
                self.frame_buffer.append(self.last_known_feat)

        # ==========================================
        # BƯỚC 2: CHẠY AI DỰ ĐOÁN & CẬP NHẬT TRẠNG THÁI
        # ==========================================
        if len(self.frame_buffer) == self.SEQ_LENGTH:
            if self.frame_count % self.PREDICT_INTERVAL == 0:
                try:
                    inp = np.expand_dims(self.scaler.transform(np.array(self.frame_buffer)), 0).astype(np.float32)
                    out = self.ort_session.run(None, {self.input_name: inp})[0][0]
                    self.pred_buffer.append(float(out[1] if len(out) > 1 else out[0]))
                except Exception as e:
                    pass

            self.smooth_prediction = sum(self.pred_buffer) / len(self.pred_buffer) if self.pred_buffer else 0

            if self.smooth_prediction > 0.5:
                self.fall_frames_count += 1
            elif self.smooth_prediction < 0.3:
                self.fall_frames_count = 0
                self.is_fallen_state = False

            if self.fall_frames_count >= self.FALL_CONFIRM_FRAMES and not self.is_fallen_state:
                self.is_fallen_state = True
                is_fall_detected = True

        # ==========================================
        # BƯỚC 3: VẼ KHUNG XƯƠNG DỰA TRÊN TRẠNG THÁI MỚI
        # ==========================================
        if valid_pose:
            # Lúc này self.is_fallen_state đã mang giá trị MỚI NHẤT sau khi AI tính toán
            status_c = (0, 0, 255) if self.is_fallen_state else (0, 255, 0)

            for link in self.skeleton:
                pt1_idx, pt2_idx = link[0], link[1]

                if draw_confs[pt1_idx] > self.CONF_THRESHOLD and draw_confs[pt2_idx] > self.CONF_THRESHOLD:
                    pt1 = (int(draw_data[pt1_idx * 2]), int(draw_data[pt1_idx * 2 + 1]))
                    pt2 = (int(draw_data[pt2_idx * 2]), int(draw_data[pt2_idx * 2 + 1]))
                    if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                        cv2.line(display_frame, pt1, pt2, status_c, 3)
                        cv2.circle(display_frame, pt1, 4, (255, 255, 255), -1)

        return display_frame, is_fall_detected, self.smooth_prediction, self.is_fallen_state