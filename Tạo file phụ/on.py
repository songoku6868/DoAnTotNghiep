import tensorflow as tf
import tf2onnx
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Masking, LayerNormalization

# 1. Build lại model giống hệt code cũ
model = Sequential([
    Masking(mask_value=-99.0, input_shape=(60, 70)),
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
    LayerNormalization(),
    Dropout(0.3),
    Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.2)),
    LayerNormalization(),
    Dropout(0.4),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# 2. Load weights
model.load_weights("best_model.h5")

# 3. Convert sang ONNX
spec = (tf.TensorSpec((None, 60, 70), tf.float32, name="input"),)
output_path = "best_model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
print(f"✅ Đã lưu model ONNX tại: {output_path}")