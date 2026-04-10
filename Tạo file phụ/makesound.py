import wave
import struct
import math

# Thông số file âm thanh
sample_rate = 44100.0
duration = 3.0  # Kêu trong 3 giây
file_name = "siren.wav"

print(f"Đang tạo file {file_name}...")
wavef = wave.open(file_name, 'w')
wavef.setnchannels(1)  # Mono
wavef.setsampwidth(2)  # 16-bit
wavef.setframerate(sample_rate)

# Tạo hiệu ứng còi hú (cao - thấp xen kẽ)
for i in range(int(duration * sample_rate)):
    # Đổi tần số mỗi 0.5 giây để tạo tiếng còi "Nino Nino"
    if (i // (sample_rate / 2)) % 2 == 0:
        freq = 800.0  # Âm cao
    else:
        freq = 600.0  # Âm thấp

    value = int(32767.0 * math.sin(freq * math.pi * float(i) / sample_rate))
    data = struct.pack('<h', value)
    wavef.writeframesraw(data)

wavef.close()
print(f"Thành công! Đã tạo xong file {file_name} tại thư mục hiện tại.")