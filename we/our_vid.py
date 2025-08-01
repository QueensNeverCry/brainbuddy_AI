import cv2
import time

user_id = "user01"
video_id = "001"
duration_sec = 10  # 녹화 시간
fps = 30
output_path = f"{user_id}_{video_id}.mp4"

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

start_time = time.time()
while time.time() - start_time < duration_sec:
    ret, frame = cap.read()
    if ret:
        out.write(frame)
        cv2.imshow('Recording...', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
