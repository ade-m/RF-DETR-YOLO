import cv2
import numpy as np
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import supervision as sv

# Load model RF-DETR
model = RFDETRBase()

def resize_frame(frame, scale=0.5):
    return cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

def rotate_frame(frame, angle=270):
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

# Buka video input
video_path = "asset/lain/IMG_4210.MOV"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Tidak dapat membuka video.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Video selesai

    frame = rotate_frame(frame, angle=90)
    frame = resize_frame(frame, scale=0.25)

    # Konversi ke format PIL
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detections = model.predict(image, threshold=0.5)

    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    # Anotasi
    annotated_image = sv.BoxAnnotator().annotate(image.copy(), detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

    # Tampilkan realtime di layar
    frame_display = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
    cv2.imshow("RF-DETR Real-time Detection", frame_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1
    print(f"Frame {frame_count} diproses.")

cap.release()
cv2.destroyAllWindows()
print("✅ Proses selesai. Window ditutup.")
