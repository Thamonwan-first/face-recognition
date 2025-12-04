import cv2
import face_recognition
import sys
import os
import numpy as np  # <--- 1. ต้องเพิ่มบรรทัดนี้เพื่อช่วยคำนวณหาคนที่เหมือนที่สุด

# ---------------------------------------------------------
sys.stdout.reconfigure(encoding='utf-8')
# ---------------------------------------------------------

folder_path = "faces"
known_face_encodings = []
known_face_names = []

print(f"📂 กำลังอ่านรูปภาพทั้งหมดจากโฟลเดอร์ '{folder_path}' ...")

if not os.path.exists(folder_path):
    print(f"❌ ไม่พบโฟลเดอร์ {folder_path} กรุณาสร้างโฟลเดอร์และใส่รูปภาพก่อน")
    os.makedirs(folder_path)
    exit()

for filename in os.listdir(folder_path):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(folder_path, filename)
        name = os.path.splitext(filename)[0]
        
        try:
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(name)
            print(f"✔️  จดจำ: {name}")
        except IndexError:
            print(f"⚠️  ข้ามไฟล์: {filename} (หาใบหน้าไม่เจอ)")
        except Exception as e:
            print(f"❌ Error: {e}")

print(f"✅ เสร็จสิ้น! จำหน้าได้ทั้งหมด {len(known_face_names)} คน")
print("---------------------------------------")
print("📷 กำลังเปิดกล้อง...")

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # ค้นหาตำแหน่งและ encoding ของทุกคนในภาพปัจจุบัน
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    
    # วนลูปเช็ค "ทุกคน" ที่เจอในกล้องตอนนี้
    for face_encoding in face_encodings:
        # 2. ส่วนที่แก้ไข: ใช้ face_distance เพื่อหาคนที่เหมือนที่สุด
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # คำนวณความห่างของใบหน้า (ยิ่งเลขน้อย ยิ่งเหมือนมาก)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        if len(face_distances) > 0:
            # หา index ของคนที่ face_distance น้อยที่สุด (เหมือนที่สุด)
            best_match_index = np.argmin(face_distances)
            
            # ถ้าคนที่เหมือนที่สุด อยู่ในเกณฑ์ที่ match กันด้วย
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        face_names.append(name)

    # วาดกรอบสี่เหลี่ยมให้ทุกคนที่เจอ
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # ถ้าไม่รู้จักเป็นสีแดง รู้จักเป็นสีเขียว
        if name == "Unknown":
            color = (0, 0, 255) 
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Face Recognition Multi-Person', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()