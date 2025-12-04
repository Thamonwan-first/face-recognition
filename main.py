import cv2
import face_recognition
import sys

sys.stdout.reconfigure(encoding='utf-8')

# ----------------ส่วนตั้งค่า----------------
# ใส่ชื่อไฟล์รูปภาพ และ ชื่อคน ตรงนี้ครับ
image_file = "me.jpg"       # ชื่อไฟล์รูปต้นฉบับ
person_name = "first"     # ชื่อที่จะให้แสดงบนจอ
# ----------------------------------------

print(f"กำลังเรียนรู้ใบหน้าจากไฟล์ {image_file} ...")

try:
    # 1. โหลดภาพและถอดรหัสใบหน้า (Encoding)
    your_image = face_recognition.load_image_file(image_file)
    your_face_encoding = face_recognition.face_encodings(your_image)[0]
    
    # เก็บข้อมูลไว้ในลิสต์ (ถ้ามีหลายคนก็เพิ่มตรงนี้ได้)
    known_face_encodings = [your_face_encoding]
    known_face_names = [person_name]
    
    print("จำหน้าเสร็จแล้ว! กำลังเปิดกล้อง...")

except IndexError:
    print("Error: ไม่พบใบหน้าในรูปภาพต้นฉบับ กรุณาเปลี่ยนรูปใหม่")
    exit()
except FileNotFoundError:
    print(f"Error: หาไฟล์ {image_file} ไม่เจอ")
    exit()

# 2. เปิดกล้อง
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # ลดขนาดภาพลง 1/4 เพื่อให้ทำงานเร็ว ไม่กระตุก
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # ค้นหาใบหน้าในกล้อง
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # เทียบใบหน้าที่เจอกับฐานข้อมูล
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown" # ถ้าไม่รู้จักให้ขึ้น Unknown

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    # วาดกรอบและชื่อ
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # ขยายขนาดกลับ (x4)
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # เลือกสี: สีเขียว(รู้จัก) / สีแดง(ไม่รู้จัก)
        if name == "Unknown":
            color = (0, 0, 255) # แดง
        else:
            color = (0, 255, 0) # เขียว

        # วาดกรอบ
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # เขียนชื่อ
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Face Recognition System', frame)

    # กด 'q' เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()