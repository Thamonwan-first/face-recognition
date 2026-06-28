---
name: face-recognition-pi5
description: คู่มือการติดตั้งและใช้งานระบบเช็คชื่อด้วยใบหน้าพร้อมเว็บแอปบน Raspberry Pi 5 (Python GUI + Node.js Backend)
---

# คู่มือการติดตั้งและเปิดใช้งานระบบบน Raspberry Pi 5

เอกสารนี้ระบุขั้นตอนการจัดเตรียมสภาพแวดล้อม (Environment Setup), การรันโปรแกรมระบบเช็คชื่อด้วยใบหน้า (Python Tkinter GUI) และระบบเว็บแอปพลิเคชัน (Node.js) บนบอร์ด **Raspberry Pi 5** รวมถึงวิธีตั้งค่าให้โปรแกรมทำงานอัตโนมัติทันทีที่เปิดเครื่อง (Autorun)

---

## 📋 สิ่งที่ต้องเตรียม (Prerequisites)
1. **บอร์ด Raspberry Pi 5** ที่ติดตั้งระบบปฏิบัติการ **Raspberry Pi OS (64-bit) Bookworm**
2. **กล้อง USB Webcam** หรือ **Raspberry Pi Camera Module** (ต่อและตั้งค่าทดสอบผ่าน `rpicam-hello` เรียบร้อยแล้ว)
3. การเชื่อมต่อเครือข่ายภายใน (Local Wi-Fi หรือ LAN)

---

## 🛠️ ขั้นตอนที่ 1: ติดตั้งและตั้งค่า Node.js (Web App Backend)
ระบบเช็คชื่อใช้เว็บแอป Node.js พอร์ต 5000 เป็นฐานข้อมูลหลักและรับการสแกน

1. **ติดตั้ง Node.js (แนะนำ v20 หรือ v22 LTS ขึ้นไป):**
   ```bash
   # ดาวน์โหลดและติดตั้ง Node Source
   curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
   sudo apt-get install -y nodejs
   ```

2. **ทดสอบเวอร์ชันของ Node.js และ npm:**
   ```bash
   node -v
   npm -v
   ```

3. **ติดตั้ง Package ของ Web App:**
   เข้าไปที่โฟลเดอร์ `web_app` แล้วรันคำสั่งติดตั้ง dependencies:
   ```bash
   cd /home/pi/face_recognition/web_app
   npm install
   ```

4. **ทดสอบรัน Web Server:**
   ```bash
   node server.js
   ```
   *หน้าเว็บแอปจะทำงานและพร้อมให้เข้าใช้ที่ลิงก์ `http://<IP-ของ-Pi-5>:5000`*

---

## 🐍 ขั้นตอนที่ 2: ตั้งค่า Python Environment บน Pi 5
เนื่องจาก Raspberry Pi OS Bookworm บังคับใช้ระบบความปลอดภัยในการติดตั้ง Library ผ่าน pip (PEP 668) จึงจำเป็นต้องรันโปรแกรมผ่าน Virtual Environment:

1. **สร้าง Python Virtual Environment:**
   ```bash
   cd /home/pi/face_recognition
   python3 -m venv --system-site-packages env
   ```

2. **เปิดใช้งาน Virtual Environment:**
   ```bash
   source env/bin/activate
   ```

3. **ติดตั้ง Dependency Libraries ที่จำเป็น:**
   ```bash
   pip install --upgrade pip
   pip install opencv-python face-recognition numpy requests pillow
   ```
   *(หมายเหตุ: หากบอร์ด Pi 5 ยังไม่มีไลบรารีสำหรับประมวลผลใบหน้า dlib สามารถติดตั้งผ่านไฟล์ล้อ .whl ที่อยู่ในโฟลเดอร์โครงการได้ โดยสั่ง: `pip install dlib-19.22.99-cp310-cp310-win_amd64.whl` หรือ compile dlib สดผ่านคำสั่ง `pip install dlib`)*

4. **ทดสอบเปิดใช้งานโปรแกรมสแกนใบหน้า:**
   ```bash
   python main.py
   ```

---

## 🚀 ขั้นตอนที่ 3: ตั้งค่าให้เปิดโปรแกรมอัตโนมัติเมื่อเปิดเครื่อง (Autorun)
ในการนำไปใช้งานจริง (เช่น เครื่องตู้คีออสสแกนหน้า) เราต้องการให้ระบบเปิดขึ้นมาเองโดยไม่ต้องคีย์คำสั่งใน Terminal:

### 1. ตั้งค่า Autorun สำหรับ Web App (รันเบื้องหลังด้วย PM2)
PM2 เป็นเครื่องมือจัดการโปรเซสยอดนิยมที่จะช่วยรัน Node.js เสมอเมื่อเปิดเครื่อง

```bash
# ติดตั้ง PM2 ทั่วทั้งระบบ
sudo npm install -g pm2

# สั่งรันเว็บแอปผ่าน PM2
cd /home/pi/face_recognition/web_app
pm2 start server.js --name "attendance-web"

# ตั้งค่าให้ PM2 สตาร์ทเองทุกครั้งเมื่อบอร์ดบูต
pm2 startup
# (ทำตามคำสั่ง sudo env... ที่ระบบแสดงบนหน้าจอ)

# บันทึกสถานะโปรเซสปัจจุบัน
pm2 save
```

### 2. ตั้งค่า Autorun สำหรับกล้อง Python GUI (รันเมื่อเข้าสู่หน้าจอ Desktop)
เนื่องจาก Python GUI เป็นโปรแกรมประเภทแสดงผลกราฟิก (X11/Wayland Desktop) การรันผ่าน `systemd` อาจจะพบปัญหาเรื่องการเรียกหน้าจอ เราจึงแนะนำให้เรียกใช้ผ่านตัวจัดการ Autostart ของหน้าจอเดสก์ท็อปโดยตรง:

1. **สร้างโฟลเดอร์ autostart (ถ้ายังไม่มี):**
   ```bash
   mkdir -p ~/.config/autostart
   ```

2. **สร้างไฟล์ไอคอนทางลัดสำหรับเปิดโปรแกรม:**
   ```bash
   nano ~/.config/autostart/face_attendance.desktop
   ```

3. **ใส่ข้อมูลการตั้งค่าต่อไปนี้ลงในไฟล์:**
   ```ini
   [Desktop Entry]
   Type=Application
   Name=Face Recognition Attendance GUI
   Exec=/bin/bash -c "cd /home/pi/face_recognition && source env/bin/activate && python main.py"
   StartupNotify=false
   Terminal=false
   ```
   *(หมายเหตุ: ให้แก้ไขพาร์ท `/home/pi/face_recognition` ให้ตรงกับพาร์ทโฟลเดอร์จริงบนบอร์ด Raspberry Pi 5 ของท่าน)*

4. **เซฟไฟล์และปิดโปรแกรม (กด `Ctrl+O` แล้ว `Enter` ตามด้วย `Ctrl+X`)*

---

## 🗃️ โครงสร้างไฟล์ข้อมูลและการสำรองระบบ (Backup Guide)
ข้อมูลทั้งหมดถูกออกแบบให้อยู่ในรูปแบบของ **ไฟล์เรียบง่าย (JSON)** เพื่อให้เกิดความคล่องตัวสูงสุดในการสำรองหรือย้ายเครื่อง:

*   📁 **/faces/**: แฟ้มข้อมูลภายนอก เก็บรูปถ่ายตัวอย่างใบหน้าแยกโฟลเดอร์ตามชื่อของนักศึกษา ใช้สำหรับเทรนข้อมูลวิเคราะห์หน้า
*   📄 **face_encodings_cache.pkl**: ไฟล์แคชดัชนีใบหน้าของ Python AI ที่ถูกคำนวณและคอมไพล์แล้ว ทำให้สแกนหน้าได้ไว
*   📄 **web_app/db.json**: ฐานข้อมูลหลักที่เก็บ รายชื่อนักศึกษา, ตารางคาบเรียนเปิดเช็คชื่อ, และประวัติการลงบันทึกเวลา
*   📄 **offline_attendance.json**: ไฟล์จัดเก็บคิวเช็คชื่อชั่วคราวขณะเครื่องสแกนอยู่ในสภาวะออฟไลน์ (เน็ตหลุด/เว็บดับ)

> [!TIP]
> **การย้ายเครื่องหรือแบ็กอัปประวัติการเช็คชื่อทั้งหมด:**
> เพียงคัดลอกโฟลเดอร์ `/faces` และไฟล์ `web_app/db.json` ไปลงทับในโปรเจกต์เครื่องใหม่ ท่านก็จะได้ข้อมูลผู้ใช้และประวัติการบันทึกกลับคืนมาครบถ้วนทันที หรือจะกดสำรองประวัติผ่านหน้าเว็บแอปแถบ "สำรองระบบ" ก็ได้เช่นกัน
