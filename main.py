
import tkinter as tk
from tkinter import messagebox, simpledialog, scrolledtext
import cv2
import face_recognition
import os
import sys
import numpy as np
import time
from PIL import Image as PILImage
import shutil
import threading
from datetime import datetime

# ตั้งค่า Encoding ภาษาไทย
sys.stdout.reconfigure(encoding='utf-8')

# ---------------------------------------------------------
# HELPER FUNCTION: แปลงภาพให้เหมาะสมกับ dlib
# ---------------------------------------------------------
def to_rgb_image(frame):
    """
    แปลง Frame จากกล้อง (BGR/BGRA) หรือไฟล์ (Gray) ให้เป็น RGB 3 Channel มาตรฐาน
    """
    if frame is None: return None
    
    # 1. แปลง Data Type
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    # 2. แปลง Channel
    if len(frame.shape) == 2:
        # Grayscale (H, W) -> RGB (H, W, 3)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif len(frame.shape) == 3:
        if frame.shape[2] == 4:
            # BGRA -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif frame.shape[2] == 3:
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
    # 3. จัดเรียง Memory
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
        
    return frame

import pickle

# ---------------------------------------------------------
# CLASS: ระบบ Logic หลัก
# ---------------------------------------------------------
class FaceSystemCore:
    def __init__(self, log_callback=None):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.faces_dir = os.path.join(self.script_dir, "faces")
        self.cache_path = os.path.join(self.script_dir, "face_encodings_cache.pkl")
        self.known_face_encodings = []
        self.known_face_names = []
        self.log_callback = log_callback
        
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)

    def log(self, message):
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}

    def save_cache(self, cache_data):
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            self.log(f"⚠️ บันทึก Cache ไม่สำเร็จ: {e}")

    def reload_known_faces(self):
        """
        BIG O OPTIMIZATION: 
        1. ใช้ Caching เพื่อไม่ต้องประมวลผลรูปเดิมซ้ำ (O(New) แทน O(Total))
        2. ใช้ Mean Encoding เพื่อลด Search Space ใน Loop กล้อง (O(N) แทน O(N*M))
        """
        self.log("🔄 กำลังอัปเดตฐานข้อมูล...")
        cache = self.load_cache()
        updated_cache = {}
        
        people_folders = [f for f in os.listdir(self.faces_dir) if os.path.isdir(os.path.join(self.faces_dir, f))]
        
        temp_encodings = []
        temp_names = []

        for person_name in people_folders:
            person_folder = os.path.join(self.faces_dir, person_name)
            images = [f for f in os.listdir(person_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            
            if not images: continue

            # ตรวจสอบว่าคนนี้มีใน Cache และจำนวนรูปเท่าเดิมไหม (เพื่อข้ามการประมวลผล)
            current_state = (person_name, len(images))
            if person_name in cache and cache[person_name]['count'] == len(images):
                avg_encoding = cache[person_name]['encoding']
                updated_cache[person_name] = cache[person_name]
                self.log(f"📦 {person_name}: ใช้ข้อมูลจาก Cache")
            else:
                # ประมวลผลใหม่เฉพาะคนที่มีการเปลี่ยนแปลง
                person_encodings = []
                for filename in images:
                    image_path = os.path.join(person_folder, filename)
                    try:
                        img_bgr = cv2.imread(image_path)
                        img_rgb = to_rgb_image(img_bgr)
                        encs = face_recognition.face_encodings(img_rgb)
                        if len(encs) > 0:
                            person_encodings.append(encs[0])
                    except: continue
                
                if person_encodings:
                    # คำนวณ Mean Encoding (Centroid) เพื่อลด Big O ในการเปรียบเทียบ
                    avg_encoding = np.mean(person_encodings, axis=0)
                    updated_cache[person_name] = {'encoding': avg_encoding, 'count': len(images)}
                    self.log(f"⚡ {person_name}: ประมวลผลใหม่ ({len(person_encodings)} ภาพ)")
                else:
                    continue

            temp_encodings.append(avg_encoding)
            temp_names.append(person_name)

        self.known_face_encodings = temp_encodings
        self.known_face_names = temp_names
        self.save_cache(updated_cache)
        self.log(f"📊 ระบบพร้อม: {len(self.known_face_names)} คน (เปรียบเทียบ 1:1 ต่อคน)")


    def get_users(self):
        users = {}
        if not os.path.exists(self.faces_dir): return users
        for name in os.listdir(self.faces_dir):
            path = os.path.join(self.faces_dir, name)
            if os.path.isdir(path):
                count = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))])
                users[name] = count
        return users

    def delete_user(self, name):
        path = os.path.join(self.faces_dir, name)
        if os.path.exists(path):
            shutil.rmtree(path)
            self.log(f"🗑️ ลบ {name}")
            self.reload_known_faces()
            return True
        return False

import requests

# --- ตั้งค่า Google Apps Script Web App URL ---
# นำ URL ที่ได้จากขั้นตอน Deploy ใน Google Sheets มาวางตรงนี้
ATTENDANCE_URL = "https://script.google.com/macros/s/AKfycbxmq4TG_A--c0mo7jj0_g96VUxAjOlsXP74SbcsLchJR5UdcJ_DOhzug291n0LVMbM8KA/exec"

# ---------------------------------------------------------
# CLASS: UI
# ---------------------------------------------------------
class FaceRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🤖 Face Recognition Attendance System")
        self.geometry("600x650")
        self.configure(bg="#2c3e50")
        
        # ระบบป้องกันการเช็คซ้ำ (เก็บชื่อและเวลาที่เช็คล่าสุดใน Session นี้)
        self.recorded_today = {}
        
        # UI Setup
        tk.Label(self, text="ระบบเช็คชื่อด้วยใบหน้า", font=("TH Sarabun New", 24, "bold"), fg="white", bg="#2c3e50").pack(pady=20)
        self.status_label = tk.Label(self, text="Status: Loading...", font=("TH Sarabun New", 14), fg="#bdc3c7", bg="#2c3e50")
        self.status_label.pack(pady=5)
        
        btn_frame = tk.Frame(self, bg="#2c3e50")
        btn_frame.pack(pady=20)
        btn_style = {"font": ("TH Sarabun New", 16, "bold"), "width": 25, "height": 2, "bd": 0, "cursor": "hand2"}
        
        self.btn_start = tk.Button(btn_frame, text="📷 เริ่มระบบจดจำ & เช็คชื่อ", bg="#27ae60", fg="white", command=self.start_recognition_thread, **btn_style)
        self.btn_start.pack(pady=5)
        
        self.btn_register = tk.Button(btn_frame, text="➕ ลงทะเบียนนักศึกษาใหม่", bg="#2980b9", fg="white", command=self.open_register_window, **btn_style)
        self.btn_register.pack(pady=5)
        
        self.btn_manage = tk.Button(btn_frame, text="📋 จัดการรายชื่อนักศึกษา", bg="#8e44ad", fg="white", command=self.open_manager, **btn_style)
        self.btn_manage.pack(pady=5)
        
        self.btn_exit = tk.Button(btn_frame, text="🚪 ออกจากโปรแกรม", bg="#c0392b", fg="white", command=self.quit_app, **btn_style)
        self.btn_exit.pack(pady=5)

        tk.Label(self, text="บันทึกกิจกรรมระบบ (System Log)", font=("TH Sarabun New", 12), fg="#bdc3c7", bg="#2c3e50").pack(anchor="w", padx=20)
        self.log_area = scrolledtext.ScrolledText(self, height=8, font=("Consolas", 9), bg="#ecf0f1", state="disabled")
        self.log_area.pack(fill="x", padx=20, pady=5)

        # Core Logic
        self.core = FaceSystemCore(log_callback=self.update_log)
        self.core.reload_known_faces()
        self.refresh_status()
        self.is_running = False

    def update_log(self, message):
        def _update():
            if hasattr(self, 'log_area') and self.log_area.winfo_exists():
                self.log_area.config(state="normal")
                self.log_area.insert(tk.END, f"{message}\n")
                self.log_area.see(tk.END)
                self.log_area.config(state="disabled")
        self.after(0, _update)

    def refresh_status(self):
        def _update():
            users = self.core.get_users()
            self.status_label.config(text=f"👥 นักศึกษาในระบบ: {len(users)} คน | Encoding: {len(self.core.known_face_names)}")
        self.after(0, _update)

    def disable_buttons(self):
        def _update():
            for btn in [self.btn_start, self.btn_register, self.btn_manage]: btn.config(state="disabled")
        self.after(0, _update)

    def enable_buttons(self):
        def _update():
            for btn in [self.btn_start, self.btn_register, self.btn_manage]: btn.config(state="normal")
        self.after(0, _update)

    def open_manager(self):
        ManagerWindow(self, self.core)

    def start_recognition_thread(self):
        if self.is_running: return
        self.is_running = True
        self.disable_buttons()
        if not self.core.known_face_encodings:
            self.update_log("⚠️ เริ่มโหมดบุคคลนิรนาม (ยังไม่มีข้อมูลในระบบ)")
        self.update_log("🚀 เริ่มกล้องเช็คชื่อ... (กด Q เพื่อหยุด)")
        threading.Thread(target=self.run_recognition, daemon=True).start()

    def send_attendance(self, raw_name):
        """
        ส่งข้อมูลไปที่ Google Sheets (พร้อมระบบป้องกันการเช็คซ้ำใน 1 ชม.)
        """
        if raw_name == "Unknown":
            return

        # ตรวจสอบเวลาเช็คล่าสุด (1 ชม. = 3600 วินาที)
        now = time.time()
        last_time = self.recorded_today.get(raw_name, 0)
        if now - last_time < 3600:
            return

        # บันทึกเวลาที่เช็ค (ใส่ไว้ก่อนเริ่ม Task เพื่อกัน Duplicate Request)
        self.recorded_today[raw_name] = now

        def _task():
            # ล้างช่องว่างที่อาจติดมาใน URL
            clean_url = ATTENDANCE_URL.strip()
            
            # แยก รหัส และ ชื่อ จากชื่อโฟลเดอร์ (รูปแบบ: B6615406-Thh krooo)
            parts = raw_name.split('-')
            student_id = parts[0] if len(parts) > 1 else "N/A"
            full_name = parts[1] if len(parts) > 1 else raw_name
            
            # เตรียมพารามิเตอร์ (ต้องเป็นตัวพิมพ์เล็ก id, name, status ตามที่เขียนใน GAS)
            payload = {
                "id": str(student_id),
                "name": str(full_name),
                "status": "มาเรียน"
            }
            
            try:
                # ใช้ requests.get เพื่อส่งข้อมูลแบบ URL Parameter
                # allow_redirects=True สำคัญมากสำหรับ Google Apps Script
                response = requests.get(clean_url, params=payload, timeout=15, allow_redirects=True)
                
                if response.status_code == 200:
                    # นำข้อความตอบกลับจาก Google มาแสดง (เช่น Success - Data Added...)
                    msg = response.text.strip()
                    self.update_log(f"✅ {msg}")
                else:
                    self.update_log(f"❌ Google Error: {response.status_code}")
                    self.recorded_today.pop(raw_name, None) # หากพลาด ให้ลองใหม่ได้
            except Exception as e:
                # แสดง Error สั้นๆ เพื่อไม่ให้รก Log
                err_str = str(e).split(')')[-1] if ')' in str(e) else str(e)
                self.update_log(f"❌ เชื่อมต่อ Google ไม่ได้: {err_str[:40]}")
                self.recorded_today.pop(raw_name, None)

        threading.Thread(target=_task, daemon=True).start()

    def run_recognition(self):
        time.sleep(0.5)
        video_capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
        process_this_frame = True
        face_locations = []
        face_names = []
        
        while self.is_running:
            ret, frame = video_capture.read()
            if not ret: break

            try:
                rgb_frame = to_rgb_image(frame)
                if rgb_frame is None: continue
                
                small_rgb = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
                small_rgb = np.ascontiguousarray(small_rgb)
            except Exception:
                continue

            if process_this_frame:
                try:
                    face_locations = face_recognition.face_locations(small_rgb)
                    face_encodings = face_recognition.face_encodings(small_rgb, face_locations)
                    
                    face_names = []
                    for face_encoding in face_encodings:
                        name = "Unknown"
                        encs = self.core.known_face_encodings
                        names = self.core.known_face_names
                        if len(encs) > 0:
                            distances = face_recognition.face_distance(encs, face_encoding)
                            if len(distances) > 0:
                                best_idx = np.argmin(distances)
                                if distances[best_idx] < 0.40:
                                    name = names[best_idx]
                        face_names.append(name)
                        
                        # --- ระบบเช็คชื่อ ---
                        if name != "Unknown":
                            self.send_attendance(name)
                            
                except Exception:
                    face_locations = []
                    face_names = []

            process_this_frame = not process_this_frame

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4; right *= 4; bottom *= 4; left *= 4
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            # --- ส่วนแสดงผลด้านล่าง (Bottom Bar) ---
            h, w = frame.shape[:2]
            bar_height = 80
            # สร้างพื้นที่สีดำด้านล่าง
            bottom_bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
            bottom_bar[:] = (44, 62, 80) # สีน้ำเงินเข้ม (Matching UI)

            # ข้อมูล วัน เวลา
            now = datetime.now()
            date_str = now.strftime("%d/%m/%Y")
            time_str = now.strftime("%H:%M:%S")
            
            # รายชื่อที่ตรวจพบ
            detected_names = [n for n in face_names if n != "Unknown"]
            name_to_show = detected_names[0] if detected_names else "Scanning..."
            if len(detected_names) > 1:
                name_to_show = f"{detected_names[0]} (+{len(detected_names)-1})"

            # วาดข้อความลงบน Bottom Bar
            cv2.putText(bottom_bar, f"DATE: {date_str}", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(bottom_bar, f"TIME: {time_str}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            
            # คำนวณตำแหน่งขวาสุดสำหรับ NAME
            full_name_text = f"NAME: {name_to_show}"
            (tw, th), _ = cv2.getTextSize(full_name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            name_x = w - tw - 20 # ลบความกว้างตัวอักษรและระยะขอบ 20px
            cv2.putText(bottom_bar, full_name_text, (name_x, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

            # รวม Frame กล้องกับ Bottom Bar
            display_frame = np.vstack((frame, bottom_bar))

            cv2.imshow('Attendance System (Press Q to Stop)', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
        time.sleep(1)
        self.is_running = False
        self.enable_buttons()
        self.update_log("🛑 หยุดกล้อง")


    def open_register_window(self):
        name = simpledialog.askstring("ลงทะเบียน", "รหัสนักษาศึก-ชื่อ นามสกุล:", parent=self)
        if not name: return
        self.disable_buttons()
        self.update_log(f"📝 เริ่มลงทะเบียน: {name}")
        threading.Thread(target=self.register_process, args=(name,), daemon=True).start()

    def register_process(self, name):
        # --- ตรวจสอบพื้นที่ดิสก์ก่อนเริ่ม ---
        total, used, free = shutil.disk_usage(self.core.faces_dir)
        if free < 50 * 1024 * 1024: # ถ้าน้อยกว่า 50MB
            messagebox.showerror("Error", "พื้นที่ดิสก์ไม่เพียงพอ กรุณาลบไฟล์ที่ไม่จำเป็นออกก่อน")
            self.update_log("❌ ยกเลิก: พื้นที่ดิสก์เต็ม")
            self.enable_buttons()
            return

        save_path = os.path.join(self.core.faces_dir, name)
        os.makedirs(save_path, exist_ok=True)
        
        video_capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
        time.sleep(2.0)
        
        if not video_capture.isOpened():
            self.update_log("❌ เปิดกล้องไม่สำเร็จ")
            self.enable_buttons()
            return

        count = 0
        total_pics = 10
        btn_coords = [0,0,0,0]
        self.capture_clicked = False

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if btn_coords[0] <= x <= btn_coords[2] and btn_coords[1] <= y <= btn_coords[3]:
                    self.capture_clicked = True

        cv2.namedWindow('Register')
        cv2.setMouseCallback('Register', mouse_cb)

        while True:
            ret, frame = video_capture.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1)
            
            # Detection (ต้องใช้ RGB สำหรับ Detection)
            rgb_frame = to_rgb_image(frame)
            
            h, w = frame.shape[:2]
            bx1 = int((w-200)/2); by1 = h-80; bx2 = bx1+200; by2 = by1+60
            btn_coords = [bx1, by1, bx2, by2]
            
            face_detected = False
            if rgb_frame is not None:
                try:
                    if len(face_recognition.face_locations(rgb_frame)) > 0:
                        face_detected = True
                except: pass
            
            color = (0, 200, 0) if face_detected else (0, 0, 200)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
            cv2.putText(frame, "CAPTURE", (bx1+40, by1+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(frame, f"Count: {count}/{total_pics}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            cv2.imshow('Register', frame)
            key = cv2.waitKey(1) & 0xFF

            if self.capture_clicked or key == ord(' '):
                if count < total_pics:
                    try:
                        # 1. แปลงเป็น Gray
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                        # 2. บันทึกไฟล์ .jpg
                        file_name = f"{name}_{int(time.time())}.jpg"
                        file_path = os.path.join(save_path, file_name)
                        
                        # ตรวจสอบพื้นที่ก่อนบันทึกแต่ละรูป
                        _, _, free_now = shutil.disk_usage(self.core.faces_dir)
                        if free_now < 1 * 1024 * 1024: # ถ้าน้อยกว่า 1MB
                             raise OSError("Disk Full")

                        success = cv2.imwrite(file_path, gray_frame)
                        
                        if success:
                            count += 1
                            self.update_log(f"📸 ถ่ายภาพ {count}/{total_pics}")
                            flash = np.full(frame.shape, 255, dtype=np.uint8)
                            cv2.imshow('Register', flash); cv2.waitKey(100)
                        else:
                            raise Exception("cv2.imwrite failed")
                            
                    except (OSError, Exception) as e:
                        msg = "พื้นที่ดิสก์เต็ม" if "Disk" in str(e) else f"Error: {e}"
                        self.update_log(f"❌ {msg}")
                        messagebox.showerror("Fatal Error", f"ไม่สามารถบันทึกรูปได้: {msg}")
                        break # หยุดการลงทะเบียนทันที
                        
                self.capture_clicked = False

            if key == ord('q') or count >= total_pics:
                break

        video_capture.release()
        cv2.destroyAllWindows()
        time.sleep(1)
        
        
        if count > 0:
            self.update_log("✅ เสร็จสิ้น กำลังโหลดข้อมูลใหม่...")
            self.core.reload_known_faces()
            self.refresh_status()
        else:
            self.update_log("⚠️ ยกเลิกการลงทะเบียน")
        
        self.enable_buttons()


    def quit_app(self):
        if messagebox.askyesno("ออก", "ปิดโปรแกรม?"): 
            self.is_running = False
            self.destroy()

class ManagerWindow(tk.Toplevel):
    def __init__(self, parent, core):
        super().__init__(parent)
        self.core = core
        self.title("จัดการผู้ใช้")
        self.geometry("400x500")
        tk.Label(self, text="รายชื่อ", font=("TH Sarabun New", 16)).pack(pady=10)
        
        self.lb = tk.Listbox(self, font=("TH Sarabun New", 14))
        self.lb.pack(fill="both", expand=True, padx=10)
        
        tk.Button(self, text="ลบ", font=("TH Sarabun New", 12), bg="red", fg="white", command=self.delete).pack(pady=5)
        self.refresh()

    def refresh(self):
        self.lb.delete(0, tk.END)
        for n, c in self.core.get_users().items():
            self.lb.insert(tk.END, f"{n} ({c} ภาพ)")
        if not self.core.get_users(): self.lb.insert(tk.END, "ไม่มีข้อมูล")

    def delete(self):
        sel = self.lb.curselection()
        if not sel: return
        txt = self.lb.get(sel[0])
        if "ไม่มี" in txt: return
        name = txt.split(" (")[0]
        if messagebox.askyesno("ยืนยัน", f"ลบ {name}?"):
            self.core.delete_user(name)
            self.refresh()
            self.master.refresh_status()

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.mainloop()
