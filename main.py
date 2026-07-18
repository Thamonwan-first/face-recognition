import tkinter as tk 
from tkinter import messagebox, simpledialog, scrolledtext 
import cv2 
import face_recognition 
import os 
import numpy as np 
import time 
import multiprocessing as mp 
import requests 
import threading 
import pickle
import json
from datetime import datetime 
from PIL import Image as PILImage, ImageTk, ImageDraw, ImageFont 
from http.server import HTTPServer, BaseHTTPRequestHandler

# --- CONFIGURATION --- 
WEB_APP_URL = "https://pinkseals.xyz/api/attendance/checkin"

# --------------------------------------------------------- 
# GLOBAL APP INSTANCE & CONTROL HTTP SERVER FOR WEB APP
# --------------------------------------------------------- 
app_instance = None 

class PythonAPIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress logging HTTP requests to stderr
        return
        
    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b'{"status": "running"}')
        else:
            self.send_response(404)
            self.end_headers()
            
    def do_POST(self):
        if self.path == '/reload':
            global app_instance
            if app_instance:
                app_instance.add_log("🔄 Web App แจ้งเตือน: รีโหลดและประมวลผลฐานข้อมูลใบหน้าใหม่...")
                app_instance.manual_train()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b'{"status": "reloading"}')
        else:
            self.send_response(404)
            self.end_headers()

# --------------------------------------------------------- 
# FUNCTION: AI CORE PROCESS
# --------------------------------------------------------- 
def load_encodings(cache_path):
    known_encs = []
    known_names = []
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
                for name, data in cache.items():
                    known_encs.append(data['encoding'])
                    known_names.append(name)
        except: pass
    return np.array(known_encs), known_names

def ai_worker(frame_q, result_q, ctrl_ev, reload_ev, faces_dir, cache_path, rotation_val): 
    known_encs, known_names = load_encodings(cache_path)
     
    cap = cv2.VideoCapture(1) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    latest_frame = [None]
    frame_lock = threading.Lock()

    # Thread สำหรับดึงเฟรมจากกล้องตลอดเวลา (Camera Grabber Thread) เพื่อความลื่นไหลของภาพแสดงผล
    def grab_frames():
        while ctrl_ev.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # หมุนภาพตามที่ตั้งไว้ใน UI
            rot = rotation_val.value
            if rot == 90: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) 
            elif rot == 180: frame = cv2.rotate(frame, cv2.ROTATE_180) 
            elif rot == 270: frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) 

            with frame_lock:
                latest_frame[0] = frame.copy()

            # ส่งเฟรมไปให้ GUI แสดงผลแบบ Real-time (ล้างคิวเก่าออกหากคิวเต็มเพื่อให้แสดงผลภาพล่าสุดเสมอ)
            if frame_q.full():
                try:
                    frame_q.get_nowait()
                except:
                    pass
            try:
                frame_q.put_nowait(frame)
            except:
                pass

            # พักเล็กน้อยเพื่อให้กล้องส่งเฟรมประมาณ 30 FPS และไม่แย่ง CPU เกินไป
            time.sleep(0.03)

    grab_thread = threading.Thread(target=grab_frames, daemon=True)
    grab_thread.start()

    # Main thread ของ ai_worker ทำหน้าที่ตรวจจับใบหน้าอย่างเดียว (AI Face Recognition Loop)
    while ctrl_ev.is_set(): 
        # เช็คสัญญาณรีโหลดข้อมูลใบหน้าใหม่
        if reload_ev.is_set():
            known_encs, known_names = load_encodings(cache_path)
            reload_ev.clear()

        frame_to_process = None
        with frame_lock:
            if latest_frame[0] is not None:
                frame_to_process = latest_frame[0].copy()
                latest_frame[0] = None # เคลียร์หลังดึงไปประมวลผลแล้วเพื่อไม่ให้ทำซ้ำเฟรมเดิม

        if frame_to_process is None:
            time.sleep(0.01)
            continue

        small_frame = cv2.resize(frame_to_process, (0, 0), fx=0.25, fy=0.25) 
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB) 
         
        locs = face_recognition.face_locations(rgb_small, model="hog") 
        encs = face_recognition.face_encodings(rgb_small, locs) 
         
        names = [] 
        for e in encs: 
            name = "Unknown" 
            if len(known_encs) > 0: 
                dist = np.linalg.norm(known_encs - e, axis=1) 
                if np.min(dist) < 0.40: 
                    name = known_names[np.argmin(dist)] 
            names.append(name) 

        # ส่งผลลัพธ์ไปที่ GUI (ล้างคิวเก่าออกหากคิวเต็ม)
        if result_q.full():
            try:
                result_q.get_nowait()
            except:
                pass
        try:
            result_q.put_nowait((locs, names))
        except:
            pass

    cap.release() 

# --------------------------------------------------------- 
# CLASS: DASHBOARD (Optimized for Portrait Screen)
# --------------------------------------------------------- 
class Pi5PortraitDash(tk.Tk): 
    def __init__(self): 
        super().__init__() 
        self.title("PI 5 PORTRAIT DASHBOARD") 
        
        # ตั้งค่าให้เปิดเต็มหน้าจอ (Fullscreen) - เพิ่ม self.update() เพื่อรองรับ Wayland บน Pi 5
        self.update()
        self.attributes('-fullscreen', True)
        # กดปุ่ม Esc เพื่อออกจากโหมดเต็มจอ
        self.bind("<Escape>", lambda event: self.attributes("-fullscreen", False))
        
        self.configure(bg="#050505") 

        self.frame_q = mp.Queue(maxsize=2) 
        self.result_q = mp.Queue(maxsize=2) 
        self.ctrl_ev = mp.Event() 
        self.reload_ev = mp.Event() # สำหรับแจ้ง AI ให้โหลดไฟล์ cache ใหม่
        self.rotation_val = mp.Value('i', 0) 
        self.proc = None 
         
        self.last_locs = [] 
        self.last_names = [] 
        self.recorded = {} 
        self.current_frame = None 
        self.capture_count = 0
        self.is_training = False # สถานะว่ากำลังประมวลผลอยู่หรือไม่

        # ตัวแปรควบคุมการแสดงสถานะเช็คชื่อบนหน้ากล้อง
        self.status_display_text = ""
        self.status_display_color = (255, 255, 255)
        self.status_display_expiry = 0

        # ตัวแปรระบบล็อก Thread สำหรับไฟล์สำรองออฟไลน์เพื่อป้องกัน Race Condition
        self.offline_lock = threading.Lock()

        # ตัวแปรควบคุมการล็อกเช็คชื่อในคอมพิวเตอร์โลคอล
        self.current_session_id = None
        self.logged_checkins = set()

        # ตัวแปรสำหรับเก็บข้อมูลสถานะและประวัติคาบล่าสุดเมื่อออฟไลน์
        self.last_active_session = None
        self.last_attendance_records = []
        self.is_server_offline = False

        self.init_core_paths() 
        self.init_ui() 
        
        # ตั้งค่า instance และสตาร์ทเซิร์ฟเวอร์ควบคุมเบื้องหลัง (Port 5001)
        global app_instance
        app_instance = self
        threading.Thread(target=self.start_local_server, daemon=True).start()

        # สตาร์ท Thread ตรวจสอบและอัปโหลดประวัติออฟไลน์ย้อนหลัง
        threading.Thread(target=self.sync_offline_loop, daemon=True).start()

        threading.Thread(target=self.train_ai, daemon=True).start()
        
        # เริ่มต้นดึงข้อมูลวิชาที่เช็คชื่อและคิวประวัติการเช็คชื่อแบบเรียลไทม์
        self.fetch_active_session_data()
        self.update_clock()
        self.main_loop() 

    def init_core_paths(self): 
        self.script_dir = os.path.dirname(os.path.abspath(__file__)) 
        self.faces_dir = os.path.join(self.script_dir, "faces") 
        self.cache_path = os.path.join(self.script_dir, "face_encodings_cache.pkl") 
        if not os.path.exists(self.faces_dir): os.makedirs(self.faces_dir) 

 

    def init_ui(self): 
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1) # ส่วนแสดงรายวิชา (บนสุด)
        self.rowconfigure(1, weight=25) # ส่วนกล้อง (ให้พื้นที่กล้องขนาดใหญ่ขึ้น)
        self.rowconfigure(2, weight=3) # ส่วน Log (ลดขนาดความสูงลง)
        self.rowconfigure(3, weight=2) # ส่วนควบคุม (ลดขนาดความสูงลง)

        # 0. Active Session Info Banner (Top)
        self.top_banner = tk.Frame(self, bg="#1a1a1a")
        self.top_banner.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 0))
        self.lbl_session_info = tk.Label(self.top_banner, text="วิชา: ไม่มีวิชาเรียนที่กำลังเปิดเช็คชื่อ", bg="#1a1a1a", fg="#ffcc00", font=("Arial", 16, "bold"), anchor="w")
        self.lbl_session_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15, pady=10)
        self.lbl_clock = tk.Label(self.top_banner, text="00:00:00", bg="#1a1a1a", fg="#ffffff", font=("Consolas", 18, "bold"))
        self.lbl_clock.pack(side=tk.RIGHT, padx=15, pady=10)

        # 1. Video Frame (Middle)
        self.v_frame = tk.Frame(self, bg="#000") 
        self.v_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.v_label = tk.Label(self.v_frame, bg="#000") 
        self.v_label.pack(fill=tk.BOTH, expand=True) 

        # 2. Log and Attendance Container (Under Camera)
        self.log_container = tk.Frame(self, bg="#050505")
        self.log_container.grid(row=2, column=0, sticky="nsew", padx=15, pady=5)
        self.log_container.columnconfigure(0, weight=1) # 100% width for attendance list
        self.log_container.rowconfigure(0, weight=1)

        # 2a. Attendance List Card (Full Width)
        self.attendance_frame = tk.Frame(self.log_container, bg="#161b22", highlightthickness=1, highlightbackground="#30363d")
        self.attendance_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=5)
        
        # Header sub-frame to keep title and status dot on the same top line
        self.attendance_header = tk.Frame(self.attendance_frame, bg="#161b22")
        self.attendance_header.pack(fill=tk.X, padx=10, pady=(5, 2))
        
        self.lbl_attendance_title = tk.Label(
            self.attendance_header, 
            text="📋 รายชื่อที่เช็คชื่อเข้าเรียนในวิชานี้ (0 คน)", 
            font=("Segoe UI", 10, "bold"), 
            fg="#ffcc00", 
            bg="#161b22"
        )
        self.lbl_attendance_title.pack(side=tk.LEFT, anchor="w")
        
        self.lbl_status_dot = tk.Label(
            self.attendance_header, 
            text="● Online", 
            bg="#161b22", 
            fg="#00ff00", 
            font=("Segoe UI", 10, "bold")
        )
        self.lbl_status_dot.pack(side=tk.RIGHT, anchor="e")
        
        self.attendance_list = scrolledtext.ScrolledText(
            self.attendance_frame, 
            bg="#0d1117", 
            fg="#cbd5e1", 
            font=("Segoe UI", 10), 
            bd=0, 
            height=8,
            wrap=tk.NONE,
            tabs=('50', '200', '420', '540')
        )
        self.attendance_list.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.attendance_list.tag_config("header", foreground="#ffcc00", font=("Segoe UI", 10, "bold"))
        self.attendance_list.tag_config("ontime", foreground="#00ff00")
        self.attendance_list.tag_config("late", foreground="#ffa500")
        self.attendance_list.tag_config("offline", foreground="#00ffff")
        self.attendance_list.config(state=tk.DISABLED)

        # 3. Control Panel (Bottom)
        self.p_frame = tk.Frame(self, bg="#0b0f19") 
        self.p_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0, 10))

        self.p_frame.columnconfigure(0, weight=1)
        self.p_frame.columnconfigure(1, weight=1)

        # --- ส่วนซ้าย: ระบบหลัก (System Status Card) ---
        left_ctrl = tk.Frame(self.p_frame, bg="#161b22", highlightthickness=1, highlightbackground="#30363d")
        left_ctrl.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        left_ctrl.columnconfigure(0, weight=1)

        tk.Label(left_ctrl, text="🖥️ SYSTEM STATUS", font=("Segoe UI", 10, "bold"), fg="#58a6ff", bg="#161b22").pack(pady=(12, 8))
        
        self.btn_run = tk.Button(left_ctrl, text="STOP SYSTEM", bg="#ff4d4d", fg="white", font=("Segoe UI", 9, "bold"), 
                                command=self.toggle_engine, bd=0, height=2, activebackground="#ff3333", activeforeground="white", cursor="hand2") 
        self.btn_run.pack(fill=tk.X, padx=15, pady=4) 

        btn_rotate = tk.Button(left_ctrl, text="🔄 ROTATE CAMERA", bg="#30363d", fg="#cbd5e1", font=("Segoe UI", 9, "bold"),
                               command=self.rotate, bd=0, height=2, activebackground="#21262d", activeforeground="white", cursor="hand2")
        btn_rotate.pack(fill=tk.X, padx=15, pady=(4, 15))

        # --- ส่วนขวา: ลงทะเบียน (Registration Card) ---
        right_ctrl = tk.Frame(self.p_frame, bg="#161b22", highlightthickness=1, highlightbackground="#30363d")
        right_ctrl.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        right_ctrl.columnconfigure(0, weight=1)

        tk.Label(right_ctrl, text="👤 REGISTRATION", font=("Segoe UI", 10, "bold"), fg="#ffcc00", bg="#161b22").pack(pady=(12, 4))
        tk.Label(right_ctrl, text="ตัวอย่าง: รหัสนักศึกษา-ชื่อสกุล", font=("Arial", 8), fg="#8b949e", bg="#161b22").pack(pady=(0, 4))
        
        self.ent_name = tk.Entry(right_ctrl, bg="#0d1117", fg="white", bd=0, font=("Arial", 10),
                                 highlightthickness=1, highlightbackground="#30363d", highlightcolor="#58a6ff", insertbackground="white")
        self.ent_name.pack(fill=tk.X, padx=15, pady=4, ipady=3)
        
        self.btn_capture = tk.Button(right_ctrl, text="📸 CAPTURE FACE (0/10)", bg="#1f6feb", fg="white", font=("Segoe UI", 9, "bold"), 
                                command=self.capture_photo, bd=0, height=2, activebackground="#1050b3", activeforeground="white", cursor="hand2") 
        self.btn_capture.pack(fill=tk.X, padx=15, pady=(4, 15)) 

    def fetch_active_session_data(self):
        def _poll():
            base_url = WEB_APP_URL.rsplit('/', 2)[0]
            db_url = f"{base_url}/db"
            
            while True:
                try:
                    res = requests.get(db_url, timeout=3)
                    if res.status_code == 200:
                        data = res.json()
                        sessions = data.get("sessions", [])
                        attendance = data.get("attendance", [])
                        
                        active_sess = None
                        for s in sessions:
                            if s.get("active"):
                                active_sess = s
                                break
                        
                        self.after(0, self.update_session_ui, active_sess, attendance, False)
                except Exception as e:
                    self.after(0, self.update_session_ui, None, [], True)
                
                time.sleep(5)
                
        threading.Thread(target=_poll, daemon=True).start()

    def update_banner(self, text, is_offline):
        bg_color = "#331a00" if is_offline else "#1a1a1a"
        fg_color = "#ff9900" if is_offline else "#ffcc00"
        
        self.top_banner.config(bg=bg_color)
        self.lbl_session_info.config(bg=bg_color, fg=fg_color, text=text)
        self.lbl_clock.config(bg=bg_color)
        
        if hasattr(self, 'lbl_status_dot'):
            status_text = "● Offline" if is_offline else "● Online"
            status_color = "#ff3333" if is_offline else "#00ff00"
            self.lbl_status_dot.config(fg=status_color, text=status_text)

    def get_offline_records(self):
        offline_file = os.path.join(self.script_dir, "offline_attendance.json")
        records = []
        if os.path.exists(offline_file):
            with self.offline_lock:
                try:
                    with open(offline_file, 'r', encoding='utf-8') as f:
                        records = json.load(f)
                except:
                    pass
        return records

    def trigger_local_refresh(self):
        active_sess = getattr(self, 'last_active_session', None)
        attendance = getattr(self, 'last_attendance_records', [])
        is_offline = getattr(self, 'is_server_offline', False)
        self.update_session_ui(active_sess, attendance, is_offline)

    def update_session_ui(self, active_sess, attendance, is_server_offline=False):
        if is_server_offline:
            # Server is offline. Retain last known session and attendance
            active_sess = getattr(self, 'last_active_session', None)
            attendance = getattr(self, 'last_attendance_records', [])
            self.is_server_offline = True
            
            if active_sess:
                code = active_sess.get("subjectCode", "-")
                name = active_sess.get("subjectName", "-")
                start = active_sess.get("startTime", "-")
                end = active_sess.get("endTime", "-")
                banner_text = f"⚠️ [ออฟไลน์] วิชาเรียน: {code} - {name}   |   เวลาเรียน: {start} - {end} น."
            else:
                banner_text = "⚠️ [ออฟไลน์] ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์ และไม่มีประวัติคาบเรียนเรียนแคชไว้"
                
            self.update_banner(banner_text, is_offline=True)
        else:
            # Server is online
            self.last_active_session = active_sess
            self.last_attendance_records = attendance
            self.is_server_offline = False
            
            if active_sess:
                code = active_sess.get("subjectCode", "-")
                name = active_sess.get("subjectName", "-")
                start = active_sess.get("startTime", "-")
                end = active_sess.get("endTime", "-")
                date_str = active_sess.get("date", "")
                
                # Get current local date and time
                now = datetime.now()
                today_str = now.strftime("%Y-%m-%d")
                current_time_str = now.strftime("%H:%M")
                
                if date_str != today_str or current_time_str < start or current_time_str > end:
                    self.update_banner("วิชา: ไม่มีวิชาเรียนที่กำลังเปิดเช็คชื่อ", is_offline=False)
                    if self.current_session_id is not None:
                        self.current_session_id = None
                        self.logged_checkins.clear()
                        self.add_log("📝 คาบเรียนปัจจุบันถูกปิดลงแล้ว")
                else:
                    banner_text = f"วิชาเรียน: {code} - {name}   |   เวลาเรียน: {start} - {end} น."
                    
                    # Check for session change
                    sess_id = active_sess.get("id")
                    if self.current_session_id != sess_id:
                        self.current_session_id = sess_id
                        self.logged_checkins.clear()
                        self.add_log(f"📝 เริ่มต้นสแกนเช็คชื่อคาบใหม่: {code} - {name}")
                    
                    self.update_banner(banner_text, is_offline=False)
            else:
                self.update_banner("วิชา: ไม่มีวิชาเรียนที่กำลังเปิดเช็คชื่อ", is_offline=False)
                if self.current_session_id is not None:
                    self.current_session_id = None
                    self.logged_checkins.clear()
                    self.add_log("📝 คาบเรียนปัจจุบันถูกปิดลงแล้ว")

        # Now update the list box (it takes active_sess, attendance, and local offline records)
        sess_id = active_sess.get("id") if active_sess else None
        
        # Filter attendance to current session
        sess_attendance = []
        if sess_id:
            sess_attendance = [a for a in attendance if a.get("sessionId") == sess_id]
            
        offline_records = self.get_offline_records()
        self.display_attendance_list(active_sess, sess_attendance, offline_records)

    def display_attendance_list(self, active_sess, sess_attendance, offline_records):
        # Format helper (Converts UTC ISO time string to local time or fallback)
        def format_time_str(t_str):
            try:
                if 'T' in t_str:
                    if t_str.endswith('Z'):
                        # Replace Z with +00:00 for python 3.10 and older compatibility
                        dt = datetime.fromisoformat(t_str.replace('Z', '+00:00'))
                        return dt.astimezone().strftime('%H:%M:%S')
                    else:
                        dt = datetime.fromisoformat(t_str)
                        return dt.strftime('%H:%M:%S')
                return t_str[:19]
            except:
                try:
                    if 'T' in t_str:
                        return t_str.split('T')[1].split('.')[0]
                except:
                    pass
                return t_str

        # Log new check-ins to the system log (once per check-in)
        if active_sess:
            sess_attendance.sort(key=lambda x: x.get("time", ""))
            for a in sess_attendance:
                std_id = a.get("studentId", "-")
                std_name = a.get("studentName", "-")
                t_str = a.get("time", "")
                key = f"{std_id}_{t_str}"
                
                if key not in self.logged_checkins:
                    self.logged_checkins.add(key)
                    status = a.get("status", "ontime")
                    status_thai = "ตรงเวลา" if status == "ontime" else "สาย"
                    t_disp = format_time_str(t_str)
                    self.add_log(f"✅ ลำดับที่ {len(self.logged_checkins)}: {std_id} - {std_name} ({status_thai}) @ {t_disp}")
                    
        # Build merged lists for display
        online_student_ids = {a.get("studentId").strip().upper() for a in sess_attendance if a.get("studentId")}
        
        merged_attendance = []
        
        # Add online check-ins
        for a in sess_attendance:
            std_id = a.get("studentId", "-")
            std_name = a.get("studentName", "-")
            time_disp = format_time_str(a.get("time", ""))
            status = a.get("status", "ontime")
            status_thai = "ตรงเวลา" if status == "ontime" else "สาย"
            
            merged_attendance.append({
                "id": std_id,
                "name": std_name,
                "time": time_disp,
                "status": status_thai,
                "is_offline": False,
                "sort_time": a.get("time", "")
            })
            
        # Add unsynced offline check-ins
        added_offline_ids = set()
        for rec in offline_records:
            name_id = rec.get("name_id", "")
            hyphen_idx = name_id.find('-')
            if hyphen_idx != -1:
                std_id = name_id[:hyphen_idx].strip()
                std_name = name_id[hyphen_idx+1:].strip()
            else:
                std_id = name_id.strip()
                std_name = ""
                
            std_id_upper = std_id.upper()
            if std_id_upper not in online_student_ids and std_id_upper not in added_offline_ids:
                added_offline_ids.add(std_id_upper)
                time_disp = format_time_str(rec.get("time", ""))
                status_thai = "ตรงเวลา"
                if active_sess and active_sess.get("lateAfter"):
                    try:
                        rec_time_str = time_disp[:5] # HH:MM
                        if rec_time_str > active_sess.get("lateAfter"):
                            status_thai = "สาย"
                    except:
                        pass
                
                merged_attendance.append({
                    "id": std_id,
                    "name": std_name,
                    "time": time_disp,
                    "status": status_thai,
                    "is_offline": True,
                    "sort_time": rec.get("time", "")
                })
                
        # Sort chronologically
        merged_attendance.sort(key=lambda x: x["sort_time"])
        
        # Display in Text widget
        self.lbl_attendance_title.config(text=f"📋 รายชื่อที่เช็คชื่อเข้าเรียนในวิชานี้ ({len(merged_attendance)} คน)")
        
        self.attendance_list.config(state=tk.NORMAL)
        self.attendance_list.delete("1.0", tk.END)
        
        # Header
        header = "ลำดับ\tรหัสนักศึกษา\tชื่อ-นามสกุล\tเวลา\tสถานะ\n"
        separator = "─" * 50 + "\n"
        self.attendance_list.insert(tk.END, header, "header")
        self.attendance_list.insert(tk.END, separator)
        
        for idx, item in enumerate(merged_attendance, 1):
            lbl = f"{idx}."
            std_id = item["id"]
            name = item["name"]
            if len(name) > 22:
                name = name[:19] + "..."
            time_val = item["time"]
            
            self.attendance_list.insert(tk.END, f"{lbl}\t{std_id}\t{name}\t{time_val}\t")
            
            if item["is_offline"]:
                status_str = f"{item['status']} (ออฟไลน์ 💾)\n"
                self.attendance_list.insert(tk.END, status_str, "offline")
            else:
                status_str = f"{item['status']}\n"
                tag = "late" if item["status"] == "สาย" else "ontime"
                self.attendance_list.insert(tk.END, status_str, tag)
                
        self.attendance_list.see(tk.END)
        self.attendance_list.config(state=tk.DISABLED)

    def update_clock(self):
        now_str = datetime.now().strftime("%H:%M:%S")
        if hasattr(self, 'lbl_clock'):
            self.lbl_clock.config(text=now_str)
        self.after(1000, self.update_clock)

    def start_local_server(self):
        try:
            server = HTTPServer(('0.0.0.0', 5001), PythonAPIHandler)
            server.serve_forever()
        except Exception as e:
            self.add_log(f"⚠️ ไม่สามารถสตาร์ทพอร์ตควบคุม 5001 ได้: {e}")

    def rotate(self): 
        new_rot = (self.rotation_val.value + 90) % 360
        self.rotation_val.value = new_rot
        self.add_log(f"Camera rotated to {new_rot}°")

    def train_ai(self):
        if self.is_training: return
        self.is_training = True
        self.add_log("AI: Training/Updating Database...")
        
        cache = {}
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f: cache = pickle.load(f)
            except: pass
        
        updated_cache = {}
        people = [f for f in os.listdir(self.faces_dir) if os.path.isdir(os.path.join(self.faces_dir, f))]
        for name in people:
            p_path = os.path.join(self.faces_dir, name)
            imgs = [f for f in os.listdir(p_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not imgs: continue
            if name in cache and cache[name].get('count') == len(imgs):
                updated_cache[name] = cache[name]
            else:
                encs = []
                for img_name in imgs:
                    try:
                        img = face_recognition.load_image_file(os.path.join(p_path, img_name))
                        e = face_recognition.face_encodings(img)
                        if e: encs.append(e[0])
                    except: continue
                if encs: updated_cache[name] = {'encoding': np.mean(encs, axis=0), 'count': len(imgs)}
        
        with open(self.cache_path, 'wb') as f: pickle.dump(updated_cache, f)
        
        self.add_log("AI: Update Success! System is ready.")
        self.is_training = False
        
        # ถ้ากล้องรันอยู่ ให้แจ้งเตือน AI ให้รีโหลด cache ทันที
        if self.ctrl_ev.is_set():
            self.reload_ev.set()
        else:
            self.after(0, self.toggle_engine)

    def manual_train(self):
        threading.Thread(target=self.train_ai, daemon=True).start()

    def toggle_engine(self): 
        if not self.ctrl_ev.is_set(): 
            self.ctrl_ev.set() 
            self.proc = mp.Process(target=ai_worker, args=(self.frame_q, self.result_q, self.ctrl_ev, self.reload_ev, self.faces_dir, self.cache_path, self.rotation_val)) 
            self.proc.start() 
            self.btn_run.config(text="STOP SYSTEM", bg="#ff4d4d", fg="white", activebackground="#ff3333", activeforeground="white") 
        else: 
            self.ctrl_ev.clear() 
            if self.proc: self.proc.terminate() 
            self.btn_run.config(text="START SYSTEM", bg="#2ea043", fg="white", activebackground="#238636", activeforeground="white") 
            self.v_label.config(image='') 

    def capture_photo(self):
        name = self.ent_name.get().strip()
        if not name or self.current_frame is None: return
        save_path = os.path.join(self.faces_dir, name)
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(os.path.join(save_path, f"{int(time.time())}.jpg"), self.current_frame)
        self.capture_count += 1
        self.btn_capture.config(text=f"📸 CAPTURE ({self.capture_count}/10)")
        if self.capture_count >= 10:
            messagebox.showinfo("Done", f"ลงทะเบียนคุณ {name} เรียบร้อยแล้ว\nระบบจะทำการอัปเดตข้อมูลอัตโนมัติ")
            self.ent_name.delete(0, tk.END); self.capture_count = 0
            self.btn_capture.config(text="📸 CAPTURE (0/10)")
            
            # ส่งสัญญาณซิงก์นักศึกษาใหม่ไปยังเว็บแอปเพื่อเพิ่มลง db.json อัตโนมัติ
            def _sync_web():
                try:
                    base_url = WEB_APP_URL.rsplit('/', 2)[0]
                    requests.post(f"{base_url}/students/sync", timeout=4)
                except:
                    pass
            threading.Thread(target=_sync_web, daemon=True).start()
            
            self.manual_train()

    def add_log(self, m): 
        print(f"{datetime.now().strftime('%H:%M:%S')}> {m}")

    def main_loop(self): 
        is_reg = len(self.ent_name.get().strip()) > 0
        try: 
            if not self.result_q.empty():
                locs, names = self.result_q.get_nowait()
                if not is_reg:
                    self.last_locs, self.last_names = locs, names
                    for n in self.last_names: 
                        if n != "Unknown": 
                            self.cloud_sync(n) 
                else: self.last_locs, self.last_names = [], []
        except: pass 

        try: 
            if not self.frame_q.empty(): 
                raw = self.frame_q.get_nowait() 
                self.current_frame = raw.copy()
                frame = raw.copy()
                if not is_reg:
                    for (t, r, b, l), name in zip(self.last_locs, self.last_names): 
                        t, r, b, l = t*4, r*4, b*4, l*4 
                        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0) # แดงถ้าไม่รู้จัก, เขียวถ้ารู้จัก
                        cv2.rectangle(frame, (l, t), (r, b), color, 2) 
                        cv2.putText(frame, name, (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1) 
                else:
                    cv2.putText(frame, "REGISTRATION MODE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 204, 255), 2)

                img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 

                # ดึงขนาดของวิดเจ็ตเฟรมกล้องจริงแบบไดนามิก เพื่อให้ภาพขยายเต็มพื้นที่จอโดยอัตโนมัติแบบรักษาสัดส่วน (Aspect Ratio)
                w = self.v_label.winfo_width()
                h = self.v_label.winfo_height()
                if w > 10 and h > 10:
                    img_w, img_h = img.size
                    scale = min(w / img_w, h / img_h)
                    new_w = max(1, int(img_w * scale))
                    new_h = max(1, int(img_h * scale))
                    img = img.resize((new_w, new_h), PILImage.Resampling.BILINEAR)
                else:
                    img_w, img_h = img.size
                    scale = min(700 / img_w, 1022 / img_h)
                    new_w = max(1, int(img_w * scale))
                    new_h = max(1, int(img_h * scale))
                    img = img.resize((new_w, new_h), PILImage.Resampling.BILINEAR) 
                tk_img = ImageTk.PhotoImage(image=img) 
                self.v_label.imgtk = tk_img; self.v_label.config(image=tk_img) 
        except: pass 
        self.after(16, self.main_loop) 

    def save_offline_attendance(self, name):
        parts = name.split('-', 1)
        student_id_to_check = parts[0].strip().upper()

        offline_file = os.path.join(self.script_dir, "offline_attendance.json")
        with self.offline_lock:
            records = []
            if os.path.exists(offline_file):
                try:
                    with open(offline_file, 'r', encoding='utf-8') as f:
                        records = json.load(f)
                except:
                    pass
            
            # Check if this student is already in offline records
            for r in records:
                r_name = r.get("name_id", "")
                r_parts = r_name.split('-', 1)
                r_id = r_parts[0].strip().upper()
                if r_id == student_id_to_check:
                    # Already logged offline, skip adding it again
                    return
            
            # Check if this student is already in online records
            if hasattr(self, 'last_attendance_records') and self.last_attendance_records:
                for a in self.last_attendance_records:
                    a_id = a.get("studentId", "").strip().upper()
                    if a_id == student_id_to_check:
                        # Already logged online, skip adding it again
                        return
            
            now_iso = datetime.now().isoformat()
            records.append({"name_id": name, "time": now_iso})
            
            try:
                with open(offline_file, 'w', encoding='utf-8') as f:
                    json.dump(records, f, indent=2)
                self.add_log(f"💾 ออฟไลน์: บันทึกประวัติของ {name} ลงเครื่องสำเร็จ (รอเน็ตเชื่อมต่อเพื่อซิงค์)")
            except Exception as e:
                self.add_log(f"❌ เกิดข้อผิดพลาดในการบันทึกออฟไลน์ลงเครื่อง: {e}")
        self.after(0, self.trigger_local_refresh)

    def sync_offline_loop(self):
        offline_file = os.path.join(self.script_dir, "offline_attendance.json")
        temp_sync_file = os.path.join(self.script_dir, "offline_attendance_syncing.json")
        
        while True:
            time.sleep(10)
            
            # หากมีไฟล์แคชชั่วคราวค้างอยู่จากรอบที่แล้ว ให้พยายามจัดการก่อน
            if os.path.exists(temp_sync_file):
                self.merge_temp_file_back(temp_sync_file, offline_file)
                
            if not os.path.exists(offline_file):
                continue
                
            # สลับเปลี่ยนชื่อไฟล์เพื่อป้องกันการแทรกแซงเขียนไฟล์ขณะทำงาน (Thread Safe Swapping)
            with self.offline_lock:
                try:
                    os.rename(offline_file, temp_sync_file)
                except:
                    continue
                    
            # ทำงานกับไฟล์แคชที่ล็อกไว้เพื่อความปลอดภัย
            records = []
            try:
                with open(temp_sync_file, 'r', encoding='utf-8') as f:
                    records = json.load(f)
            except:
                try: os.remove(temp_sync_file)
                except: pass
                continue
                
            if not records:
                try: os.remove(temp_sync_file)
                except: pass
                continue
                
            try:
                # ส่งรายการออฟไลน์ทั้งหมดไปซิงค์กับเว็บแอป
                res = requests.post("https://pinkseals.xyz/api/attendance/sync_offline", json={"records": records}, timeout=5)
                if res.status_code == 200:
                    self.add_log(f"🔄 Sync: อัปโหลดประวัติสแกนออฟไลน์ {len(records)} รายการขึ้นเว็บสำเร็จ!")
                    try: os.remove(temp_sync_file)
                    except: pass
                    self.after(0, self.trigger_local_refresh)
                else:
                    self.merge_temp_file_back(temp_sync_file, offline_file)
            except requests.exceptions.RequestException:
                # เชื่อมต่อเว็บไม่ได้ ให้ดึงกลับไปรวมที่ไฟล์ออฟไลน์หลักเพื่อรอรอบถัดไป
                self.merge_temp_file_back(temp_sync_file, offline_file)

    def merge_temp_file_back(self, temp_file, target_file):
        with self.offline_lock:
            temp_records = []
            if os.path.exists(temp_file):
                try:
                    with open(temp_file, 'r', encoding='utf-8') as f:
                        temp_records = json.load(f)
                except:
                    pass
            
            if not temp_records:
                try: os.remove(temp_file)
                except: pass
                return
                
            target_records = []
            if os.path.exists(target_file):
                try:
                    with open(target_file, 'r', encoding='utf-8') as f:
                        target_records = json.load(f)
                except:
                    pass
            
            # รวมประวัติ (เอาออฟไลน์เก่าเรียงไว้หน้าสุดตามลำดับวันเวลา)
            merged = temp_records + target_records
            
            try:
                with open(target_file, 'w', encoding='utf-8') as f:
                    json.dump(merged, f, indent=2)
                os.remove(temp_file)
            except:
                pass

    def cloud_sync(self, name):
        current_time = time.time()
        last_sent_time = self.recorded.get(name, 0)
        
        # หน่วงเวลาสแกนหน้าซ้ำของเครื่องไว้ที่ 5 วินาทีเพื่อหลีกเลี่ยงการสแปมคิว
        if current_time - last_sent_time < 5:
            return
        
        self.recorded[name] = current_time
        
        def _task():
            try:
                res = requests.post(
                    WEB_APP_URL, 
                    json={"name_id": name}, 
                    timeout=4
                )
                if res.status_code == 200:
                    data = res.json()
                    status = data.get("status")
                    msg = data.get("message", "")
                    student = data.get("student", {})
                    std_name = student.get("name", name)
                    
                    if status == "success":
                        attendance_status = data.get("attendanceStatus", "ontime")
                        checkin_time = data.get("checkinTime", datetime.now().strftime("%H:%M:%S"))
                        std_id = student.get("id", "")
                        
                        if attendance_status == "late":
                            self.add_log(f"✅ เช็คชื่อสำเร็จ (สาย): {std_name}")
                            self.status_display_text = f"เช็คชื่อสำเร็จ (สาย) @ {checkin_time}"
                            self.status_display_color = (0, 165, 255) # Orange in BGR
                        else:
                            self.add_log(f"✅ เช็คชื่อสำเร็จ (ตรงเวลา): {std_name}")
                            self.status_display_text = f"เช็คชื่อสำเร็จ (ตรงเวลา) @ {checkin_time}"
                            self.status_display_color = (0, 255, 0) # Green in BGR
                        self.status_display_expiry = time.time() + 4.0
                    elif status == "already_checked_in":
                        self.add_log(f"⚠️ เช็คชื่อไปแล้ว: {std_name}")
                        self.status_display_text = "เช็คชื่อซ้ำ: เช็คชื่อเข้าเรียนไปแล้ว"
                        self.status_display_color = (0, 165, 255) # Orange/Yellow in BGR
                        self.status_display_expiry = time.time() + 4.0
                    elif status == "no_session":
                        self.add_log(f"❌ ปฏิเสธ: {msg}")
                        self.status_display_text = "ปิดเช็คชื่อ / ไม่อยู่ในเวลาเรียน"
                        self.status_display_color = (0, 0, 255) # Red in BGR
                        self.status_display_expiry = time.time() + 4.0
                else:
                    self.add_log(f"❌ เซิร์ฟเวอร์ตอบกลับไม่ถูกต้อง: {res.status_code}")
                    self.status_display_text = "เกิดข้อผิดพลาดในการเชื่อมต่อระบบ"
                    self.status_display_color = (0, 0, 255)
                    self.status_display_expiry = time.time() + 4.0
            except requests.exceptions.RequestException:
                # เซิร์ฟเวอร์เว็บล่ม / ตัดการเชื่อมต่อ -> สลับเข้าโหมดบันทึกออฟไลน์
                self.add_log(f"⚠️ ตัดการเชื่อมต่อเว็บแอป: สลับเข้าสู่โหมดออฟไลน์")
                parts = name.split('-', 1)
                std_name = parts[1] if len(parts) >= 2 else name
                
                # แสดงข้อความออฟไลน์บนหน้าจอ
                self.status_display_text = "บันทึกออฟไลน์แล้ว (รอเชื่อมต่อเซิร์ฟเวอร์)"
                self.status_display_color = (0, 255, 255) # Yellow in BGR
                self.status_display_expiry = time.time() + 4.0
                
                # กำหนดสถานะออฟไลน์ทันทีเพื่ออัปเดต UI หน้าจอ
                self.is_server_offline = True
                
                # บันทึกออฟไลน์ลงเครื่อง
                self.save_offline_attendance(name)
        
        threading.Thread(target=_task, daemon=True).start()

if __name__ == "__main__": 
    mp.set_start_method('spawn', force=True) 
    app = Pi5PortraitDash()
    app.mainloop()
