import tkinter as tk 

from tkinter import messagebox, simpledialog, scrolledtext 

import cv2 

import face_recognition 

import os 

import numpy as np 

import time 

import multiprocessing as mp 

import requests 

from datetime import datetime 

from PIL import Image as PILImage, ImageTk 

 

# --- CONFIGURATION (ปรับแต่งตามความเร็วเน็ต) --- 

ATTENDANCE_URL = "https://script.google.com/macros/s/AKfycbxmq4TG_A--c0mo7jj0_g96VUxAjOlsXP74SbcsLchJR5UdcJ_DOhzug291n0LVMbM8KA/exec" 

 

# --------------------------------------------------------- 

# FUNCTION: AI CORE PROCESS (รันแยกจาก UI) 

# --------------------------------------------------------- 

def ai_worker(frame_q, result_q, ctrl_ev, faces_dir, cache_path, rotation): 

    """กระบวนการคำนวณ AI ที่แยกออกมาเพื่อไม่ให้กินแรง Main Thread ของ RasPi""" 

    known_encs = [] 

    known_names = [] 

 

    # โหลด Cache ใบหน้า 

    if os.path.exists(cache_path): 

        import pickle 

        with open(cache_path, 'rb') as f: 

            cache = pickle.load(f) 

            for name, data in cache.items(): 

                known_encs.append(data['encoding']) 

                known_names.append(name) 

     

    known_encs = np.array(known_encs) 

     

    # เปิดกล้อง (สำหรับ RasPi แนะนำให้ใช้ VideoCapture(0)) 

    cap = cv2.VideoCapture(0) 

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480) # ลด Resolution ตั้งแต่ต้นทางเพื่อความเร็ว 

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) 

 

    while ctrl_ev.is_set(): 

        ret, frame = cap.read() 

        if not ret: continue 

 

        # จัดการหมุนจอ (Rotation) ก่อนส่งไปแสดงผล 

        if rotation == 90: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) 

        elif rotation == 180: frame = cv2.rotate(frame, cv2.ROTATE_180) 

        elif rotation == 270: frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) 

 

        # 1. ส่งภาพไปโชว์ที่ UI (ย่อขนาดเพื่อประหยัด Queue Bandwidth) 

        ui_view_frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8) 

        if not frame_q.full(): 

            frame_q.put(ui_view_frame) 

 

        # 2. คำนวณ AI (ทำ 1 เฟรม ข้าม 4 เฟรม เพื่อไม่ให้ RasPi ร้อนเกินไป) 

        # ใช้เทคนิคส่งข้อมูลพิกัดและชื่อกลับไปที่ UI Process 

        small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2) 

        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB) 

         

        locs = face_recognition.face_locations(rgb_small, model="hog") # hog เร็วกว่า cnn บน RasPi 

        encs = face_recognition.face_encodings(rgb_small, locs) 

         

        names = [] 

        for e in encs: 

            name = "Unknown" 

            if len(known_encs) > 0: 

                dist = np.linalg.norm(known_encs - e, axis=1) 

                if np.min(dist) < 0.45: 

                    name = known_names[np.argmin(dist)] 

            names.append(name) 

 

        if not result_q.full(): 

            result_q.put((locs, names)) 

 

    cap.release() 

 

# --------------------------------------------------------- 

# CLASS: PRO DASHBOARD (UI PROCESS) 

# --------------------------------------------------------- 

class RasPiFacePro(tk.Tk): 

    def __init__(self): 

        super().__init__() 

        self.title("RASPI FACE DASHBOARD V2") 

        self.attributes('-fullscreen', False) # ตั้งเป็น True ได้ถ้าจะรันแบบ Kiosk 

        self.geometry("1024x600") 

        self.configure(bg="#050505") 

 

        # Multiprocessing Setup 

        self.frame_q = mp.Queue(maxsize=1) 

        self.result_q = mp.Queue(maxsize=1) 

        self.ctrl_ev = mp.Event() 

        self.proc = None 

         

        self.rotation = 0 # 0, 90, 180, 270 

        self.last_locs = [] 

        self.last_names = [] 

        self.recorded = {} 

 

        self.init_ui() 

        self.init_core_paths() 

        self.main_loop() 

 

    def init_core_paths(self): 

        self.script_dir = os.path.dirname(os.path.abspath(__file__)) 

        self.faces_dir = os.path.join(self.script_dir, "faces") 

        self.cache_path = os.path.join(self.script_dir, "face_encodings_cache.pkl") 

 

    def init_ui(self): 

        # Layout Division 

        self.columnconfigure(0, weight=4) 

        self.columnconfigure(1, weight=1) 

        self.rowconfigure(0, weight=1) 

 

        # Video Frame 

        self.v_frame = tk.Frame(self, bg="#000") 

        self.v_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10) 

        self.v_label = tk.Label(self.v_frame, bg="#000") 

        self.v_label.pack(fill=tk.BOTH, expand=True) 

 

        # Side Panel 

        self.p_frame = tk.Frame(self, bg="#111", width=250) 

        self.p_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10) 

        self.p_frame.grid_propagate(False) 

 

        tk.Label(self.p_frame, text="RASPI CORE", font=("Verdana", 14, "bold"), fg="#00ffcc", bg="#111").pack(pady=20) 

         

        self.btn_run = tk.Button(self.p_frame, text="START SYSTEM", bg="#003322", fg="#00ff88", font=("Arial", 10, "bold"), 

                                command=self.toggle_engine, bd=0, height=2, cursor="hand2") 

        self.btn_run.pack(fill=tk.X, padx=15, pady=5) 

 

        tk.Button(self.p_frame, text="ROTATE 90°", bg="#332200", fg="#ffcc00", command=self.rotate, bd=0).pack(fill=tk.X, padx=15, pady=5) 

 

        self.log = scrolledtext.ScrolledText(self.p_frame, bg="#050505", fg="#666", font=("Monospace", 8), bd=0) 

        self.log.pack(fill=tk.BOTH, expand=True, padx=10, pady=20) 

 

    def rotate(self): 

        self.rotation = (self.rotation + 90) % 360 

        self.add_log(f"Rotate to {self.rotation}") 

        if self.ctrl_ev.is_set(): 

            self.toggle_engine() # Restart to apply rotation 

            self.toggle_engine() 

 

    def toggle_engine(self): 

        if not self.ctrl_ev.is_set(): 

            self.ctrl_ev.set() 

            self.proc = mp.Process(target=ai_worker, args=(self.frame_q, self.result_q, self.ctrl_ev, self.faces_dir, self.cache_path, self.rotation)) 

            self.proc.start() 

            self.btn_run.config(text="STOP SYSTEM", bg="#441111", fg="#ff5555") 

        else: 

            self.ctrl_ev.clear() 

            if self.proc: self.proc.terminate() 

            self.btn_run.config(text="START SYSTEM", bg="#003322", fg="#00ff88") 

            self.v_label.config(image='') 

 

    def add_log(self, m): 

        self.log.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')}> {m}\n") 

        self.log.see(tk.END) 

 

    def main_loop(self): 

        # 1. Update Results 

        try: 

            while not self.result_q.empty(): 

                self.last_locs, self.last_names = self.result_q.get_nowait() 

                for n in self.last_names: 

                    if n != "Unknown": self.cloud_sync(n) 

        except: pass 

 

        # 2. Update Video UI 

        try: 

            if not self.frame_q.empty(): 

                frame = self.frame_q.get_nowait() 

                 

                # วาดกรอบบน UI (คำนวณสเกลคืนจากที่ย่อไป 0.2 ใน Process AI) 

                for (t, r, b, l), name in zip(self.last_locs, self.last_names): 

                    t, r, b, l = t*5, r*5, b*5, l*5 # ขยายกลับ 5 เท่าตาม fx=0.2 

                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255) 

                    cv2.rectangle(frame, (l, t), (r, b), color, 2) 

                    cv2.putText(frame, name, (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1) 

 

                img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 

                 

                # Responsive Resize ให้เต็มจอ RasPi 

                w, h = self.v_label.winfo_width(), self.v_label.winfo_height() 

                if w > 10: img = img.resize((w, h), PILImage.Resampling.NEAREST) # NEAREST เร็วสุดสำหรับ Pi 

 

                tk_img = ImageTk.PhotoImage(image=img) 

                self.v_label.imgtk = tk_img 

                self.v_label.config(image=tk_img) 

        except: pass 

 

        self.after(25, self.main_loop) 

 

    def cloud_sync(self, name): 

        if time.time() - self.recorded.get(name, 0) < 300: return # เช็คชื่อซ้ำทุก 5 นาที 

        self.recorded[name] = time.time() 

        def _task(): 

            try: 

                p = name.split('-') 

                requests.get(ATTENDANCE_URL, params={"id":p[0],"name":p[1],"status":"มาเรียน"}, timeout=5) 

                self.after(0, lambda: self.add_log(f"Synced: {name}")) 

            except: self.recorded.pop(name, None) 

        import threading 

        threading.Thread(target=_task, daemon=True).start() 

 

if __name__ == "__main__": 

    mp.set_start_method('spawn', force=True) # ปลอดภัยที่สุดสำหรับ RasPi/Linux 

    app = RasPiFacePro() 

    app.mainloop() 
