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
from datetime import datetime 
from PIL import Image as PILImage, ImageTk 

# --- CONFIGURATION --- 
ATTENDANCE_URL = "https://script.google.com/macros/s/AKfycbxmq4TG_A--c0mo7jj0_g96VUxAjOlsXP74SbcsLchJR5UdcJ_DOhzug291n0LVMbM8KA/exec" 

# --------------------------------------------------------- 
# FUNCTION: AI CORE PROCESS
# --------------------------------------------------------- 
def ai_worker(frame_q, result_q, ctrl_ev, faces_dir, cache_path, rotation_val): 
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
     
    known_encs = np.array(known_encs) 
    cap = cv2.VideoCapture(0) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while ctrl_ev.is_set(): 
        ret, frame = cap.read() 
        if not ret: continue 

        # หมุนภาพตามที่ตั้งไว้ใน UI
        rot = rotation_val.value
        if rot == 90: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) 
        elif rot == 180: frame = cv2.rotate(frame, cv2.ROTATE_180) 
        elif rot == 270: frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) 

        if not frame_q.full(): 
            frame_q.put(frame) 

        small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2) 
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB) 
         
        locs = face_recognition.face_locations(rgb_small, model="hog") 
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
# CLASS: DASHBOARD (Optimized for Portrait Screen)
# --------------------------------------------------------- 
class Pi5PortraitDash(tk.Tk): 
    def __init__(self): 
        super().__init__() 
        self.title("PI 5 PORTRAIT DASHBOARD") 
        # ตั้งขนาดให้เหมาะกับจอแนวตั้ง (เช่น 600x1024)
        self.geometry("600x1024") 
        self.configure(bg="#050505") 

        self.frame_q = mp.Queue(maxsize=2) 
        self.result_q = mp.Queue(maxsize=2) 
        self.ctrl_ev = mp.Event() 
        self.rotation_val = mp.Value('i', 90) # เริ่มต้นที่ 90 องศาสำหรับจอแนวตั้ง
        self.proc = None 
         
        self.last_locs = [] 
        self.last_names = [] 
        self.recorded = {} 
        self.current_frame = None 
        self.capture_count = 0

        self.init_core_paths() 
        self.init_ui() 
        
        threading.Thread(target=self.train_ai, daemon=True).start()
        self.main_loop() 

    def init_core_paths(self): 
        self.script_dir = os.path.dirname(os.path.abspath(__file__)) 
        self.faces_dir = os.path.join(self.script_dir, "faces") 
        self.cache_path = os.path.join(self.script_dir, "face_encodings_cache.pkl") 
        if not os.path.exists(self.faces_dir): os.makedirs(self.faces_dir) 

    def init_ui(self): 
        # จัด Grid แบบแถวเดียว (แนวตั้ง)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=3) # ส่วนกล้อง (ใหญ่)
        self.rowconfigure(1, weight=1) # ส่วนควบคุม (เล็กกว่า)

        # 1. Video Frame (Top)
        self.v_frame = tk.Frame(self, bg="#000") 
        self.v_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.v_label = tk.Label(self.v_frame, bg="#000") 
        self.v_label.pack(fill=tk.BOTH, expand=True) 

        # 2. Control Panel (Bottom)
        self.p_frame = tk.Frame(self, bg="#111") 
        self.p_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # แบ่งส่วนภายใน Panel เป็น 2 คอลัมน์ (ซ้าย: ปุ่มระบบ, ขวา: ลงทะเบียน)
        self.p_frame.columnconfigure(0, weight=1)
        self.p_frame.columnconfigure(1, weight=1)

        # --- ส่วนซ้าย: ระบบหลัก ---
        left_ctrl = tk.Frame(self.p_frame, bg="#111")
        left_ctrl.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        tk.Label(left_ctrl, text="SYSTEM STATUS", font=("Verdana", 10, "bold"), fg="#00ffcc", bg="#111").pack(pady=5)
        self.btn_run = tk.Button(left_ctrl, text="STOP SYSTEM", bg="#441111", fg="#ff5555", font=("Arial", 10, "bold"), 
                                command=self.toggle_engine, bd=0, height=2) 
        self.btn_run.pack(fill=tk.X, pady=2) 

        self.lbl_rot = tk.Label(left_ctrl, text="Rotate: 90°", fg="#ffcc00", bg="#111", font=("Arial", 9))
        self.lbl_rot.pack()
        tk.Button(left_ctrl, text="ROTATE SCREEN", bg="#332200", fg="#ffcc00", command=self.rotate, bd=0).pack(fill=tk.X, pady=2)

        # --- ส่วนขวา: ลงทะเบียน ---
        right_ctrl = tk.Frame(self.p_frame, bg="#111")
        right_ctrl.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        tk.Label(right_ctrl, text="REGISTRATION", font=("Verdana", 10, "bold"), fg="#ffcc00", bg="#111").pack(pady=5)
        self.ent_name = tk.Entry(right_ctrl, bg="#222", fg="white", bd=0, font=("Arial", 11))
        self.ent_name.pack(fill=tk.X, pady=2)

        self.btn_capture = tk.Button(right_ctrl, text="📸 CAPTURE (0/10)", bg="#2980b9", fg="white", font=("Arial", 10, "bold"), 
                                command=self.capture_photo, bd=0, height=2) 
        self.btn_capture.pack(fill=tk.X, pady=2) 

        self.btn_train = tk.Button(right_ctrl, text="🔄 RELOAD AI", bg="#8e44ad", fg="white", font=("Arial", 9, "bold"), 
                                command=self.manual_train, bd=0) 
        self.btn_train.pack(fill=tk.X, pady=2)

        # 3. Log Area (Full Width at the very bottom)
        self.log = scrolledtext.ScrolledText(self.p_frame, bg="#050505", fg="#666", font=("Monospace", 8), bd=0, height=4) 
        self.log.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

    def rotate(self): 
        new_rot = (self.rotation_val.value + 90) % 360
        self.rotation_val.value = new_rot
        self.lbl_rot.config(text=f"Rotate: {new_rot}°")
        self.add_log(f"Camera rotated to {new_rot}°")

    def train_ai(self):
        self.add_log("AI: Training...")
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
        self.add_log("AI: Ready")
        self.after(0, self.toggle_engine)

    def manual_train(self):
        if self.ctrl_ev.is_set(): self.toggle_engine()
        threading.Thread(target=self.train_ai, daemon=True).start()

    def toggle_engine(self): 
        if not self.ctrl_ev.is_set(): 
            self.ctrl_ev.set() 
            self.proc = mp.Process(target=ai_worker, args=(self.frame_q, self.result_q, self.ctrl_ev, self.faces_dir, self.cache_path, self.rotation_val)) 
            self.proc.start() 
            self.btn_run.config(text="STOP SYSTEM", bg="#441111", fg="#ff5555") 
        else: 
            self.ctrl_ev.clear() 
            if self.proc: self.proc.terminate() 
            self.btn_run.config(text="START SYSTEM", bg="#003322", fg="#00ff88") 
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
            messagebox.showinfo("Done", "ลงทะเบียนครบ 10 รูปแล้ว")
            self.ent_name.delete(0, tk.END); self.capture_count = 0
            self.btn_capture.config(text="📸 CAPTURE (0/10)")

    def add_log(self, m): 
        self.log.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')}> {m}\n") 
        self.log.see(tk.END) 

    def main_loop(self): 
        is_reg = len(self.ent_name.get().strip()) > 0
        try: 
            while not self.result_q.empty(): 
                locs, names = self.result_q.get_nowait()
                if not is_reg:
                    self.last_locs, self.last_names = locs, names
                    for n in self.last_names: 
                        if n != "Unknown": self.cloud_sync(n) 
                else: self.last_locs, self.last_names = [], []
        except: pass 

        try: 
            if not self.frame_q.empty(): 
                raw = self.frame_q.get_nowait() 
                self.current_frame = raw.copy()
                frame = raw.copy()
                if not is_reg:
                    for (t, r, b, l), name in zip(self.last_locs, self.last_names): 
                        t, r, b, l = t*5, r*5, b*5, l*5 
                        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2) 
                        cv2.putText(frame, name, (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) 
                else:
                    cv2.putText(frame, "REGISTRATION MODE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 204, 255), 2)

                img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
                w, h = self.v_label.winfo_width(), self.v_label.winfo_height() 
                if w > 10: img = img.resize((w, h), PILImage.Resampling.NEAREST) 
                tk_img = ImageTk.PhotoImage(image=img) 
                self.v_label.imgtk = tk_img; self.v_label.config(image=tk_img) 
        except: pass 
        self.after(16, self.main_loop) 

    def cloud_sync(self, name): 
        if time.time() - self.recorded.get(name, 0) < 300: return 
        self.recorded[name] = time.time() 
        def _task(): 
            try: 
                p = name.split('-') 
                if len(p) >= 2: requests.get(ATTENDANCE_URL, params={"id":p[0],"name":p[1],"status":"มาเรียน"}, timeout=5) 
            except: self.recorded.pop(name, None) 
        threading.Thread(target=_task, daemon=True).start() 

if __name__ == "__main__": 
    mp.set_start_method('spawn', force=True) 
    app = Pi5PortraitDash(); app.mainloop() 

