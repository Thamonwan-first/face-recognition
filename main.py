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
from datetime import datetime 
from PIL import Image as PILImage, ImageTk 

# --- CONFIGURATION --- 
ATTENDANCE_URL = "https://script.google.com/macros/s/AKfycbxmq4TG_A--c0mo7jj0_g96VUxAjOlsXP74SbcsLchJR5UdcJ_DOhzug291n0LVMbM8KA/exec" 

# --------------------------------------------------------- 
# FUNCTION: AI CORE PROCESS (Optimized for Pi 5 CPU)
# --------------------------------------------------------- 
def ai_worker(frame_q, result_q, ctrl_ev, faces_dir, cache_path, rotation): 
    known_encs = [] 
    known_names = [] 

    # Load Cache
    if os.path.exists(cache_path): 
        import pickle 
        try:
            with open(cache_path, 'rb') as f: 
                cache = pickle.load(f) 
                for name, data in cache.items(): 
                    known_encs.append(data['encoding']) 
                    known_names.append(name) 
        except: pass
     
    known_encs = np.array(known_encs) 
     
    # Pi 5: Use standard backend (V4L2)
    cap = cv2.VideoCapture(0) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while ctrl_ev.is_set(): 
        ret, frame = cap.read() 
        if not ret: continue 

        # Rotation
        if rotation == 90: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) 
        elif rotation == 180: frame = cv2.rotate(frame, cv2.ROTATE_180) 
        elif rotation == 270: frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) 

        # Send to UI (High Priority)
        if not frame_q.full(): 
            frame_q.put(frame) 

        # AI Recognition (Big O: O(N) where N is detected faces)
        # Pi 5 can handle higher resolution, but 0.2 is best for zero-latency
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
# CLASS: DASHBOARD (UI PROCESS)
# --------------------------------------------------------- 
class Pi5FacePro(tk.Tk): 
    def __init__(self): 
        super().__init__() 
        self.title("PI 5 FACE DASHBOARD") 
        # Pi 5: Optimized for 7-10 inch Touchscreens
        self.geometry("1024x600") 
        self.configure(bg="#050505") 

        self.frame_q = mp.Queue(maxsize=2) 
        self.result_q = mp.Queue(maxsize=2) 
        self.ctrl_ev = mp.Event() 
        self.proc = None 
         
        self.rotation = 0 
        self.last_locs = [] 
        self.last_names = [] 
        self.recorded = {} 
        self.current_frame = None 
        self.capture_count = 0

        self.init_core_paths() 
        self.init_ui() 
        
        # Auto Start (Wait for UI ready)
        self.after(500, self.toggle_engine) 
        self.main_loop() 

    def init_core_paths(self): 
        self.script_dir = os.path.dirname(os.path.abspath(__file__)) 
        self.faces_dir = os.path.join(self.script_dir, "faces") 
        self.cache_path = os.path.join(self.script_dir, "face_encodings_cache.pkl") 
        if not os.path.exists(self.faces_dir): 
            os.makedirs(self.faces_dir) 

    def init_ui(self): 
        self.columnconfigure(0, weight=4) 
        self.columnconfigure(1, weight=1) 
        self.rowconfigure(0, weight=1) 

        self.v_frame = tk.Frame(self, bg="#000") 
        self.v_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10) 
        self.v_label = tk.Label(self.v_frame, bg="#000") 
        self.v_label.pack(fill=tk.BOTH, expand=True) 

        self.p_frame = tk.Frame(self, bg="#111", width=280) 
        self.p_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10) 
        self.p_frame.grid_propagate(False) 

        tk.Label(self.p_frame, text="PI 5 CORE", font=("Verdana", 14, "bold"), fg="#00ffcc", bg="#111").pack(pady=10) 
         
        self.btn_run = tk.Button(self.p_frame, text="START SYSTEM", bg="#003322", fg="#00ff88", font=("Arial", 10, "bold"), 
                                command=self.toggle_engine, bd=0, height=2, cursor="hand2") 
        self.btn_run.pack(fill=tk.X, padx=15, pady=5) 

        tk.Frame(self.p_frame, height=2, bg="#333").pack(fill=tk.X, pady=10)
        tk.Label(self.p_frame, text="REGISTRATION", font=("Verdana", 10, "bold"), fg="#ffcc00", bg="#111").pack()
        
        tk.Label(self.p_frame, text="Student ID-Name:", fg="#999", bg="#111", font=("Arial", 8)).pack(anchor="w", padx=15)
        self.ent_name = tk.Entry(self.p_frame, bg="#222", fg="white", insertbackground="white", bd=0)
        self.ent_name.pack(fill=tk.X, padx=15, pady=5)

        self.btn_capture = tk.Button(self.p_frame, text="📸 CAPTURE (0/10)", bg="#2980b9", fg="white", font=("Arial", 10, "bold"), 
                                command=self.capture_photo, bd=0, height=2, cursor="hand2") 
        self.btn_capture.pack(fill=tk.X, padx=15, pady=5) 
        
        self.btn_reset_reg = tk.Button(self.p_frame, text="RESET / SCAN MODE", bg="#c0392b", fg="white", font=("Arial", 9, "bold"), 
                                command=self.reset_registration, bd=0, height=2) 
        self.btn_reset_reg.pack(fill=tk.X, padx=15, pady=5)

        tk.Button(self.p_frame, text="ROTATE 90°", bg="#332200", fg="#ffcc00", command=self.rotate, bd=0).pack(fill=tk.X, padx=15, pady=10) 

        self.log = scrolledtext.ScrolledText(self.p_frame, bg="#050505", fg="#666", font=("Monospace", 8), bd=0) 
        self.log.pack(fill=tk.BOTH, expand=True, padx=10, pady=10) 

    def rotate(self): 
        self.rotation = (self.rotation + 90) % 360 
        if self.ctrl_ev.is_set(): 
            self.toggle_engine(); self.toggle_engine() 

    def toggle_engine(self): 
        if not self.ctrl_ev.is_set(): 
            self.ctrl_ev.set() 
            self.proc = mp.Process(target=ai_worker, args=(self.frame_q, self.result_q, self.ctrl_ev, self.faces_dir, self.cache_path, self.rotation)) 
            self.proc.start() 
            self.btn_run.config(text="STOP SYSTEM", bg="#441111", fg="#ff5555") 
            self.add_log("Pi 5 Engine Online")
        else: 
            self.ctrl_ev.clear() 
            if self.proc: self.proc.terminate() 
            self.btn_run.config(text="START SYSTEM", bg="#003322", fg="#00ff88") 
            self.v_label.config(image='') 
            self.current_frame = None

    def capture_photo(self):
        name = self.ent_name.get().strip()
        if not name or not self.current_frame is not None: return

        save_path = os.path.join(self.faces_dir, name)
        os.makedirs(save_path, exist_ok=True)
        img_name = f"{name}_{int(time.time())}.jpg"
        cv2.imwrite(os.path.join(save_path, img_name), self.current_frame)
        
        self.capture_count += 1
        self.btn_capture.config(text=f"📸 CAPTURE ({self.capture_count}/10)")
        self.add_log(f"Captured {self.capture_count}/10")
        
        if self.capture_count >= 10:
            messagebox.showinfo("Success", f"ลงทะเบียน {name} ครบแล้ว!")
            self.reset_registration()

    def reset_registration(self):
        self.ent_name.delete(0, tk.END)
        self.capture_count = 0
        self.btn_capture.config(text="📸 CAPTURE (0/10)")
        self.add_log("Mode: Scanning")

    def add_log(self, m): 
        self.log.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')}> {m}\n") 
        self.log.see(tk.END) 

    def main_loop(self): 
        is_registering = len(self.ent_name.get().strip()) > 0
        try: 
            while not self.result_q.empty(): 
                locs, names = self.result_q.get_nowait()
                if not is_registering:
                    self.last_locs, self.last_names = locs, names
                    for n in self.last_names: 
                        if n != "Unknown": self.cloud_sync(n) 
                else:
                    self.last_locs, self.last_names = [], []
        except: pass 

        try: 
            if not self.frame_q.empty(): 
                raw_frame = self.frame_q.get_nowait() 
                self.current_frame = raw_frame.copy() 
                
                frame = raw_frame.copy()
                if not is_registering:
                    for (t, r, b, l), name in zip(self.last_locs, self.last_names): 
                        t, r, b, l = t*5, r*5, b*5, l*5 
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255) 
                        cv2.rectangle(frame, (l, t), (r, b), color, 2) 
                        cv2.putText(frame, name, (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1) 
                else:
                    cv2.putText(frame, "REGISTRATION MODE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 204, 255), 2)

                img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
                w, h = self.v_label.winfo_width(), self.v_label.winfo_height() 
                if w > 10: img = img.resize((w, h), PILImage.Resampling.NEAREST) 

                tk_img = ImageTk.PhotoImage(image=img) 
                self.v_label.imgtk = tk_img 
                self.v_label.config(image=tk_img) 
        except: pass 

        self.after(16, self.main_loop) # Pi 5 can handle ~60 FPS UI

    def cloud_sync(self, name): 
        if time.time() - self.recorded.get(name, 0) < 300: return 
        self.recorded[name] = time.time() 
        def _task(): 
            try: 
                p = name.split('-') 
                if len(p) >= 2:
                    requests.get(ATTENDANCE_URL, params={"id":p[0],"name":p[1],"status":"มาเรียน"}, timeout=5) 
                    self.after(0, lambda: self.add_log(f"Synced: {name}")) 
            except: self.recorded.pop(name, None) 
        threading.Thread(target=_task, daemon=True).start() 

if __name__ == "__main__": 
    # Important for Pi OS / Linux
    mp.set_start_method('spawn', force=True) 
    app = Pi5FacePro() 
    app.mainloop() 
