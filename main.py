import tkinter as tk
from tkinter import ttk, messagebox
import threading, time, json, os
from collections import deque
import cv2
import numpy as np

try:
    import mss
except ImportError:
    mss = None

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(APP_DIR, "config.json")

DEFAULTS = {
    "source": "Capture Card",
    "device_index": 0,
    "monitor_index": 1,
    "roi_x": 0.25, "roi_y": 0.15, "roi_w": 0.50, "roi_h": 0.70,
    "motion_threshold": 7.0,
    "brightness_threshold": 18.0,
    "offset_ms": 0,
    "show_preview": True
}

def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {**DEFAULTS, **data}
    except Exception:
        return DEFAULTS.copy()

def save_config(data):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

class VisionEngine:
    """Visual cue analyzer. It reports candidate visual events; it does not press/release buttons."""
    def __init__(self, config, callback):
        self.config = config
        self.callback = callback
        self.running = False
        self.thread = None
        self.cap = None
        self.sct = None
        self.prev_gray = None
        self.motion_history = deque(maxlen=30)
        self.last_event = 0
        self.frames = 0
        self.started = 0

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.sct = None

    def _open_source(self):
        if self.config["source"] == "Capture Card":
            idx = int(self.config.get("device_index", 0))
            self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            if not self.cap.isOpened():
                self.callback("ERROR", f"Could not open capture card at camera index {idx}. Try 0, 1, or 2.")
                self.running = False
                return False
        else:
            if mss is None:
                self.callback("ERROR", "Remote Play desktop capture needs the optional 'mss' package. Install requirements.txt.")
                self.running = False
                return False
            self.sct = mss.mss()
        return True

    def _get_frame(self):
        if self.config["source"] == "Capture Card":
            ok, frame = self.cap.read()
            return frame if ok else None
        monitors = self.sct.monitors
        mi = int(self.config.get("monitor_index", 1))
        mi = max(1, min(mi, len(monitors)-1))
        shot = self.sct.grab(monitors[mi])
        arr = np.asarray(shot)
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    def _run(self):
        if not self._open_source():
            return
        self.prev_gray = None
        self.started = time.time()
        self.frames = 0
        while self.running:
            frame = self._get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            self.frames += 1
            h, w = frame.shape[:2]
            x = int(w * float(self.config["roi_x"]))
            y = int(h * float(self.config["roi_y"]))
            rw = max(20, int(w * float(self.config["roi_w"])))
            rh = max(20, int(h * float(self.config["roi_h"])))
            x, y = max(0, x), max(0, y)
            roi = frame[y:min(h, y+rh), x:min(w, x+rw)]
            gray = cv2.cvtColor(cv2.resize(roi, (320, 240)), cv2.COLOR_BGR2GRAY)
            motion = 0.0
            brightness_delta = 0.0
            if self.prev_gray is not None:
                diff = cv2.absdiff(gray, self.prev_gray)
                motion = float(np.mean(diff))
                brightness_delta = abs(float(np.mean(gray)) - float(np.mean(self.prev_gray)))
            self.prev_gray = gray
            self.motion_history.append(motion)
            avg_motion = float(np.mean(self.motion_history)) if self.motion_history else 0.0
            threshold = float(self.config["motion_threshold"])
            bthreshold = float(self.config["brightness_threshold"])
            now = time.time()
            candidate = (motion >= threshold or brightness_delta >= bthreshold)
            if candidate and now - self.last_event > 0.65:
                self.last_event = now
                confidence = min(99, int(45 + min(54, max(0, motion-threshold)*2.2)))
                self.callback("CUE", {
                    "motion": round(motion, 2),
                    "brightness": round(brightness_delta, 2),
                    "confidence": confidence,
                    "offset": int(self.config["offset_ms"])
                })
            fps = self.frames / max(0.1, now-self.started)
            self.callback("FRAME", {
                "frame": frame.copy(),
                "motion": round(motion, 2),
                "fps": round(fps, 1),
                "candidate": candidate
            })
        if self.cap:
            self.cap.release()
            self.cap = None
        self.sct = None

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("2K27 AUTO GREEN CV")
        self.root.geometry("1080x700")
        self.root.minsize(900, 600)
        self.cfg = load_config()
        self.engine = None
        self.preview_image = None
        self.enabled = False
        self.events = 0
        self._style()
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _style(self):
        self.root.configure(bg="#101318")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#101318")
        style.configure("Panel.TFrame", background="#191f28")
        style.configure("TLabel", background="#101318", foreground="#e8edf4", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 21, "bold"), foreground="#ffffff")
        style.configure("Muted.TLabel", foreground="#9ba8b8")
        style.configure("TButton", padding=8, font=("Segoe UI", 10, "bold"))
        style.configure("TCheckbutton", background="#191f28", foreground="#e8edf4")
        style.configure("TCombobox", fieldbackground="#252d38", background="#252d38", foreground="#ffffff")
        style.configure("Horizontal.TScale", background="#191f28")

    def _build(self):
        header = ttk.Frame(self.root, padding=(20, 16))
        header.pack(fill="x")
        ttk.Label(header, text="2K27 AUTO GREEN CV", style="Title.TLabel").pack(side="left")
        self.status = tk.Label(header, text="● OFFLINE", bg="#101318", fg="#ff6565",
                               font=("Segoe UI", 10, "bold"))
        self.status.pack(side="right")

        body = ttk.Frame(self.root, padding=(18, 0, 18, 18))
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, style="Panel.TFrame", padding=16)
        left.pack(side="left", fill="y", padx=(0, 14))
        right = ttk.Frame(body, style="Panel.TFrame", padding=12)
        right.pack(side="left", fill="both", expand=True)

        tk.Label(left, text="VIDEO INPUT", bg="#191f28", fg="#8fdaa8",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(left, text="Input mode").pack(anchor="w", pady=(10, 3))
        self.source_var = tk.StringVar(value=self.cfg["source"])
        self.source_box = ttk.Combobox(left, textvariable=self.source_var,
             values=["Capture Card", "Xbox Remote Play (screen capture)"], state="readonly", width=28)
        self.source_box.pack(fill="x")
        self.source_box.bind("<<ComboboxSelected>>", self._source_changed)

        ttk.Label(left, text="Capture card device index (usually 0)").pack(anchor="w", pady=(12, 3))
        self.device_var = tk.IntVar(value=self.cfg["device_index"])
        ttk.Spinbox(left, from_=0, to=10, textvariable=self.device_var, width=8).pack(anchor="w")
        ttk.Label(left, text="Remote Play: open Xbox Remote Play on this PC,\nthen select its display below.",
                  style="Muted.TLabel").pack(anchor="w", pady=(8, 2))
        self.monitor_var = tk.IntVar(value=self.cfg["monitor_index"])
        ttk.Label(left, text="Monitor number (1 = primary)").pack(anchor="w", pady=(8, 3))
        ttk.Spinbox(left, from_=1, to=8, textvariable=self.monitor_var, width=8).pack(anchor="w")

        ttk.Separator(left).pack(fill="x", pady=14)
        tk.Label(left, text="DETECTION", bg="#191f28", fg="#8fdaa8",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(left, text="Motion threshold").pack(anchor="w", pady=(8, 2))
        self.motion_var = tk.DoubleVar(value=self.cfg["motion_threshold"])
        ttk.Scale(left, from_=1, to=35, variable=self.motion_var, orient="horizontal").pack(fill="x")
        self.motion_label = ttk.Label(left, text=f'{self.motion_var.get():.1f}')
        self.motion_label.pack(anchor="e")
        self.motion_var.trace_add("write", lambda *_: self.motion_label.config(text=f"{self.motion_var.get():.1f}"))

        ttk.Label(left, text="Brightness-change threshold").pack(anchor="w", pady=(8, 2))
        self.bright_var = tk.DoubleVar(value=self.cfg["brightness_threshold"])
        ttk.Scale(left, from_=2, to=60, variable=self.bright_var, orient="horizontal").pack(fill="x")
        self.bright_label = ttk.Label(left, text=f'{self.bright_var.get():.1f}')
        self.bright_label.pack(anchor="e")
        self.bright_var.trace_add("write", lambda *_: self.bright_label.config(text=f"{self.bright_var.get():.1f}"))

        ttk.Label(left, text="Timing offset (ms)").pack(anchor="w", pady=(8, 2))
        self.offset_var = tk.IntVar(value=self.cfg["offset_ms"])
        ttk.Spinbox(left, from_=-250, to=250, increment=5, textvariable=self.offset_var, width=10).pack(anchor="w")

        ttk.Separator(left).pack(fill="x", pady=14)
        self.start_btn = ttk.Button(left, text="START ANALYZER", command=self.start)
        self.start_btn.pack(fill="x", pady=3)
        ttk.Button(left, text="STOP ANALYZER", command=self.stop).pack(fill="x", pady=3)
        ttk.Button(left, text="SAVE SETTINGS", command=self.save).pack(fill="x", pady=3)
        ttk.Button(left, text="CLEAR EVENT LOG", command=self.clear_log).pack(fill="x", pady=3)

        tk.Label(right, text="LIVE VIDEO / REMOTE PLAY PREVIEW", bg="#191f28", fg="#8fdaa8",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.preview = tk.Label(right, text="Start the analyzer to view your Xbox feed.",
                                bg="#080a0d", fg="#aab5c3", font=("Segoe UI", 12),
                                width=80, height=22)
        self.preview.pack(fill="both", expand=True, pady=(8, 10))
        metrics = ttk.Frame(right, style="Panel.TFrame")
        metrics.pack(fill="x")
        self.motion_metric = self._metric(metrics, "MOTION", "—")
        self.fps_metric = self._metric(metrics, "FPS", "—")
        self.conf_metric = self._metric(metrics, "LAST CUE", "—")
        self.events_metric = self._metric(metrics, "EVENTS", "0")
        tk.Label(right, text="EVENT LOG", bg="#191f28", fg="#8fdaa8",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(12, 4))
        self.log = tk.Listbox(right, height=5, bg="#0d1117", fg="#dce5ef",
                              selectbackground="#24583b", borderwidth=0,
                              font=("Consolas", 9))
        self.log.pack(fill="x")

        note = tk.Label(self.root,
            text="Visual timing assistant only — it does not automate controller inputs or guarantee green releases.",
            bg="#101318", fg="#9ba8b8", font=("Segoe UI", 9))
        note.pack(pady=(0, 10))

    def _metric(self, parent, title, value):
        box = ttk.Frame(parent, style="Panel.TFrame", padding=(8, 3))
        box.pack(side="left", fill="x", expand=True)
        tk.Label(box, text=title, bg="#191f28", fg="#9ba8b8",
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        lab = tk.Label(box, text=value, bg="#191f28", fg="#ffffff",
                       font=("Segoe UI", 15, "bold"))
        lab.pack(anchor="w")
        return lab

    def _source_changed(self, *_):
        pass

    def _config_now(self):
        return {
            **self.cfg,
            "source": "Capture Card" if self.source_var.get() == "Capture Card" else "Remote Play",
            "device_index": int(self.device_var.get()),
            "monitor_index": int(self.monitor_var.get()),
            "motion_threshold": float(self.motion_var.get()),
            "brightness_threshold": float(self.bright_var.get()),
            "offset_ms": int(self.offset_var.get())
        }

    def save(self):
        self.cfg = self._config_now()
        save_config(self.cfg)
        messagebox.showinfo("Saved", "Settings saved to config.json.")

    def start(self):
        if self.engine:
            self.stop()
        self.cfg = self._config_now()
        self.engine = VisionEngine(self.cfg, self._on_engine)
        self.engine.start()
        self.status.config(text="● STARTING", fg="#f0c66a")

    def stop(self):
        if self.engine:
            self.engine.stop()
            self.engine = None
        self.status.config(text="● OFFLINE", fg="#ff6565")

    def _on_engine(self, kind, data):
        self.root.after(0, lambda: self._handle_event(kind, data))

    def _handle_event(self, kind, data):
        if kind == "ERROR":
            self.status.config(text="● ERROR", fg="#ff6565")
            self.log.insert(0, data)
            return
        if kind == "CUE":
            self.events += 1
            self.conf_metric.config(text=f'{data["confidence"]}%')
            self.events_metric.config(text=str(self.events))
            self.log.insert(0, time.strftime("%H:%M:%S") +
                f'  Candidate cue | motion {data["motion"]} | brightness {data["brightness"]} | offset {data["offset"]:+d} ms')
            if self.log.size() > 100:
                self.log.delete(100, tk.END)
        elif kind == "FRAME":
            self.status.config(text="● LIVE", fg="#70e090")
            self.motion_metric.config(text=str(data["motion"]))
            self.fps_metric.config(text=str(data["fps"]))
            frame = data["frame"]
            # Scale preview to fit while preserving aspect ratio.
            h, w = frame.shape[:2]
            maxw, maxh = 760, 430
            scale = min(maxw/w, maxh/h)
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            cv2.putText(frame, "CANDIDATE VISUAL CHANGE" if data["candidate"] else "ANALYZING",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (40, 230, 110) if data["candidate"] else (220, 220, 220), 2)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image, ImageTk
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.preview.configure(image=img, text="")
            self.preview.image = img

    def clear_log(self):
        self.log.delete(0, tk.END)
        self.events = 0
        self.events_metric.config(text="0")

    def close(self):
        self.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
