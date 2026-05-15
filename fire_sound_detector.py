"""
🔥 FireGuard Audio — Fire Detection via YAMNet
================================================
Science Exhibition · TAB → 10s audio clip → YAMNet AI → result

Uses Google's YAMNet pretrained model (521 AudioSet classes).
Runs 100% offline after download_model.py has been run once.

YAMNet fire-relevant classes detected:
  🔥  Fire, Crackling, Hiss           → FIRE alert
  🚨  Smoke detector, Fire alarm       → SMOKE ALARM alert
  ✅  Everything else (low fire score) → SAFE

Controls
────────
  TAB     Record 10 s and analyse
  Q/ESC   Quit
  H       Toggle info panel
  S       Save result to log
  C       Clear / reset display

Requirements
────────────
  pip install tensorflow sounddevice numpy blessed
  python download_model.py  (once, needs internet)
  python fire_detector.py
"""

import os, sys, json, time, threading, platform, datetime, collections
from pathlib import Path

MODEL_DIR  = Path("yamnet_local")
META_FILE  = MODEL_DIR / "meta.json"
SAVED_MODEL= MODEL_DIR / "saved_model"
LOG_FILE   = Path("fire_detections.log")

IS_PI      = platform.machine().startswith("aarch") or \
             "raspberrypi" in platform.node().lower()
IS_WINDOWS = platform.system() == "Windows"

SAMPLE_RATE    = 16000       # YAMNet requires 16 kHz
CLIP_SECONDS   = 10
ALERT_THRESHOLD = 0.25       # YAMNet fire-class mean score to trigger FIRE
ALARM_THRESHOLD = 0.30       # alarm-class threshold

# ── Dependency check ──────────────────────────────────────────────────────────
def check_deps():
    missing = []
    for pkg, imp in [("tensorflow","tensorflow"), ("numpy","numpy"),
                     ("sounddevice","sounddevice"), ("blessed","blessed")]:
        try: __import__(imp)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"[!] pip install {' '.join(missing)}")
        sys.exit(1)

check_deps()

import numpy as np
import sounddevice as sd
import tensorflow as tf
import blessed


# ─────────────────────────────────────────────────────────────────────────────
# YAMNet wrapper
# ─────────────────────────────────────────────────────────────────────────────
class YAMNetFireDetector:
    """
    Loads saved YAMNet locally and maps its 521-class output to
    three fire-relevant states: SAFE / FIRE / SMOKE ALARM.
    """

    def __init__(self):
        if not SAVED_MODEL.exists():
            print("[!] Model not found. Run:  python download_model.py")
            sys.exit(1)

        print("[*] Loading YAMNet from local cache ...", end=" ", flush=True)
        # Suppress TF info/warning logs
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.get_logger().setLevel("ERROR")

        self.model = tf.saved_model.load(str(SAVED_MODEL))

        with open(META_FILE) as f:
            self.meta = json.load(f)
        with open(MODEL_DIR / "class_names.json") as f:
            self.class_names = json.load(f)

        self.fire_idx  = set(self.meta["fire_indices"])
        self.alarm_idx = set(self.meta["alarm_indices"])
        print("ready ✓")
        print(f"    Fire classes  : {self.meta['fire_classes']}")
        print(f"    Alarm classes : {self.meta['alarm_classes']}")

    def predict(self, waveform: np.ndarray):
        """
        waveform : float32 numpy array, 16 kHz mono, range [-1, 1]
        Returns  : (state, confidence, details_dict)
                   state ∈ {"SAFE", "FIRE", "SMOKE ALARM"}
        """
        # YAMNet processes in 0.96s frames; scores shape = (n_frames, 521)
        scores, embeddings, spectrogram = self.model(waveform)
        scores_np = scores.numpy()                    # (n_frames, 521)

        # Mean score per class across all frames
        mean_scores = scores_np.mean(axis=0)          # (521,)

        # Aggregate fire and alarm scores
        fire_score  = float(mean_scores[list(self.fire_idx)].max())
        alarm_score = float(mean_scores[list(self.alarm_idx)].max())

        # Top-5 classes for display
        top5_idx    = mean_scores.argsort()[-5:][::-1]
        top5        = [(self.class_names[i], float(mean_scores[i])) for i in top5_idx]

        details = {
            "fire_score":  fire_score,
            "alarm_score": alarm_score,
            "top5":        top5,
            "n_frames":    len(scores_np),
        }

        # Decision: alarm takes priority
        if alarm_score >= ALARM_THRESHOLD:
            state = "SMOKE ALARM"
            conf  = alarm_score
        elif fire_score >= ALERT_THRESHOLD:
            state = "FIRE"
            conf  = fire_score
        else:
            # Confidence = 1 - max(fire, alarm) so high = very safe
            state = "SAFE"
            conf  = 1.0 - max(fire_score, alarm_score)

        return state, conf, details


# ─────────────────────────────────────────────────────────────────────────────
# Audio recorder
# ─────────────────────────────────────────────────────────────────────────────
def record_clip(duration=CLIP_SECONDS, sr=SAMPLE_RATE, progress_cb=None):
    """Record from default mic. Returns float32 numpy array normalised to [-1,1]."""
    frames = int(sr * duration)
    audio  = sd.rec(frames, samplerate=sr, channels=1, dtype='float32')
    for i in range(duration):
        time.sleep(1)
        if progress_cb:
            progress_cb(i + 1, duration)
    sd.wait()
    wav = audio.flatten()
    peak = np.abs(wav).max()
    if peak > 1e-6:
        wav = wav / peak
    return wav


# ─────────────────────────────────────────────────────────────────────────────
# Live waveform monitor
# ─────────────────────────────────────────────────────────────────────────────
def waveform_monitor(rms_buffer: collections.deque, stop_event: threading.Event):
    BLOCK = 1024
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                             blocksize=BLOCK, dtype='float32') as stream:
            while not stop_event.is_set():
                data, _ = stream.read(BLOCK)
                rms = float(np.sqrt((data ** 2).mean()))
                rms_buffer.append(rms)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Terminal Exhibition Display
# ─────────────────────────────────────────────────────────────────────────────
class FireDisplay:
    """
    Full-screen terminal UI designed for a science exhibition table.

    Layout:
    ┌────────────────────────────────────────────────────┐
    │              🔥 FIREGUARD AUDIO AI 🔥              │
    ├─────────────────────────┬──────────────────────────┤
    │   BIG STATE BLOCK       │  Live waveform           │
    │   (SAFE/FIRE/ALARM)     │  Top-5 YAMNet classes    │
    │                         │  Detection history       │
    │   Probability bar       │                          │
    ├─────────────────────────┴──────────────────────────┤
    │  HUD: platform · model · time                      │
    │  Controls: TAB=record  Q=quit  H=hud  S=save       │
    └────────────────────────────────────────────────────┘
    """

    ANIM = ["◐", "◓", "◑", "◒"]

    FIRE_ART = [
        "    ) )  ( (   ) (   ",
        "   ( (   ) )  ( ) )  ",
        "  _.,-'\"\"\"\"\"\"'-.,_   ",
        " /   🔥 FIRE 🔥   \\  ",
        "|   DETECTED !!!   | ",
        " \\_________________/ ",
    ]
    SAFE_ART = [
        "  .───────────────.  ",
        " /   ✅  A L L    \\  ",
        "|     C L E A R    | ",
        "|   No fire sound  | ",
        " \\  detected ✓    /  ",
        "  `───────────────'  ",
    ]
    ALARM_ART = [
        " *** 🚨 WARNING 🚨 *** ",
        "  SMOKE ALARM TONE   ",
        " *** * *** * *** * ** ",
        "|  !! EVACUATE !!   | ",
        "|  Check for smoke  | ",
        " ******************  ",
    ]
    IDLE_ART = [
        "  ╔═══════════════╗  ",
        "  ║  🔥 FIREGUARD ║  ",
        "  ║   AUDIO  AI   ║  ",
        "  ╠═══════════════╣  ",
        "  ║  Press  TAB   ║  ",
        "  ║  to  analyse  ║  ",
    ]

    def __init__(self):
        self.term       = blessed.Terminal()
        self.show_hud   = True
        self.state      = "idle"
        self.result     = None            # (state, conf, details)
        self.rec_elapsed= 0
        self.history    = []              # [(datetime, state, conf)]
        self.rms_buf    = collections.deque([0.0] * 70, maxlen=70)
        self._frame     = 0
        self._running   = True
        self._blink     = True

        t = threading.Thread(target=self._tick, daemon=True)
        t.start()

    def _tick(self):
        while self._running:
            time.sleep(0.4)
            self._frame = (self._frame + 1) % 4
            self._blink = not self._blink

    def stop(self):
        self._running = False

    # ── colour helpers ────────────────────────────────────────────────────────
    def _state_style(self, state=None):
        t = self.term
        s = state or (self.result[0] if self.result else "idle")
        if s == "FIRE":        return t.bold + t.red
        if s == "SMOKE ALARM": return t.bold + t.yellow
        if s == "SAFE":        return t.bold + t.green
        if s == "recording":   return t.bold + t.cyan
        if s == "analysing":   return t.bold + t.magenta
        return t.white

    def _border_style(self):
        if self.state == "result" and self.result:
            return self._state_style(self.result[0])
        if self.state == "recording": return self.term.cyan
        if self.state == "analysing": return self.term.magenta
        return self.term.white + self.term.dim

    # ── primitives ────────────────────────────────────────────────────────────
    def _hline(self, y, char="─"):
        t = self.term
        print(t.move(y, 0) + self._border_style() + char * t.width + t.normal)

    def _centre(self, text, y, style=""):
        t = self.term
        x = max(0, (t.width - len(text)) // 2)
        print(t.move(y, x) + style + text + t.normal)

    def _bar(self, fraction, width=40, fg="█", bg="░"):
        n = max(0, min(width, int(fraction * width)))
        return fg * n + bg * (width - n)

    # ── waveform ──────────────────────────────────────────────────────────────
    def _draw_waveform(self, y, x, width):
        t     = self.term
        vals  = list(self.rms_buf)[-width:]
        chars = " ▁▂▃▄▅▆▇█"
        sty   = t.cyan if self.state == "recording" else (t.white + t.dim)
        line  = ""
        for v in vals:
            v    = max(0.0, min(1.0, v * 8))
            line += chars[int(v * (len(chars) - 1))]
        print(t.move(y, x) + sty + line[:width] + t.normal)

    # ── ASCII art ─────────────────────────────────────────────────────────────
    def _draw_art(self, y, x):
        t = self.term
        if self.state == "result" and self.result:
            s = self.result[0]
            art = self.FIRE_ART if s == "FIRE" else \
                  self.ALARM_ART if s == "SMOKE ALARM" else self.SAFE_ART
            sty = self._state_style(s)
        elif self.state == "recording":
            art = self.IDLE_ART; sty = t.cyan
        elif self.state == "analysing":
            art = self.IDLE_ART; sty = t.magenta
        else:
            art = self.IDLE_ART; sty = t.white + t.dim

        for i, line in enumerate(art):
            print(t.move(y + i, x) + sty + line + t.normal)

    # ── top-5 classes ─────────────────────────────────────────────────────────
    def _draw_top5(self, y, x):
        t = self.term
        print(t.move(y, x) + t.bold + " TOP YAMNET CLASSES" + t.normal)
        if not self.result:
            print(t.move(y + 1, x) + t.dim + " — awaiting scan —" + t.normal)
            return
        top5 = self.result[2].get("top5", [])
        for i, (name, score) in enumerate(top5):
            bar  = self._bar(score, width=18)
            pct  = f"{score*100:5.1f}%"
            # highlight fire-relevant names
            fire_related = any(w in name for w in
                               ["Fire","Crack","Smoke","Alarm","Hiss","Buzz","Beep"])
            nsty = (t.red + t.bold) if fire_related else t.white
            line = f" {name[:18]:<18} {pct}"
            print(t.move(y + 1 + i, x) + nsty + line + t.normal)
            print(t.move(y + 1 + i, x + 27) + t.cyan + bar + t.normal)

    # ── history ───────────────────────────────────────────────────────────────
    def _draw_history(self, y, x):
        t = self.term
        print(t.move(y, x) + t.bold + " RECENT SCANS" + t.normal)
        for i, (ts, state, conf) in enumerate(self.history[-5:]):
            sty  = self._state_style(state)
            line = f" {ts.strftime('%H:%M:%S')}  {state:<12} {conf*100:.0f}%"
            print(t.move(y + 1 + i, x) + sty + line[:35] + t.normal)

    # ── fire/alarm score bars ─────────────────────────────────────────────────
    def _draw_score_bars(self, y, x):
        t = self.term
        if not self.result:
            return
        det    = self.result[2]
        fscore = det.get("fire_score",  0.0)
        ascore = det.get("alarm_score", 0.0)
        print(t.move(y,   x) + t.red    + f" 🔥 Fire score  {fscore:.3f}  "
              + self._bar(fscore, 20) + t.normal)
        print(t.move(y+1, x) + t.yellow + f" 🚨 Alarm score {ascore:.3f}  "
              + self._bar(ascore, 20) + t.normal)
        print(t.move(y+2, x) + t.dim
              + f"    Thresholds  fire≥{ALERT_THRESHOLD:.2f}  alarm≥{ALARM_THRESHOLD:.2f}"
              + t.normal)

    # ── main render ───────────────────────────────────────────────────────────
    def render(self):
        t = self.term
        w = t.width
        h = t.height

        print(t.home + t.clear, end="")

        # Title
        title = "🔥  F I R E G U A R D   ·   A U D I O   A I   ·   Y A M N e t  🔥"
        self._centre(title, 0, t.bold + t.red)
        self._centre("Powered by Google YAMNet · 521 AudioSet classes · Fully Offline", 1,
                     t.dim)
        self._hline(2)

        # Status line
        if self.state == "idle":
            status = "READY  ·  Press  TAB  to record a 10-second audio clip"
            ssty   = t.white
        elif self.state == "recording":
            a = self.ANIM[self._frame]
            status = f"{a}  RECORDING  {self.rec_elapsed} / {CLIP_SECONDS} s  — speak or make sound near the mic"
            ssty   = t.cyan + t.bold
        elif self.state == "analysing":
            a = self.ANIM[self._frame]
            status = f"{a}  ANALYSING  — YAMNet processing {CLIP_SECONDS}s of audio ..."
            ssty   = t.magenta + t.bold
        else:
            state, conf, det = self.result
            frames = det.get("n_frames", 0)
            status = (f"{'🔥' if state=='FIRE' else '🚨' if state=='SMOKE ALARM' else '✅'}"
                      f"  {state}  ·  score {conf:.3f}  ·  {frames} YAMNet frames analysed")
            ssty   = self._state_style(state)

        self._centre(status, 4, ssty)

        # Recording progress bar
        if self.state == "recording":
            bw   = min(56, w - 4)
            bx   = (w - bw) // 2
            bar  = self._bar(self.rec_elapsed / CLIP_SECONDS, bw, "▓", "░")
            print(t.move(5, bx) + t.cyan + bar + t.normal)

        self._hline(6)

        # Left column: art + score bars
        art_x = max(2, w // 2 - 30)
        art_y = 8
        self._draw_art(art_y, art_x)

        score_y = art_y + 7
        self._draw_score_bars(score_y, art_x)

        # Divider
        div_x = w // 2 + 2
        for row in range(7, h - 6):
            print(t.move(row, div_x - 1) + t.dim + "│" + t.normal)

        # Right column: waveform + top5 + history
        rx = div_x + 1
        print(t.move(7,  rx) + t.dim + "LIVE WAVEFORM" + t.normal)
        self._draw_waveform(8, rx, w - rx - 1)
        print(t.move(9,  rx) + t.dim + "─" * min(50, w - rx - 1) + t.normal)

        self._draw_top5(10, rx)
        self._draw_history(17, rx)

        # HUD
        if self.show_hud:
            self._hline(h - 5)
            now  = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
            info = (f" YAMNet (MobileNetV1) · {SAMPLE_RATE} Hz · {CLIP_SECONDS}s clips"
                    f" · {platform.system()} {platform.machine()} · {now}")
            print(t.move(h - 4, 0) + t.dim + info[:w] + t.normal)

        self._hline(h - 3)
        ctrl = "  TAB: Record    Q: Quit    H: HUD    S: Save log    C: Clear"
        print(t.move(h - 2, 0) + t.bold + ctrl[:w] + t.normal)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    detector = YAMNetFireDetector()
    display  = FireDisplay()
    t        = display.term

    # Start waveform monitor
    stop_wave = threading.Event()
    wave_t    = threading.Thread(
        target=waveform_monitor,
        args=(display.rms_buf, stop_wave),
        daemon=True,
    )
    wave_t.start()

    lock = threading.Lock()
    busy = False

    def do_scan():
        nonlocal busy
        if busy:
            return
        busy = True

        try:
            # Record
            def on_progress(elapsed, total):
                display.rec_elapsed = elapsed
                display.state       = "recording"
                with lock: display.render()

            waveform = record_clip(progress_cb=on_progress)

            # Analyse
            display.state = "analysing"
            with lock: display.render()

            state, conf, details = detector.predict(waveform)

            # Show result
            display.result = (state, conf, details)
            display.state  = "result"
            display.history.append((datetime.datetime.now(), state, conf))
            with lock: display.render()

        finally:
            busy = False

    def save_log():
        if not display.history:
            return
        with open(LOG_FILE, "a") as f:
            ts, state, conf = display.history[-1]
            f.write(f"{ts.isoformat()}  {state}  {conf:.3f}\n")

    # Full-screen loop
    print(t.enter_fullscreen() + t.hide_cursor(), end="", flush=True)
    display.render()

    try:
        with t.cbreak():
            while True:
                key = t.inkey(timeout=0.3)

                if not key:
                    with lock: display.render()
                    continue

                if key.name == "KEY_TAB" or str(key) == "\t":
                    if not busy:
                        threading.Thread(target=do_scan, daemon=True).start()

                elif str(key).lower() in ("q",) or key.name == "KEY_ESCAPE":
                    break

                elif str(key).lower() == "h":
                    display.show_hud = not display.show_hud

                elif str(key).lower() == "s":
                    save_log()

                elif str(key).lower() == "c":
                    display.state  = "idle"
                    display.result = None

                with lock: display.render()

    except KeyboardInterrupt:
        pass
    finally:
        stop_wave.set()
        display.stop()
        print(t.exit_fullscreen() + t.normal_cursor(), end="", flush=True)
        saved = f"  Log saved → {LOG_FILE}" if LOG_FILE.exists() else ""
        print(f"FireGuard closed.{saved}\n")


if __name__ == "__main__":
    main()
