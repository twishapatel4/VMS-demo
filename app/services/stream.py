"""Threaded video capture wrapper.

Decouples camera read I/O from the main inference loop so HTTP / RTSP latency
doesn't stall frame processing.
"""

import threading
import cv2


class VideoStream:
    def __init__(self, src=0, label="cam", width=1280, height=720):
        self.label = label
        self.src = src
        self.cap = cv2.VideoCapture(src)
        # Resolution hints only meaningful for local devices; remote streams ignore.
        if isinstance(src, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.opened = self.cap.isOpened()
        # Skip blocking initial read; background thread (started in .start()) will populate self.frame.
        # Main loop already handles `if frame is None: continue`.
        self.ret, self.frame = (False, None)
        self.stopped = False

    def start(self):
        if not self.opened:
            print(f"[WARN] {self.label}: failed to open source {self.src}")
            return self
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self.opened:
            self.cap.release()
