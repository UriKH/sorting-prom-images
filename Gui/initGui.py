import queue
import tkinter as tk
from initializer import Init


class InitGui(Init):
    root = None
    progress_queue = None

    def __init__(self):
        super().__init__()
        if not InitGui.root:
            InitGui.root = tk.Tk()
        if not InitGui.progress_queue:
            InitGui.progress_queue = queue.Queue()
