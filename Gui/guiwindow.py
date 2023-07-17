import os
import queue
import threading
from tkinter import filedialog, messagebox, ttk
from Gui.initGui import InitGui as Init
import tkinter as tk
from sort import Sorter


class GuiWindow(Init):
    def __init__(self):
        super().__init__()
        self.root.title("Image Checker")
        self.root.geometry("400x200")
        self.sort = None
        self.anchor_folder = None
        self.prom_photos_folders = None
        self.progress_bar_list = []
        self.progress_bar_label_list = []
        self.status_label = tk.Label(self.root, text="Select the anchor folder:")
        self.status_label.pack(pady=10)
        self.select_anchor_btn = tk.Button(self.root, text="Select Anchor Folder", command=self.select_anchor_folder)
        self.select_anchor_btn.pack()

        self.start_btn = tk.Button(self.root, text="start sorting", command=self.start_sorting, state=tk.DISABLED)
        self.start_btn.pack(pady=10)
        # show which folder is selected
        self.anchor_folder_label = tk.Label(self.root, text="")
        self.anchor_folder_label.pack()
        self.prom_photos_folder_label = tk.Label(self.root, text="")
        self.prom_photos_folder_label.pack()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def select_anchor_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.anchor_folder = folder
            self.anchor_folder_label.config(text="Anchor folder: " + folder.split("/")[-1])
            self.status_label.config(text="Select the prom photos folder:")
            self.select_anchor_btn.config(text="Select Prom Photos Folder", command=self.prom_photos_folder)

    def prom_photos_folder(self):
        folder = filedialog.askdirectory()
        self.prom_photos_folder_label.config(text="Prom photos folder: " + folder.split("/")[-1])
        if folder:
            self.prom_photos_folders = folder
            self.start_btn.config(state=tk.NORMAL)

    def update_progress_bar(self):
        try:
            index, value = self.progress_queue.get_nowait()
            self.progress_bar_list[index]['value'] = value
            if value == 100:
                self.progress_bar_label_list[index].config(
                    text="Folder " + self.progress_bar_label_list[index].cget("text").split(" ")[1] + " Done!")
            else:
                self.progress_bar_label_list[index].config(
                    text="Folder " + self.progress_bar_label_list[index].cget("text").split(" ")[1] + " " + str(value) + "%")

            self.root.after(10, self.update_progress_bar)
        except queue.Empty:
            self.root.after(10, self.update_progress_bar)

    def start_sorting(self):
        if not self.anchor_folder or not self.prom_photos_folders:
            messagebox.showwarning("Error", "Please select the anchor and other folders first.")
            return
        if not os.listdir(self.anchor_folder):
            messagebox.showwarning("Error", "The anchor folder is empty.")
            return
        if not os.listdir(self.prom_photos_folders):
            messagebox.showwarning("Error", "The prom photos folder is empty.")
            return

        # disable buttons
        self.start_btn.config(state=tk.DISABLED)
        self.select_anchor_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Sorting...")
        anchors = Sorter.initialize_anchors(self.anchor_folder)
        self.sort = Sorter(self.prom_photos_folders, self.anchor_folder, anchors)
        # create progress bars according to the number of folders
        for i, folder in enumerate(os.listdir(self.prom_photos_folders)):
            # create progress bar
            # create progress bar label
            progress_label = tk.Label(self.root, text="Folder " + folder + ":")
            progress_label.pack()
            # create progress bar
            progress_bar = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, length=100, mode='determinate')
            progress_bar.pack(pady=10)
            progress_bar['value'] = 0
            progress_bar['maximum'] = 100
            self.progress_bar_list.append(progress_bar)
            self.progress_bar_label_list.append(progress_label)
        threading.Thread(target=self.sort.handle_sorting).start()
        self.root.after(10, self.update_progress_bar)

    def on_close(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()
