import tkinter as tk
import cv2 as cv
from PIL import Image, ImageTk
from Gui.initGui import InitGui as Init


class ChooseAnchor(Init):
    def __init__(self):
        super().__init__()

    @classmethod
    def select_anchor_name(cls, face, anchors, image_to_embedding):
        image = cv.cvtColor(face, cv.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image.thumbnail((300, 300))
        image = ImageTk.PhotoImage(image)
        choose_name = tk.Toplevel(ChooseAnchor.root)
        choose_name.title("Choose name")
        image_label = tk.Label(choose_name, image=image)
        image_label.pack(pady=10)
        name_label = tk.Label(choose_name, text="Enter name:")
        name_label.pack(pady=10)
        name_entry = tk.Entry(choose_name)
        name_entry.pack(pady=10)
        submit_btn = tk.Button(choose_name, text="Submit", command=lambda: cls.choose_anchor(
            name_entry.get(), face, anchors, image_to_embedding, choose_name))
        submit_btn.pack(pady=10)
        choose_name.mainloop()

    @classmethod
    def choose_anchor(cls, name, face, anchors, image_to_embedding, choose_name):
        anchors[name] = [image_to_embedding(face), {"60": [], "70": [], "80": [], "90": []}, face]
        choose_name.destroy()
        # stop the mainloop
        choose_name.quit()
