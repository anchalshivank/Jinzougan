import tkinter as tk
from tkinter import ttk
from camera_app2 import CameraApp
from manual_classifier import ManualClassifier

def switch_to_detection(root):
    """Switch to defect detection view."""
    for widget in root.winfo_children():
        widget.destroy()
    CameraApp(root, back_to_main=lambda: main_menu(root))


def switch_to_manual_classification(root):
    """Switch to manual classification view."""
    for widget in root.winfo_children():
        widget.destroy()
    ManualClassifier(root, back_to_main=lambda: main_menu(root))


def main_menu(root=None):
    """Main menu for the defect detection system."""
    if root is None:
        root = tk.Tk()
        root.title("Defect Detection System")
        root.geometry("800x600")
    else:
        for widget in root.winfo_children():
            widget.destroy()

    # Main menu content
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    title_label = tk.Label(main_frame, text="Main Menu", font=("Arial", 24))
    title_label.pack(pady=20)

    btn_manual = tk.Button(
        main_frame,
        text="Manual Classification",
        font=("Arial", 18),
        command=lambda: switch_to_manual_classification(root),
    )
    btn_manual.pack(pady=10)

    btn_start_detection = tk.Button(
        main_frame,
        text="Start Defect Detection",
        font=("Arial", 18),
        command=lambda: switch_to_detection(root),
    )
    btn_start_detection.pack(pady=10)

    btn_exit = tk.Button(main_frame, text="Exit", font=("Arial", 18), command=root.destroy)
    btn_exit.pack(pady=10)

    if not isinstance(root, tk.Tk):
        return

    root.mainloop()
