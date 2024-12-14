import tkinter as tk
import shutil
import os
from collections import deque
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from datetime import datetime


class ImageCache:
    """A simple LRU cache for storing loaded images."""

    def __init__(self, capacity=10):
        self.cache = {}
        self.eviction_queue = deque()
        self.capacity = capacity

    def get(self, index):
        """Retrieve an image from the cache."""
        return self.cache.get(index)

    def load(self, index, path):
        """Load an image into the cache if it's not already present."""
        if index in self.cache:
            return self.cache[index]

        # Load the image
        try:
            image = Image.open(path)
            image.thumbnail((200, 150))  # Resize for display
        except FileNotFoundError:
            print(f"File not found: {path}")
            return None

        # Add to cache
        self.cache[index] = image
        self.eviction_queue.append(index)

        # Evict if over capacity
        if len(self.eviction_queue) > self.capacity:
            evicted_index = self.eviction_queue.popleft()
            self.cache.pop(evicted_index, None)

        return image


class ManualClassifier:
    def __init__(self, parent, back_to_main):
        self.parent = parent
        self.folder_path = ""
        self.image_files = []
        self.current_index = 0
        self.back_to_main_callback = back_to_main

        # Image counts
        self.good_count = 0
        self.bad_count = 0

        # Temporary reclassification state
        self.reclassification = {}

        # Image cache
        self.cache = ImageCache(capacity=10)

        # Set up the UI
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI for manual classification."""
        self.header_frame = tk.Frame(self.parent)
        self.header_frame.pack(pady=10)

        self.good_count_label = tk.Label(self.header_frame, text="Good: 0")
        self.good_count_label.pack(side=tk.LEFT, padx=10)

        self.bad_count_label = tk.Label(self.header_frame, text="Bad: 0")
        self.bad_count_label.pack(side=tk.LEFT, padx=10)

        self.progress_label = tk.Label(self.header_frame, text="Viewing 0-0 of 0")
        self.progress_label.pack(side=tk.LEFT, padx=10)

        self.instruction_label = tk.Label(
            self.parent, text="Open a folder containing images for classification."
        )
        self.instruction_label.pack(pady=10)

        self.open_button = tk.Button(self.parent, text="Open Folder", command=self.open_folder)
        self.open_button.pack(pady=10)

        self.image_frame = tk.Frame(self.parent)
        self.image_frame.pack(pady=10, padx=10)

        self.navigation_frame = tk.Frame(self.parent)
        self.navigation_frame.pack(pady=10)

        self.prev_button = tk.Button(self.navigation_frame, text="Previous", command=lambda: self.navigate(-1))
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(self.navigation_frame, text="Next", command=lambda: self.navigate(1))
        self.next_button.pack(side=tk.LEFT, padx=5)

        self.back_button = tk.Button(self.navigation_frame, text="Back", command=self.back_to_main)
        self.back_button.pack(side=tk.LEFT, padx=5)

    def open_folder(self):
        """Open a folder and load images."""
        folder_path = filedialog.askdirectory(title="Select Folder for Classification")
        if not folder_path:
            return

        self.folder_path = folder_path

        # Load all images and sort them by their filenames
        self.image_files = sorted(
            [
                os.path.join(self.folder_path, f)
                for f in os.listdir(self.folder_path)
                if os.path.isfile(os.path.join(self.folder_path, f))
                and f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        if not self.image_files:
            messagebox.showinfo("No Images", "No images found in the selected folder.")
            return

        # Count classifications
        self.good_count = len([f for f in self.image_files if os.path.basename(f).endswith("good")])
        self.bad_count = len([f for f in self.image_files if os.path.basename(f).endswith("bad")])
        self.current_index = 0
        self.update_counts()
        self.display_images()

    def update_counts(self):
        """Update counts and progress labels."""
        self.good_count_label.config(text=f"Good: {self.good_count}")
        self.bad_count_label.config(text=f"Bad: {self.bad_count}")

        total_images = len(self.image_files)
        start = self.current_index + 1
        end = min(self.current_index + 6, total_images)
        self.progress_label.config(text=f"Viewing {start}-{end} of {total_images}")

    def display_images(self):
        """Display six images at a time."""
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        
        self.reclassification = {}

        for i in range(6):
            if self.current_index + i >= len(self.image_files):
                break

            image_path = self.image_files[self.current_index + i]
            image_name = os.path.basename(image_path)

            # Check if the image has been reclassified; if not, classify based on its name
            if image_path not in self.reclassification:
                is_good = not image_name.endswith("_bad")
                self.reclassification[image_path] = "good" if is_good else "bad"

            # Get the current classification
            current_class = self.reclassification[image_path]
            initial_status = current_class.capitalize()

            image = self.cache.load(self.current_index + i, image_path)
            if image is None:
                continue

            image_tk = ImageTk.PhotoImage(image)

            frame = tk.Frame(self.image_frame, relief=tk.RAISED, borderwidth=1)
            frame.grid(row=i // 3, column=i % 3, padx=5, pady=5)

            label = tk.Label(frame, image=image_tk)
            label.image = image_tk
            label.pack()

            name_label = tk.Label(frame, text=f"Name: {image_name}", wraplength=180, justify="center")
            name_label.pack()

            status_var = tk.StringVar(value=f"Classified as {initial_status}")
            status_label = tk.Label(frame, textvariable=status_var)
            status_label.pack()

            btn_toggle = tk.Button(
                frame,
                text="Toggle Classification",
                command=lambda path=image_path, var=status_var: self.toggle_classification(path, var),
            )
            btn_toggle.pack()

        self.update_counts()

    def toggle_classification(self, image_path, status_var):
        """Toggle classification for an image and update the dictionary and status."""
        # Get current classification from reclassification dictionary
        current_class = self.reclassification[image_path]

        # Toggle classification
        new_class = "bad" if current_class == "good" else "good"

        # Update the dictionary and status variable
        self.reclassification[image_path] = new_class
        status_var.set(f"Classified as {new_class.capitalize()}")


    def commit_changes(self):
        """Commit reclassifications by renaming files."""
        if not self.folder_path:
            # messagebox.showerror("Error", "Noe folder is currently selected")
            return
        for image_path, target_class in self.reclassification.items():
            current_name = os.path.basename(image_path)
            new_prefix = f"{target_class}_"
            new_name = new_prefix + current_name.lstrip("bad_good_")  # Remove existing prefix
            new_path = os.path.join(self.folder_path, new_name)

            # Rename the file
            if os.path.exists(image_path):
                os.rename(image_path, new_path)
                # Update the file list to reflect the new name
                self.image_files[self.image_files.index(image_path)] = new_path

        # Clear the reclassification dictionary
        self.reclassification.clear()

        # Reload images and update counts
        self.image_files = sorted(
            [
                os.path.join(self.folder_path, f)
                for f in os.listdir(self.folder_path)
                if os.path.isfile(os.path.join(self.folder_path, f))
                and f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        self.good_count = len([f for f in self.image_files if os.path.basename(f).endswith("_good")])
        self.bad_count = len([f for f in self.image_files if os.path.basename(f).endswith("_bad")])
        self.update_counts()




    def reclassify(self, image_path, target_folder, status_var):
        """Temporarily reclassify an image and update the status label."""
        self.reclassification[image_path] = target_folder
        status_var.set(f"Marked as {target_folder.capitalize()}")

    def navigate(self, direction):
        """Navigate to the next or previous set of images."""
        new_index = self.current_index + (6 * direction)
        if new_index < 0 or new_index >= len(self.image_files):
            messagebox.showinfo("Info", "No more images in this direction.")
            return

        # Commit changes before navigating
        self.commit_changes()
        self.current_index = new_index
        self.display_images()

    def back_to_main(self):
        """Go back to the main menu."""
        # Commit changes before exiting
        self.commit_changes()
        self.back_to_main_callback()
