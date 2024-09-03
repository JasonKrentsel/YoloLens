import tkinter as tk
from tkinter import ttk
import numpy as np
import time
import torch
from PIL import Image
import mss
import threading
import queue
from collections import deque

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize YOLOv5 model and set up CUDA if available
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Optimize CUDA settings if available
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

class YoloLens:
    def __init__(self):
        # Initialize main application window and lens window
        self.root = tk.Tk()
        self.root.title("YoloLens Control")
        
        self.lens = tk.Toplevel(self.root)
        self.lens.title("YoloLens")
        self.lens.attributes('-topmost', True)
        
        # Set up canvas for drawing detections
        self.canvas = tk.Canvas(self.lens)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Initialize variables for statistics and options
        self.initialize_variables()
        
        # Set up queues and threads for detection and rendering
        self.setup_queues_and_threads()
        
        # Configure UI elements
        self.setup_control_window()
        self.setup_lens_window()
        
        # Bind closing event and start detection loop
        self.lens.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(100, self.run_detection)
        
        self.root.mainloop()
    
    def initialize_variables(self):
        # Statistics variables
        self.lens_size = tk.StringVar(value="0x0")
        self.avg_capture_time = tk.DoubleVar(value=0.0)
        self.avg_detection_time = tk.DoubleVar(value=0.0)
        self.avg_render_time = tk.DoubleVar(value=0.0)
        self.avg_fps = tk.DoubleVar(value=0.0)
        
        # Option variables
        self.x_offset = tk.IntVar(value=0)
        self.y_offset = tk.IntVar(value=0)
        self.detection_running = tk.BooleanVar(value=False)
        
        # Deques for moving average calculations
        self.capture_times = deque(maxlen=30)
        self.detection_times = deque(maxlen=30)
        self.render_times = deque(maxlen=30)
    
    def setup_queues_and_threads(self):
        # Queues for inter-thread communication
        self.detection_queue = queue.Queue(maxsize=1)
        self.render_queue = queue.Queue(maxsize=1)
        
        # Thread placeholders
        self.detection_thread = None
        self.render_thread = None
    
    def setup_control_window(self):
        # Create a frame for settings with padding
        settings_frame = ttk.Frame(self.root, padding="10 10 10 10")
        settings_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Create labels and entries for statistics
        ttk.Label(settings_frame, text="Lens Size:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(settings_frame, textvariable=self.lens_size).grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(settings_frame, text="Avg Capture Time:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(settings_frame, textvariable=self.avg_capture_time).grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(settings_frame, text="Avg Detection Time:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(settings_frame, textvariable=self.avg_detection_time).grid(row=2, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(settings_frame, text="Avg Render Time:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(settings_frame, textvariable=self.avg_render_time).grid(row=3, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(settings_frame, text="Avg FPS:").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(settings_frame, textvariable=self.avg_fps).grid(row=4, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Checkbutton(settings_frame, text="Run Detection", variable=self.detection_running, command=self.toggle_detection).grid(row=5, column=0, columnspan=2, sticky="w", padx=5, pady=5)
    
    def setup_lens_window(self):
        # Set initial size, position, and properties
        self.lens.geometry("400x300+100+100")
        self.lens.wm_attributes("-transparentcolor", "white")
        self.canvas.config(bg="white")
        
        # Make the window resizable
        self.lens.resizable(True, True)
        
        # Add resize grip to bottom-right corner
        ttk.Sizegrip(self.lens).place(relx=1.0, rely=1.0, anchor="se")
        
        # Bind the Configure event to update canvas size
        self.lens.bind("<Configure>", self.on_lens_resize)
    
    def on_lens_resize(self, event):
        # Update canvas size when the window is resized
        self.canvas.config(width=event.width, height=event.height)
        # Update lens size display in control window
        self.lens_size.set(f"{event.width}x{event.height}")
    
    def draw_rectangle(self, x, y, width, height, color="red", thickness=2):
        return self.canvas.create_rectangle(x, y, x+width, y+height, outline=color, width=thickness)
    
    def draw_detections(self, detections):
        # Iterate through each detection
        for _, row in detections.iterrows():
            # Draw a rectangle around the detected object
            self.draw_rectangle(row['xmin'], row['ymin'], row['xmax'] - row['xmin'], row['ymax'] - row['ymin'], color="red")
            
            # Get the confidence score for the detection
            confidence = row['confidence']
            confidence_text = f"{confidence:.2f}"
            
            # Display the confidence score next to the bounding box
            self.canvas.create_text(row['xmax'] + 5, row['ymin'], text=confidence_text, fill='red', anchor='nw', font=('Arial', 14, 'bold'))

    def capture_screen(self):
        # Get the position and dimensions of the lens window
        x, y, w, h = self.lens.winfo_x(), self.lens.winfo_y(), self.lens.winfo_width(), self.lens.winfo_height()
        
        # Calculate the absolute screen coordinates for capture
        capture_x = self.lens.winfo_rootx()
        capture_y = self.lens.winfo_rooty()
        capture_w = w
        capture_h = h
        
        # Use mss to capture the screen area
        with mss.mss() as sct:
            monitor = {"top": capture_y, "left": capture_x, "width": capture_w, "height": capture_h}
            screenshot = sct.grab(monitor)
            # Convert the screenshot to a PIL Image
            screenshot = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
        
        return screenshot

    def detect_humans(self, image, confidence_threshold=0.5):
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        # Perform object detection using YOLOv5
        results = model(image_np)
        # Get the detection results as a pandas DataFrame
        detections = results.pandas().xyxy[0]
        # Filter detections to only include humans (class 0) above the confidence threshold
        return detections[(detections['class'] == 0) & (detections['confidence'] >= confidence_threshold)]

    def update_stats(self, capture_time, detection_time, render_time):
        # Add new timing measurements to the deques
        self.capture_times.append(capture_time)
        self.detection_times.append(detection_time)
        self.render_times.append(render_time)
        
        # Update the average times in the UI
        self.avg_capture_time.set(f"{np.mean(self.capture_times):.3f}")
        self.avg_detection_time.set(f"{np.mean(self.detection_times):.3f}")
        self.avg_render_time.set(f"{np.mean(self.render_times):.3f}")
        
        # Calculate and update the average FPS
        total_time = np.sum(self.capture_times) + np.sum(self.detection_times) + np.sum(self.render_times)
        num_frames = len(self.capture_times)
        self.avg_fps.set(f"{num_frames / total_time:.2f}" if total_time > 0 else "0.00")

    def detection_worker(self):
        # Main loop for the detection thread
        while self.detection_running.get():
            # Capture the screen
            capture_start = time.time()
            frame = self.capture_screen()
            capture_time = time.time() - capture_start

            # Perform human detection
            detection_start = time.time()
            detections = self.detect_humans(frame, confidence_threshold=0.5)
            detection_time = time.time() - detection_start

            # Put the results in the render queue
            self.render_queue.put((detections, capture_time, detection_time))

    def render_worker(self):
        # Main loop for the render thread
        while self.detection_running.get():
            try:
                # Get detection results from the queue
                detections, capture_time, detection_time = self.render_queue.get(timeout=1)
                render_start = time.time()
                
                # Clear previous drawings and draw new detections
                self.canvas.delete("all")
                self.draw_detections(detections)
                
                # Calculate render time and update statistics
                render_time = time.time() - render_start
                self.update_stats(capture_time, detection_time, render_time)
            except queue.Empty:
                # If the queue is empty, continue the loop
                pass

    def run_detection(self):
        # Check if detection is running
        if self.detection_running.get():
            # Start the detection thread if it's not running
            if self.detection_thread is None or not self.detection_thread.is_alive():
                self.detection_thread = threading.Thread(target=self.detection_worker)
                self.detection_thread.daemon = True
                self.detection_thread.start()

            # Start the render thread if it's not running
            if self.render_thread is None or not self.render_thread.is_alive():
                self.render_thread = threading.Thread(target=self.render_worker)
                self.render_thread.daemon = True
                self.render_thread.start()

        # Schedule the next run_detection call
        self.root.after(100, self.run_detection)

    def toggle_detection(self):
        # If detection is turned on, start the detection process
        if self.detection_running.get():
            self.run_detection()
        else:
            # If detection is turned off, clear the queues and canvas
            while not self.detection_queue.empty():
                self.detection_queue.get_nowait()
            while not self.render_queue.empty():
                self.render_queue.get_nowait()
            self.canvas.delete("all")

    def on_closing(self):
        # Stop the detection process
        self.detection_running.set(False)
        # Close the application
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    app = YoloLens()