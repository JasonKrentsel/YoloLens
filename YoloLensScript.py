import torch
from tkinter import Tk, Canvas
from PIL import Image
import win32gui
import win32ui
from ctypes import windll
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Load YOLOv5 model and set up CUDA if available
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable CUDA optimizations if available
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def list_windows():
    """List all visible windows and return their handles and titles."""
    def callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            windows.append((hwnd, win32gui.GetWindowText(hwnd)))
    windows = []
    win32gui.EnumWindows(callback, windows)
    return windows

def capture_window(hwnd, save=False):
    """Capture a screenshot of the specified window."""
    # Get window dimensions
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width, height = right - left, bottom - top

    # Create device contexts and bitmap
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)

    # Capture the window content
    result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)
    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)

    # Convert bitmap to PIL Image
    im = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'BGRX', 0, 1)
    
    # Crop the image
    crop_amount = 0
    crop_box = (crop_amount, crop_amount, im.width - crop_amount, im.height - crop_amount)
    im = im.crop(crop_box)
    
    # Save the image if requested
    if save:
        im.save("screenshot.png")

    # Clean up resources
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    if result == 0:
        print("Failed to capture the window. Error code:", windll.kernel32.GetLastError())
    else:
        return im.convert("RGB")

def window_choice():
    """Prompt user to choose a window for processing."""
    windows = list_windows()
    for i, (hwnd, title) in enumerate(windows):
        print(f"{i}: {title}")

    choice = int(input("Enter the number of the window you want to process: "))
    return windows[choice][0]

def detect_humans(image, confidence_threshold=0.5):
    """Detect humans in the given image using YOLOv5."""
    results = model(image)
    detections = results.pandas().xyxy[0]
    return detections[(detections['class'] == 0) & (detections['confidence'] >= confidence_threshold)]

def create_overlay(hwnd, offset_x=0, offset_y=0):
    """Create a transparent overlay window for drawing bounding boxes."""
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width, height = right - left + offset_x, bottom - top + offset_y

    root = Tk()
    root.attributes('-topmost', True)
    root.attributes('-transparentcolor', 'black')
    root.geometry(f'{width}x{height}+{left}+{top}')
    canvas = Canvas(root, width=width, height=height, bg='black', highlightthickness=0)
    canvas.pack()
    return root, canvas

def draw_boxes(canvas, detections, offset_x=0, offset_y=0):
    """Draw bounding boxes and confidence scores on the overlay canvas."""
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        # Apply the offset to the coordinates
        x1, y1, x2, y2 = x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y
        canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)
        # Draw confidence level next to the box
        confidence_text = f"{confidence:.2f}"
        canvas.create_text(x2 + 5, y1, text=confidence_text, fill='red', anchor='nw')

def main():
    """Main function to run the human detection overlay."""
    selected_window = window_choice()
    
    if selected_window is None:
        return
    
    root, canvas = create_overlay(selected_window, offset_x=-15, offset_y=-40)

    fps_target = 30
    last_update_time = 0
    update_interval = 1 / fps_target
    frame_times = []
    total_time_start = time.time()

    while True:
        current_time = time.time()
        if current_time - last_update_time >= update_interval:
            # Capture window content
            start_capture = time.time()
            image = capture_window(selected_window)
            capture_time = time.time() - start_capture

            # Detect humans in the image
            start_detection = time.time()
            detections = detect_humans(image, confidence_threshold=0.5)
            detection_time = time.time() - start_detection

            # Update overlay with bounding boxes
            start_render = time.time()
            canvas.delete("all")  # Clear previous boxes
            draw_boxes(canvas, detections, offset_x=-10, offset_y=-35)
            root.update_idletasks()
            root.update()
            render_time = time.time() - start_render

            # Calculate and store frame times
            total_time = capture_time + detection_time + render_time
            frame_times.append((capture_time, detection_time, render_time, total_time))
            if len(frame_times) > fps_target:
                frame_times.pop(0)

            # Print performance statistics
            if len(frame_times) == fps_target:
                avg_capture = sum(t[0] for t in frame_times) / fps_target
                avg_detection = sum(t[1] for t in frame_times) / fps_target
                avg_render = sum(t[2] for t in frame_times) / fps_target
                avg_total = sum(t[3] for t in frame_times) / fps_target
                total_time_with_rests = time.time() - total_time_start
                avg_fps = fps_target / total_time_with_rests
                print(f"Avg Capture: {avg_capture:.4f}s, Avg Detection: {avg_detection:.4f}s, Avg Render: {avg_render:.4f}s, Avg Total: {avg_total:.4f}s, Total Time (with rests): {total_time_with_rests:.4f}s, Avg FPS: {avg_fps:.2f}")
                frame_times.clear()
                total_time_start = time.time()

            last_update_time = current_time
        else:
            time.sleep(0.01)

if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    main()