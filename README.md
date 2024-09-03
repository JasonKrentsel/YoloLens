# YoloLens

YoloLens is a real-time object detection application built using Python and the YOLOv5 model. It utilizes a GUI built with Tkinter to provide an interactive experience for users to capture their screen and detect human figures in real-time.

## Features

-   Real-time human detection using YOLOv5.
-   User-friendly GUI for controlling detection settings.
-   Display of detection statistics, including average capture time, detection time, render time, and frames per second (FPS).
-   Adjustable lens window for screen capture.
-   Multi-threaded architecture for efficient processing.

## Installation

1. Clone the repository:

    ```bash
    git clone <repository-url>
    cd YoloLens
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:

    ```bash
    python YoloLensGUI.py
    ```

## Usage

-   Launch the application, and a control window will appear.
-   Adjust the settings as needed and click the "Run Detection" checkbox to start detecting humans in the screen capture.
-   The lens window will display the detected humans along with their confidence scores.
