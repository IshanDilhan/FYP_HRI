# Testing MCN Directly on Windows (No Virtualization / No ROS2 Needed)

If your computer does not support virtualization (preventing WSL2 or Virtual Machines) and you do not have a Jetson board, **you can still run and test the complete real-time MCN pipeline and your custom models directly on Windows!**

You do not need ROS2 or Linux to test the AI pipeline. We have built-in python scripts that run the exact same logic using standard Windows Python and your webcam or test video files.

---

## Method 1: Interactive Test Video Analysis (Fastest)
This runs the MCN pipeline frame-by-frame on any video inside the `testVideos/` folder. It is already fully set up and ready to run on your PC!

1. Open your standard Windows PowerShell or Command Prompt.
2. Ensure your virtual environment is active:
   ```powershell
   .\env\Scripts\activate
   ```
3. Run the interactive script:
   ```powershell
   python select_and_run.py
   ```
4. Enter the number of the video you want to analyze. The system will process it and output the predicted scenario policies (e.g., `GIVE_WAY`, `HOSTILE_CONFRONTATION`, etc.) in real-time.

---

## Method 2: Live Webcam HRI Testing (Simulating a Live Robot)
If you want to simulate a live robot interacting with you in real-time using your computer's built-in webcam or a USB webcam, run the live webcam script.

We have a dedicated script `run_video.py` or we can quickly create/modify a live webcam script to show predictions on your screen in real-time.

### How to run live webcam testing:
1. Ensure your webcam is connected to your Windows PC.
2. Run the video analysis script with `0` as the input (this tells OpenCV to open your default webcam):
   ```powershell
   python run_video.py --video 0
   ```
3. Stand in front of your webcam and interact. The MCN will output its live social intent predictions right on your terminal screen!

---

## Method 3: Cloud ROS2 Testing (ConstructSim / The Construct)
If your university or project strictly requires you to demonstrate ROS2 node execution, you can use a cloud-based platform:

1. Go to **[The Construct (ConstructSim)](https://www.theconstruct.sim/)**.
2. Create a free account. They provide a **complete web-based Ubuntu environment with ROS2 and simulation tools** running entirely in your browser.
3. You can clone your GitHub repository there, launch the ROS2 node (`python3 pipeline/ros2_mcn_node.py`), and it will execute perfectly without using any of your local PC's virtualization hardware!
