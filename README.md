🥋 Karate Kido 2 Automation Bot

A high-performance, computer vision-based automation bot for the Karate Kido 2 game on the GAMEE platform. This project leverages OpenCV and SIFT to automate gameplay with precision, handling complex mechanics like multi-hit segments and special obstacles.

🚀 Key Features

•
Robust Game Detection: Uses SIFT (Scale-Invariant Feature Transform) to identify the game window, making it resilient to window resizing and different screen resolutions.

•
Multi-Move Batching: Unlike simple bots that process one hit at a time, this algorithm analyzes the entire visible tree structure to calculate multiple moves from a single frame, significantly increasing reaction speed.

•
Advanced Obstacle Handling:

•
Multi-Hit Detection: Identifies segments requiring 2, 3, or 4 hits using custom kernel-based pixel matching.

•
Special Branches: Specialized logic to detect and avoid "Blue Branches" and "Glass" obstacles.

•
Lantern Logic: Pattern-based detection for "Lantern" states to execute timed dodges.



•
Dynamic Calibration: Automatically calculates tree boundaries, colors, and segment heights upon game start.

📂 Project Structure

Plain Text


KarateKidoBot/
├── newproject3.py       # Main automation script
├── images/              # Game UI templates for SIFT detection
├── branches/            # Templates for special branch detection
├── numbers/             # Kernels for 2, 3, and 4 hit detection
└── README.md            # Project documentation



🧠 How It Works

1. Vision & Detection

The bot captures the screen using the mss library. It first uses a template matching approach with SIFT to locate the game window. Once found, it performs a vertical scan using a Hough Line Transform to define the precise boundaries of the tree trunk.

2. The "Multi-Move" Algorithm

Instead of reacting to what's directly in front of the player, the bot divides the tree into 7 vertical segments. For each segment, it:

1.
Masks the Tree: Isolates the trunk based on its calibrated color.

2.
Scans for Branches: Uses horizontal line detection on the left and right sides of each segment.

3.
Identifies Hits: If no branch is found, it scans the trunk for number indicators (2, 3, or 4) to determine how many times to hit.

4.
Queues Moves: All 7 segments are processed at once, and the resulting moves are queued for execution.

3. Execution

The bot uses pyautogui to simulate mouse clicks. It executes the queued moves with micro-delays to stay synchronized with the game's internal animations, allowing it to maintain a high "hits-per-second" ratio.

🛠️ Requirements

•
Python 3.x

•
OpenCV (opencv-python)

•
mss (Screen capture)

•
pyautogui (Input simulation)

•
pygetwindow (Window management)

⚠️ Disclaimer

This project is for educational purposes only. Using automation tools on gaming platforms may violate their terms of service. Use responsibly.

