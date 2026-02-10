# 🤖 Cool Hand Detector 🤖

A real-time hand detection application with cool visual effects that uses your laptop camera to track and visualize hand movements.

## ✨ Features

- **Real-time Hand Detection**: Uses MediaPipe to detect up to 2 hands simultaneously
- **Cool Visual Effects**: 
  - Particle explosions around detected hands
  - Glowing trails that follow hand movements
  - Color-coded landmarks (wrist, fingertips, joints)
  - Animated borders and status indicators
- **Mirror Mode**: Shows a natural mirror view of your movements
- **High Quality**: 720p at 30fps for smooth performance

## 🚀 Quick Start

### Option 1: Run as Python Script

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python hand_detector.py
   ```

### Option 2: Build as Executable

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Build the executable:
   ```bash
   python build_exe.py
   ```

3. Run the executable:
   ```bash
   dist\HandDetector.exe
   ```

## 🎮 Controls

- **Press 'q'**: Quit the application
- **Move your hands**: Watch the cool visual effects follow your movements!

## 📋 Requirements

- Python 3.7+
- A working webcam/camera
- Dependencies listed in `requirements.txt`:
  - OpenCV (for camera capture and display)
  - MediaPipe (for hand detection)
  - NumPy (for numerical operations)

## 🔧 Technical Details

- Uses MediaPipe's hand detection model
- Supports up to 2 hands simultaneously
- 21 landmarks per hand for detailed tracking
- Particle system for visual effects
- Trail system for motion visualization
- Optimized for real-time performance

## 🎨 Visual Effects

- **Particles**: Colorful particle explosions at hand centers
- **Trails**: Fading trails that follow hand movements
- **Glowing Circles**: Multi-colored circles around hand centers
- **Landmarks**: Different colors for wrist (red), fingertips (green), and joints (yellow)
- **Connections**: Highlighted hand skeleton connections
- **Borders**: Animated frame borders

## 🐛 Troubleshooting

**Camera not working?**
- Make sure your camera is connected and not being used by another application
- Check if your camera drivers are up to date
- Try running with administrator privileges

**Performance issues?**
- Close other applications that might be using the camera
- Ensure you have sufficient RAM and CPU resources
- The application is optimized for 720p resolution

**Build issues?**
- Make sure all dependencies are installed
- Try running `pip install --upgrade pyinstaller`
- Ensure you have sufficient disk space for the build

## 📝 License

This project is open source and free to use. Enjoy the cool hand detection effects! 🎉
