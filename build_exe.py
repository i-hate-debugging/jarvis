import PyInstaller.__main__
import os
import sys

def build_executable():
    """Build the hand detector application into an executable"""
    
    print("🔨 Building Hand Detector executable...")
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # PyInstaller options
    options = [
        'hand_detector.py',  # Main script
        '--onefile',          # Create single executable
        '--windowed',         # Hide console window
        '--name=HandDetector',  # Name of the executable
        '--icon=NONE',        # No icon (you can add one later)
        '--add-data=requirements.txt;.',  # Include requirements
        '--clean',            # Clean temporary files
        '--noconfirm',        # Replace existing executable
    ]
    
    try:
        # Build the executable
        PyInstaller.__main__.run(options)
        
        print("✅ Build completed successfully!")
        print(f"📁 Executable location: {os.path.join(current_dir, 'dist', 'HandDetector.exe')}")
        print("🚀 You can now run the executable without Python!")
        
    except Exception as e:
        print(f"❌ Build failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = build_executable()
    sys.exit(0 if success else 1)
