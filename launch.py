import subprocess
import webbrowser
import time
import sys
import os

print("🚀 Starting Backend...")

# Run Flask WITHOUT reload issue
process = subprocess.Popen(
    [sys.executable, "app.py"],
    creationflags=subprocess.CREATE_NEW_CONSOLE
)

# Wait for server to start
time.sleep(5)

print("🌐 Opening Frontend...")

webbrowser.open("http://127.0.0.1:5502/pages/signlive.html")