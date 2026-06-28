#!/bin/bash

# Navigate to script directory
cd "$(dirname "$0")"

echo "========================================================"
echo "Starting Face Recognition Attendance System on Raspberry Pi"
echo "========================================================"

# Find Raspberry Pi's local IP address
LOCAL_IP=$(hostname -I | awk '{print $1}')
if [ -z "$LOCAL_IP" ]; then
    LOCAL_IP="localhost"
fi

echo "--------------------------------------------------------"
echo "🌐 เพื่อให้อาจารย์เข้าดูรายงานเช็คชื่อจากอุปกรณ์อื่น:"
echo "👉 เข้าลิงก์: http://${LOCAL_IP}:5000"
echo "--------------------------------------------------------"

# 1. Start Node.js Web App in the background
echo "1. Starting Web App Backend (Node.js)..."
cd web_app
npm start &
WEB_PID=$!
cd ..

# 2. Wait for the server to spin up
echo "Waiting for web server to start..."
sleep 3

# 3. Open browser (Chromium) to the local app
echo "2. Opening Chromium Browser..."
if command -v chromium-browser &> /dev/null; then
    chromium-browser --no-sandbox http://localhost:5000 &
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:5000 &
fi

# 4. Start Python GUI App using the local venv
echo "3. Starting Python Face Recognition Camera GUI..."
if [ -d "env" ]; then
    source env/bin/activate
    python main.py
else
    python3 main.py
fi

# When python app closes, kill the web server as well
kill $WEB_PID
echo "System closed."
