#!/bin/bash

set -e  # Exit on error
set -u  # Error on unset variables

PROJECT_DIR="$(pwd)"
VENV_DIR=".venv"

echo "🔧 Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo "📦 Installing system dependencies..."
sudo apt install -y \
  python3-venv \
  python3-dev \
  python3-pip \
  libcap-dev \
  libcamera-dev \
  build-essential \
  python3-rpi.gpio \
  python3-picamera2 \
  libopencv-dev \
  curl \
  git

echo "🐍 Creating virtual environment in $VENV_DIR ..."
python3 -m venv $VENV_DIR

echo "📥 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "🔧 Upgrading pip and installing pip-tools..."
pip install --upgrade pip
pip install pip-tools

echo "✅ Setup complete!"
echo "To activate your environment, run:"
echo "   source $PROJECT_DIR/$VENV_DIR/bin/activate"
