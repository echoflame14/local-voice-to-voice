#!/bin/bash
# Elegant Chatbot Installation Script

echo "================================"
echo "🎨 Elegant Chatbot Installer"
echo "================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "📍 Python version: $python_version"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Platform-specific audio setup
echo ""
echo "🔊 Setting up audio..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "   Linux detected - installing portaudio"
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "   macOS detected - installing portaudio"
    brew install portaudio
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    echo "   Windows detected - using pipwin for pyaudio"
    pip install pipwin
    pipwin install pyaudio
fi

# Try to install Chatterbox TTS
echo ""
echo "🎵 Installing Chatterbox TTS (optional)..."
pip install git+https://github.com/resemble-ai/chatterbox.git || echo "   ⚠️  Chatterbox installation failed, will use fallback TTS"

# Create directories
echo ""
echo "📁 Creating directories..."
mkdir -p data logs memories voices

# Check for API key
echo ""
echo "🔑 Checking API key..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "   ⚠️  OPENAI_API_KEY not set!"
    echo "   Please set it with: export OPENAI_API_KEY='your-key-here'"
else
    echo "   ✅ OpenAI API key found"
fi

# Run tests
echo ""
echo "🧪 Running basic tests..."
python test_basic.py

echo ""
echo "================================"
echo "✅ Installation complete!"
echo ""
echo "To run the chatbot:"
echo "  1. Activate venv: source venv/bin/activate"
echo "  2. Set API key: export OPENAI_API_KEY='your-key'"
echo "  3. Run: python main.py"
echo "================================"