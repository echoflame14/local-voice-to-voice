#!/bin/bash
# Voice-to-Voice Setup Script for macOS

echo "Setting up Voice-to-Voice Environment for macOS..." 

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed. Please install Python 3.8 or higher and try again.${NC}"
    echo "You can install Python using Homebrew: brew install python@3.11"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d'.' -f1)
MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [[ $MAJOR_VERSION -lt 3 ]] || [[ $MAJOR_VERSION -eq 3 && $MINOR_VERSION -lt 8 ]]; then
    echo -e "${RED}Error: This project requires Python 3.8 or higher. Your version: $PYTHON_VERSION${NC}"
    echo -e "${YELLOW}Please install a compatible Python version and try again.${NC}"
    exit 1
fi

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}Homebrew is not installed. Installing Homebrew...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install system dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"
brew install portaudio ffmpeg

# Check if LM Studio server is running
echo -e "${YELLOW}Checking LM Studio server...${NC}"
if curl -s http://localhost:1234/v1/models > /dev/null 2>&1; then
    echo -e "${GREEN}LM Studio server is running${NC}"
else
    echo -e "${RED}Error: LM Studio server is not running at http://localhost:1234${NC}"
    echo -e "${YELLOW}Please start LM Studio, load a model, and start the server before continuing.${NC}"
    echo "Press any key to continue once the server is running..."
    read -n 1 -s
fi

# Create and activate virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
python -m pip install --upgrade pip

# Install audio dependencies first
echo -e "${YELLOW}Installing audio dependencies...${NC}"
pip install wheel setuptools

# Install PyAudio (macOS specific)
echo -e "${YELLOW}Installing PyAudio for macOS...${NC}"
pip install pyaudio

# Clone and install Chatterbox
echo -e "${YELLOW}Setting up Chatterbox TTS...${NC}"
if [ -d "chatterbox" ]; then
    rm -rf chatterbox
fi
git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox
pip install -e .
cd ..

# Install PyTorch with MPS (Metal Performance Shaders) support for Apple Silicon
echo -e "${YELLOW}Installing PyTorch with MPS support...${NC}"
if [[ $(uname -m) == 'arm64' ]]; then
    echo -e "${CYAN}Detected Apple Silicon, installing PyTorch with MPS support...${NC}"
    pip install torch torchvision torchaudio
else
    echo -e "${CYAN}Detected Intel Mac, installing PyTorch...${NC}"
    pip install torch torchvision torchaudio
fi

# Install project dependencies
echo -e "${YELLOW}Installing project dependencies...${NC}"
pip install -r requirements.txt

# Create .env file
echo -e "${YELLOW}Creating configuration file...${NC}"
cat > .env << 'EOF'
# LM Studio Configuration
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_API_KEY=not-needed

# Google Gemini API (add your key here)
GEMINI_API_KEY=your-gemini-api-key-here

# Model Configuration
WHISPER_MODEL_SIZE=base
TTS_DEVICE=mps  # Use 'mps' for Apple Silicon, 'cpu' for Intel Macs

# Voice Configuration
VOICE_REFERENCE_PATH=./voices/asmr_full.wav
VOICE_EXAGGERATION=0.7
VOICE_CFG_WEIGHT=0.5
VOICE_TEMPERATURE=0.8

# Audio Settings
SAMPLE_RATE=16000
CHUNK_SIZE=480
VAD_AGGRESSIVENESS=2
INPUT_MODE=vad

# Chat Settings
SYSTEM_PROMPT=You are a helpful voice assistant. Keep your responses concise and natural for speech.
MAX_RESPONSE_TOKENS=150
LLM_TEMPERATURE=1.0
EOF

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p voices models conversation_logs

# Check if we're on Apple Silicon
if [[ $(uname -m) == 'arm64' ]]; then
    echo -e "${CYAN}Note: You're on Apple Silicon. The TTS will use MPS (Metal Performance Shaders) for acceleration.${NC}"
else
    echo -e "${CYAN}Note: You're on an Intel Mac. The TTS will use CPU.${NC}"
fi

# Final check and instructions
echo -e "\n${GREEN}Setup complete!${NC}"
echo -e "\n${CYAN}To start the voice assistant:${NC}"
echo -e "1. Activate the virtual environment: ${NC}source venv/bin/activate"
echo -e "2. Add your Gemini API key to the .env file (edit GEMINI_API_KEY)"
echo -e "3. Ensure LM Studio server is running with a loaded model (or use Gemini)"
echo -e "4. Run: ${NC}python main.py"
echo -e "\n${YELLOW}Common macOS-specific options:${NC}"
echo -e "- For better audio quality, grant microphone permissions when prompted"
echo -e "- Use --device mps for Apple Silicon acceleration (default)"
echo -e "- Use --device cpu for Intel Macs or if MPS has issues"
echo -e "- If you have audio issues, try: python fix_audio_device.py"
echo -e "\n${YELLOW}For troubleshooting and more information, please refer to the README.md${NC}"