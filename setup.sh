#!/bin/bash
# Quick setup script for Data Redaction App
# Run this to install all dependencies and verify configuration

echo "=================================================="
echo "Data Redaction App - Quick Setup"
echo "=================================================="

# Step 1: Create/activate virtual environment
echo ""
echo "[1/4] Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    source venv/Scripts/activate  # Git Bash on Windows
else
    source venv/bin/activate  # Linux/Mac
fi

# Step 2: Install dependencies
echo ""
echo "[2/4] Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Step 3: Install additional optional dependencies
echo ""
echo "[3/4] Installing optional OCR engines..."
pip install pytesseract
echo "ℹ️  Note: For Tesseract OCR, install from: https://github.com/UB-Mannheim/tesseract/wiki"
echo "✅ Optional dependencies installed"

# Step 4: Check configuration
echo ""
echo "[4/4] Verifying configuration..."

if [ -f ".env" ]; then
    echo "✅ .env file found"
    if grep -q "GOOGLE_API_KEY" .env; then
        echo "✅ GOOGLE_API_KEY configured"
    else
        echo "⚠️  GOOGLE_API_KEY not found in .env"
        echo "   Add your Gemini API key to .env file"
    fi
else
    echo "⚠️  .env file not found"
    echo "   Creating .env template..."
    cat > .env << EOF
# Gemini API Configuration
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional: Multiple API keys for rotation
# GOOGLE_API_KEY_1=key1
# GOOGLE_API_KEY_2=key2
# GOOGLE_API_KEY_3=key3

# Blockchain Configuration
GANACHE_URL=http://127.0.0.1:7545
PRIVATE_KEY=your_private_key_here
ACCOUNT_ADDRESS=your_account_address_here
EOF
    echo "✅ .env template created - please fill in your API key"
fi

if [ -f "requirements.txt" ]; then
    echo "✅ requirements.txt found"
fi

# Summary
echo ""
echo "=================================================="
echo "✅ SETUP COMPLETE"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Add your Gemini API key to .env file"
echo "2. Start the app: streamlit run Hackerearth_app.py"
echo "3. Upload an ID document to test the extraction"
echo ""
echo "For detailed information, see FIX_SUMMARY.md"
echo "=================================================="
