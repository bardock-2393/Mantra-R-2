#!/bin/bash

echo "🧹 Starting comprehensive space cleanup..."

# Check current disk usage
echo "📊 Current disk usage:"
df -h

echo ""
echo "🗂️ Current directory sizes:"
du -sh * 2>/dev/null | sort -hr

echo ""
echo "🧹 Cleaning Python caches..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
echo "✅ Python caches cleared"

echo ""
echo "🧹 Cleaning HuggingFace caches..."
if [ -d ~/.cache/huggingface ]; then
    echo "   Clearing HuggingFace cache..."
    rm -rf ~/.cache/huggingface/
    echo "   ✅ HuggingFace cache cleared"
else
    echo "   ⚠️ HuggingFace cache not found"
fi

if [ -d ~/.cache/torch ]; then
    echo "   Clearing PyTorch cache..."
    rm -rf ~/.cache/torch/
    echo "   ✅ PyTorch cache cleared"
else
    echo "   ⚠️ PyTorch cache not found"
fi

if [ -d ~/.cache/transformers ]; then
    echo "   Clearing Transformers cache..."
    rm -rf ~/.cache/transformers/
    echo "   ✅ Transformers cache cleared"
else
    echo "   ⚠️ Transformers cache not found"
fi

echo ""
echo "🧹 Cleaning application files..."
if [ -d "static/uploads" ]; then
    echo "   Clearing uploads directory..."
    rm -rf static/uploads/*
    echo "   ✅ Uploads cleared"
fi

if [ -d "sessions" ]; then
    echo "   Clearing sessions directory..."
    rm -rf sessions/*
    echo "   ✅ Sessions cleared"
fi

echo ""
echo "🧹 Cleaning large video files..."
find . -name "*.mp4" -size +100M -delete 2>/dev/null
find . -name "*.avi" -size +100M -delete 2>/dev/null
find . -name "*.mov" -size +100M -delete 2>/dev/null
find . -name "*.mkv" -size +100M -delete 2>/dev/null
echo "✅ Large video files cleared"

echo ""
echo "🧹 Cleaning model directories..."
if [ -d "models" ]; then
    echo "   Clearing models directory..."
    rm -rf models/*
    echo "   ✅ Models cleared"
fi

if [ -d "vector_embeddings" ]; then
    echo "   Clearing vector embeddings..."
    rm -rf vector_embeddings/*
    echo "   ✅ Vector embeddings cleared"
fi

echo ""
echo "🧹 Cleaning pip cache..."
pip cache purge 2>/dev/null || echo "   ⚠️ pip cache purge failed"

echo ""
echo "🧹 Cleaning system caches..."
sudo apt-get clean 2>/dev/null || echo "   ⚠️ apt-get clean failed"
sudo apt-get autoremove -y 2>/dev/null || echo "   ⚠️ apt-get autoremove failed"

echo ""
echo "🧹 Cleaning temporary files..."
sudo rm -rf /tmp/* 2>/dev/null || echo "   ⚠️ /tmp cleanup failed"
sudo rm -rf /var/tmp/* 2>/dev/null || echo "   ⚠️ /var/tmp cleanup failed"

echo ""
echo "📊 Final disk usage:"
df -h

echo ""
echo "✅ Space cleanup completed!"
echo "💡 If you still need more space, consider:"
echo "   - Removing old log files: sudo find /var/log -type f -name '*.log' -size +100M -delete"
echo "   - Clearing Docker: docker system prune -a --volumes"
echo "   - Removing old packages: sudo apt-get autoremove --purge"
