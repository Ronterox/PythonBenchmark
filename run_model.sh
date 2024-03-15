if [ $# -gt 1 ]; then
    echo "Usage: $0 <model.pth>"
    exit 0
fi

[ $# -eq 1 ] && MODEL=$1 || MODEL="model.pth"

if [ ! -f "$MODEL" ]; then
    echo "Model not found!"
    exit 1
fi

python train.py --games 1 --no-headless --no-plot --fps 15 --model "$MODEL" --resolution 3
