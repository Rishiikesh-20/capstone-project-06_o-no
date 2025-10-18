#!/bin/bash

cd "$(dirname "$0")"

echo "=========================================="
echo "Running Predictive Maintenance System"
echo "=========================================="
echo

python3 -c "import gymnasium, lightgbm, tensorflow, stable_baselines3, imblearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Installing missing packages..."
    python3 -m pip install --user --quiet gymnasium lightgbm stable-baselines3 tensorflow pytest imbalanced-learn
fi

python3 main.py "$@"

