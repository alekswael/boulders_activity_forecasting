# Activate venv
source venv/bin/activate

# Run scripts
python3 src/preprocess_data.py
python3 src/baseline_crossval.py
python3 src/np_crossval.py
python3 src/forecast_week.py

# Deactivate venv
deactivate