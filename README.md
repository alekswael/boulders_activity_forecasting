# Forecasting member activity in a climbing gym using NeuralProphet

This repo contains the code for forecasting member activity in a popular climbing gym in Aarhus (Boulders). The pipeline uses a configured *NeuralProphet* model and two baseline models for forecasting. The pipeline consists of data pre-processing, time-series cross-validation of the models and forecasting the final week activity.

Exam project for the course *Data Science, Prediction and Forecasting* as part of the Cognitive Science MSc program at Aarhus University.


## Repo structure

| Folder/File               | Description |
|---------------------------|-------------|
| `output/`                   |Contains all output from the scripts|
| `src/`               |Contains all Python scripts used in the pipeline|

*NOTE:* The data is not publicly available currently.


## System specifications
All models were trained on a laptop running Ubuntu 22.04.4 LTS with an Intel
Core i7-10510U CPU and 16GB RAM.

## Setup

Make sure the repo is set as your current working directory. From there, run the ```setup.sh``` script to create a venv and install the required dependencies.

```bash
# Create venv
python3 -m venv venv

# Activate venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Deactivate venv
deactivate
```

## Run pipeline

To run the analysis, simply run the ```run.sh``` script. This runs all the Python scripts in the ```src/``` folder.

```bash
# Activate venv
source venv/bin/activate

# Run scripts
python3 src/preprocess_data.py
python3 src/baseline_crossval.py
python3 src/np_crossval.py
python3 src/forecast_week.py

# Deactivate venv
deactivate
```

*NOTE:* Some plots can be only obtained by running the ```parameter_component_PACF_plot.ipynb``` Jupyter Notebook.
