# Bank Note Authentication Classifier
A machine learning service to classify banknotes as genuine or counterfeit using the UCI Banknote Authentication dataset from Kaggle.

# Setup & Installation
## Install uv if not already available
pip install uv

## Initialize the project
uv init bank-note-classifier
cd bank-note-classifier

## Create a virtual environment
uv venv

## Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate   # Windows

## Install dependencies
uv add -r requirements.txt

# Running the Application
## Run directly
python app.py

## Or with Uvicorn for auto-reload (development)
uvicorn app:app --reload

The API will be accessible at http://127.0.0.1:8000