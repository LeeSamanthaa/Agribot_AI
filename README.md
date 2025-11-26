<!-- README for Agribot_AI -->
# Agri-Bot AI: Crop Health Forecasting Agent

AI-driven crop health monitoring that combines a GRU time-series model with an LLM-based conversational interface for actionable agronomic advice.

---

## Quick Summary
Agri-Bot AI (Crop Health) provides a 7-day NDVI crop health forecast using a GRU model and presents analysis through a Streamlit dashboard or a terminal chatbot agent. It integrates weather, soil moisture, and satellite imagery time-series data for robust area- and field-level monitoring.

Key features:
- GRU-based time-series predictor trained on Sentinel-2, ERA5, and SMAP features
- LLM-driven interpretation (Groq + Llama) to translate predictions into actionable suggestions
- Streamlit dashboard for visualization and interactive monitoring
- Terminal chatbot and test harness for debugging/QA

---

## Features & Components
- `app.py` — Streamlit-driven UI (Agribot Command Center).
- `run_chatbot.py` — Terminal test interface for the chatbot agent.
- `ai_agent/` — LLM and model agent sources (multi-agent design: Model Agent vs. Chat Agent).
- `machine_learning/` — Training logic, utilities, and saved model artifacts.
- `data/` — Cleaned datasets (parquet) used for mapping and diagnostics.
- `pre-processing_notebooks/` — Notebooks for data cleanup and feature engineering.

---

## Prerequisites
- Python 3.11+ (the project is configured for 3.11 via Poetry)
- Poetry (recommended) or a Python virtual environment
- Optional: GPU and CUDA drivers if you want to install TensorFlow with GPU acceleration
- Groq API Key — used by the LLM agent for inference (https://console.groq.com/)

---

## Installation (Poetry - recommended)
1. Clone the repository:

```powershell
git clone https://github.com/LeeSamanthaa/Agribot_AI.git
cd Agribot_AI
```

2. Install Poetry (if not installed):

```powershell
python -m pip install --upgrade pip
python -m pip install poetry
```

3. Install dependencies with Poetry:

```powershell
poetry install
```

4. (Optional) Activate the environment and run Streamlit directly:

```powershell
poetry shell
streamlit run app.py
```

Alternatively, run files directly from Poetry without entering the shell:

```powershell
poetry run streamlit run app.py
poetry run python run_chatbot.py
```

---

## Environment Variables (.env)
Create a `.env` file at the repository root with contents similar to:

```dotenv
# Groq - LLM provider
GROQ_API_KEY=your_groq_api_key_here

# Optional: reduce TF logs
TF_CPP_MIN_LOG_LEVEL=2
```

Notes:
- The repository uses `python-dotenv` to automatically load `.env` files for local development.
- If you don’t have a GROQ key, you can still run model inference locally without the LLM layer.

---

## Running the App & Chatbot
- Streamlit UI (Dashboard):

```powershell
poetry run streamlit run app.py
```

- Terminal Chatbot (debug/test harness):

```powershell
poetry run python run_chatbot.py
```

---

## Model & Data
- Model artifacts are stored in: `machine_learning/src/feat/Crop_Health_Model_V6/saved_models/` (e.g., `best_multivar_model_v6.0.keras`).
- Training datasets & results are under `machine_learning/src/feat/Crop_Health_Model_V6/training_results/`.
- Data samples and the main parquet used by the Streamlit dashboard are in `data/`.

Important scripts:
- Training: `machine_learning/src/feat/Crop_Health_Model_V6/train_v6_gru.py`
- Evaluation / zero-shot: `machine_learning/src/feat/Crop_Health_Model_V6/evaluate_zeroshot.py` and `evaluate_growing_seasons.py`

---

## Tests, Linters & Formatters
This project uses Poetry groups for development tools (black, isort, flake8, mypy, pylint, pytest).

- Run tests:

```powershell
poetry run pytest
```

- Formatting & linting (example):

```powershell
poetry run black .
poetry run isort .
poetry run flake8
```

---

## Development & Contribution
- To contribute, fork the repo, create a feature branch, and open a PR against the `main` branch.
- Add tests and update the docs for any user-facing or developer-facing changes.

Checklist for PRs:
- Follow repo code style and run linters
- Add or update tests where relevant
- Update README (or docs folder) if behavior changes

---


## License
This repository does not include a license file. Add a LICENSE to clarify the project's terms.

---

## Contact & Author
Samantha Lee — Engineer & Data Scientist
![alt text](<streamlit satellite monitor.png>)
![alt text](<90-day crop trend.png>)
![alt text](<chatbot.png>)