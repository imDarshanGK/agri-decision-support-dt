# Decision Tree Optimization Framework for Agricultural Decision Support

An interpretable, hyperparameter-optimized Decision Tree system for crop recommendation.
Goes beyond single-label prediction by providing ranked crop alternatives, feature
importance analysis, and human-readable decision rules — served through a FastAPI
backend and a custom-designed web UI.

## Research Gap
Existing crop recommendation systems focus mainly on classification accuracy, use
manually-set Decision Tree hyperparameters, return only a single predicted crop, and
offer limited actionable explanations. This project addresses all four gaps.

## Methodology
```
Crop Recommendation Dataset
    -> Data Collection
    -> Data Preprocessing
    -> Exploratory Data Analysis
    -> Feature Analysis
    -> Train/Test Split
    -> Baseline Decision Tree
    -> Hyperparameter Optimization
    -> Optimized Decision Tree
    -> Crop Prediction
    -> Crop Suitability Ranking
    -> Explainable Analysis
    -> Decision Rules
    -> Agricultural Decision Support
    -> Model Evaluation
```

## Setup

```bash
git clone https://github.com/<your-username>/agri-decision-support-dt.git
cd agri-decision-support-dt
pip install -r requirements.txt
```

Download the Kaggle "Crop Recommendation Dataset" and place it at:
`data/raw/Crop_recommendation.csv`

## Run the ML pipeline

```bash
python main.py
```

Or run each stage individually:

```bash
python src/data_preprocessing.py
python src/train_baseline.py
python src/optimize_dt.py
python src/rank_crops.py
python src/explain.py
python src/evaluate.py
```

## Run the web app (API + UI)

```bash
uvicorn api:app --reload
```

Open http://127.0.0.1:8000 — the FastAPI backend serves both the `/api/*`
endpoints and the frontend at `frontend/index.html`.

- `GET  /api/health` — model status check
- `POST /api/predict` — ranked crop recommendations from soil/climate readings
- `GET  /api/rules` — extracted decision-tree rules as text
- Interactive API docs auto-generated at `/docs`

An alternative Streamlit demo is also available: `streamlit run app.py`

## Run the tests

```bash
pip install pytest pytest-cov httpx
pytest tests/ -v
pytest tests/ --cov=src --cov=api --cov-report=term-missing
```

43 tests cover preprocessing, baseline/optimized training, ranking, explainability,
evaluation, and every API endpoint (including validation and error paths), using
an isolated synthetic dataset so tests never touch or depend on your real data.

## Project Structure

```
agri-decision-support-dt/
├── data/                 # raw and processed data
├── notebooks/            # exploratory work
├── src/                  # pipeline scripts
├── tests/                # pytest suite (43 tests)
├── frontend/              # web UI (HTML/CSS/JS)
├── models/                # saved trained models
├── outputs/                # figures, rules, comparison tables
├── report/                 # final written report
├── api.py                  # FastAPI backend serving model + UI
├── app.py                  # alternative Streamlit demo
└── main.py                 # runs the ML pipeline end-to-end
```

## Design system (frontend)
- Palette: soil-dark #1E2419, soil-mid #2C3524, wheat #D9A441, leaf #7FA65C, sky-rain #6FA8B5, paper #EDE6D6
- Type: Fraunces (display), IBM Plex Sans (body), IBM Plex Mono (data/metrics)
- Signature interaction: ranked crop confidence renders as animated growth bars rather than generic progress bars

## Results
See `outputs/model_comparison.csv` and `outputs/figures/` after running the pipeline.

## License
MIT
