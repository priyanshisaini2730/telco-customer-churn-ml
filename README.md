# Telco Customer Churn – End-to-End ML Project

## Purpose
A business-framed machine learning pipeline that goes beyond churn prediction — it segments customers, estimates Customer Lifetime Value (CLV), and maps retention actions.

## Problem Solved & Business Value
- **Proactive retention**: Predicts which customers are likely to churn so teams can act before they leave, not after.
- **Business-first framing**: Pairs churn probability with CLV and an RFM-style segmentation adapted for telecom, so predictions translate into prioritized actions.
- **Operationalized ML**: Model is served via a REST API and a web UI — accessible without opening a notebook.
- **Reproducible experiments**: MLflow tracks every run, its metrics, and artifacts for auditability.
- **Automated delivery**: CI/CD builds and ships a new Docker image on every push to `main`.

## What's Built
- **Feature engineering**: RFM (Recency, Frequency, Monetary) adapted for telco — `tenure_bucket` (relationship fragility), `num_services` (switching inertia), `charge_per_tenure_ratio` (avoids multicollinearity with `TotalCharges`, flagged via VIF).
- **Modeling**: XGBoost classifier tuned with Optuna; all runs logged to MLflow with metrics, parameters, and artifacts.
- **Train/serve consistency**: Feature transformation logic lives in one shared function (`src/features/rfm_features.py`), used by both the training pipeline and the inference service — closing a train/serve skew gap that existed when this logic was duplicated.
- **Inference service**: FastAPI app (`src/app/main.py`) exposing:
  - `GET /` – health check
  - `POST /predict` – churn prediction from 18 customer features
- **Web UI**: Gradio interface mounted at `/ui` on the same FastAPI app for quick, no-code testing.
- **Business layer**: `scripts/batch_scoring.py` runs batch predictions, estimates CLV, and builds a risk-value segmentation matrix for retention prioritization.
- **Containerization**: Single Docker image (`dockerfile`) running the FastAPI+Gradio app via Uvicorn on port 8000.
- **CI**: GitHub Actions builds the Docker image on every push to `main` and pushes it to Docker Hub (`prima30/telco-fastapi:latest`).
- **Deployment**: AWS EC2 (t3.micro, free tier) running the Docker Hub image directly.

## Tech Stack
| Layer | Tools |
|---|---|
| Modeling | XGBoost, Optuna |
| Experiment tracking | MLflow |
| Serving | FastAPI, Uvicorn, Gradio |
| Containerization | Docker |
| CI/CD | GitHub Actions, Docker Hub |
| Deployment | AWS EC2 |

## Key Design Decisions
- **CLV formula**: `CLV = MonthlyCharges × min(1 / predicted_churn_probability, 60)`. The 60-month cap is a deliberate business decision — framed as a conservative 5-year planning window — not a statistical artifact (the dataset's max tenure is 72 months).
- **Risk-value matrix uses `MonthlyCharges`, not CLV, on the value axis**: CLV is derived *from* churn probability, so plotting both on the same matrix would create a built-in negative correlation, making some cells structurally unreachable.
- **Retention savings metric was deliberately excluded** from batch scoring: the dataset is a single churn snapshot with no intervention-outcome data, so any "money saved by retention" figure would be ungrounded. Only metrics with real data support are reported.
- **Notebook vs. production recall/precision gap is intentional, not a bug**: the notebook optimizes recall as the sole objective during exploration; the production pipeline uses fixed hyperparameters at a deliberately chosen precision-recall trade-off point.

## Deployment Flow
1. Push to `main` → GitHub Actions builds the Docker image and pushes it to Docker Hub.
2. On the EC2 instance, the latest image is pulled and run as a container:
