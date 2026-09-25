#!/bin/bash

uv run jupyter execute --inplace notebooks/data_preprocessing.ipynb
uv run jupyter execute --inplace notebooks/data_exploration.ipynb

uv run jupyter execute --inplace notebooks/attention.ipynb &
uv run jupyter execute --inplace notebooks/catboost.ipynb &
uv run jupyter execute --inplace notebooks/elasticnet.ipynb &
uv run jupyter execute --inplace notebooks/gru.ipynb &
uv run jupyter execute --inplace notebooks/lstm.ipynb &
uv run jupyter execute --inplace notebooks/mlp.ipynb &
uv run jupyter execute --inplace notebooks/random_forest.ipynb &

wait

uv run jupyter execute --inplace notebooks/feature_selection.ipynb
uv run jupyter execute --inplace notebooks/latefusion.ipynb
uv run jupyter execute --inplace notebooks/results_analysis.ipynb
uv run jupyter execute --inplace notebooks/shap_analysis.ipynb

python combine_results.py

