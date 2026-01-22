#!/bin/bash
streamlit run ui.py \
  --server.port=${PORT:-8501} \
  --server.address=0.0.0.0 \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
