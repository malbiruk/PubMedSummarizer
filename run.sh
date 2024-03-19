#!/usr/bin/env bash
if [ -n "$1" ]; then
  streamlit run web_interface.py --server.port $1
else
  streamlit run web_interface.py
fi
