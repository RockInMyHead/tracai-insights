#!/bin/bash
cd /Users/artembutko/Desktop/trackAI/backend
python3 -m uvicorn main:app --host 127.0.0.1 --port 8002 --log-level info
