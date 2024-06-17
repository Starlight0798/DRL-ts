#!/bin/bash
find . -type d -name "log" -exec rm -rf {} +
find . -type d -name "__pycache__" -exec rm -rf {} +
echo "All log and __pycache__ directories have been deleted."
