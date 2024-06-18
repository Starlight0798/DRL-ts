#!/bin/bash
find . -type d -name log -exec rm -fr "{}" \;
find . -type d -name __pycache__ -exec rm -fr "{}" \;
echo "All log and __pycache__ directories have been deleted."
