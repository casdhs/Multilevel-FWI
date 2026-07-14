#!/bin/bash

echo "=============================="
echo "Step 1: Generate data"
echo "=============================="

python 01_gen_data.py


echo "=============================="
echo "Step 2: Run conventional MS-FWI"
echo "=============================="

python 02_fwi_ms.py


echo "=============================="
echo "Step 3: Run ML-FWI with Laplacian pyramid"
echo "=============================="

python 03_fwi_ml_lap.py


echo "=============================="
echo "Step 4: Run ML-FWI with Gaussian pyramid"
echo "=============================="

python 04_fwi_ml_gau.py


echo "=============================="
echo "All tasks finished!"
echo "=============================="