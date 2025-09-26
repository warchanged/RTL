#!/bin/bash

python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install vllm tensorboard
python -m pip install flash-attn --no-build-isolation
python -m pip install flashinfer-python

python -m pip install --upgrade -r requirements_08.txt