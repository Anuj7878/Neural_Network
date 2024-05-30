#!/bin/bash

sudo apt install python3-pip

sudo apt install python3-venv

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

code .