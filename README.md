# Quality Control Demo
Simple OAK-D camera based Quality Control demo.

## Basic camera setup
```bash
# Create a basic virtual environment.
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
# Initiate & update submodules.
git submodule update --init --recursive
# Install OAK-D camera dependencies.
cd depthai && python3 install_requirements.py
cd ../depthai-python/examples && python3 install_requirements.py
# Test the camera.
python3 ColorCamera/rgb_preview.py
```
Extra code samples: https://docs.luxonis.com/projects/api/en/latest/tutorials/code_samples/#code-samples

## Measurement
Measurement consists of basic object isolation, based upon simple image processing. It is followed by computation & visualization of oriented bbox.
To run: `python3 measure.py`
