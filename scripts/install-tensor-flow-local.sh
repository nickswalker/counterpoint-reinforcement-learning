pip3.5 install -r requirements.txt --target .
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp35-cp35m-linux_x86_64.whl
pip3.5 install --upgrade $TF_BINARY_URL --target .
