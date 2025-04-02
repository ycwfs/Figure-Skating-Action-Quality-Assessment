# Requirements

- Linux
- Python 3.10+
- PyTorch 2.4.0
- mamba_ssm
- causal_conv1d
- TensorBoard
- CUDA 11.0+
- GCC 4.9+
- 1.11 <= Numpy <= 1.23
- PyYaml
- Pandas
- h5py
- joblib

# Install mamba package

* cd ./mamba
* `pip install causal-conv1d`
* `pip install . --no-build-isolation`

# Compilation

Part of NMS is implemented in C++. The code can be compiled by

```shell
cd ./libs/utils
python setup.py install --user
cd ../..
```

The code should be recompiled every time you update PyTorch.
