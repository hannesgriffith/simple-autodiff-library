# simple-autodiff-library

Basic implementation of an autodiff library, based loosely on the PyTorch and TensorFlow API's and naming, and only using NumPy and NetworkX. simple-autodiff-library also includes some functionality to build and train deep learning models. The [features](Features) and [limitations](Limitations) of the library are summarised below. Uses of the library are shown in various Jupyter notebooks, as referenced in [Usage](Usage), which also gives a high-level summary for usage.

## Usage

This section summarises the library at a high level. More in-depth uses of the library are demonstrated for the following applications:

- [Linear Regression](<Linear Regression.ipynb>)
- Fully-Connected Network (or MLP)
- Recurrent Neural Network

Usage summary... (to be added)

## Features

- Dynamically built graph so:
  - Tensor values can be viewed for debugging
  - Graph supports native Python conditional operations, for and while loops
- Backward only propagates gradients to ancestors of tensor that backward was called on

## Limitations

- Only runs on CPU
- Only supports operations with single output tensor
- Only supports single computational graph at once
- All Input and Parameter Tensors need to be defined before any operations are added to the graph
- Backward propagates gradients to all ancestor tensors, not just ones that require gradients
- Clears graph after backward, so doesn't support gradient accumulation over multiple forward passes

## To-Do's

- Finish Jupyter notebooks:
  - MLP
  - RNN
- Add Op gradient testing
- Add regularisation options
