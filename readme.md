# MNIST CNN with user input via Flask web app

This is just a basic conv net I made and then integrated into a flask web app which I did with the help of a [Siraj Raval](https://github.com/llSourcell) video on how to [deploy a Keras model to the web](https://www.youtube.com/watch?v=f6Bf3gl4hWY&feature=youtu.be).

## Dependencies

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all the required libraries.

```bash
pip install numpy
pip install keras
pip install tensorflow
pip install flask
pip install pillow
pip install scipy==1.1.0
pip install h5py
```
I am using an older version of scipy here because any newer would not allow us to use the imread, imsave, and imresize functions that are used.
