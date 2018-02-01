# picpac-demos

## What is PicPac?

[PicPac](https://github.com/aaalgo/picpac) is an image database designed for streaming training
examples into deep-learning frameworks.  It is written in
C++ and has Python and Lua bindings, and is compatible with
common frameworks like Tensorflow, MxNet, Torch and other
frameworks based on Python.  The goal is to replace
platform-specific database format with a generic, high-
performance database that is specially-designed for
deep-learning model training.

Highlights:
- Asynchronous image decoding & augmentation with multi-threads,
  minimizing GPU waiting on CPU.
- Small databases are fully cached in memory.
- Random shuffle of huge dataset on SSD storage (cache=False).
- ImageNet 2015 down-sized (shorter edge to 255), mozjpeg-encoded and
  preloaded in a [12G
  database](http://www.aaalgo.com/picpac/datasets/ilsvrc/)(originally
  155G), ready for training.
- Explore the database and check augmentation results with [PicPac
  Explorer](http://www.aaalgo.com/picpac/binary/picpac-explorer).


## Files
- io-example.py:  How to import images into picpac database.
- cls-train.py: How to train classification models.
- fcn-train.py: How to train segmentation models with FCN.

- run-cifar10/nmist.py: Download pre-loaded database and train.

## Prerequisites
- Ubuntu 16.04
- Tensorflow > 1.3
- PicPac

```
sudo apt-get install libopencv-dev python-opencv libboost-all-dev libgoogle-glog-dev
git clone https://github.com/aaalgo/picpac
cd picpac
git submodule init
git submodule update
python setup build
sudo python setup install
```
- Other python dependencies
```
pip install tqdm simplejson
```
