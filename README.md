# picpac-demos

## Files
- io-example.py:  How to import images into picpac database.
- cls-train.py: How to train classification models.
- fcn-train.py: How to train segmentation models with FCN.

## Prerequisites
- Tensorflow > 1.3
- Picpac

```
sudo apt-get install libopencv-dev libboost-all-dev libgoogle-glog-dev
git clone https://github.com/aaalgo/picpac
cd picpac
git submodule init
git submodule update
python setup build
sudo python setup install
```

