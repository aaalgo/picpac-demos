#!/bin/bash


./download.sh http://www.aaalgo.com/picpac/datasets/cifar/cifar10-train.picpac
./download.sh http://www.aaalgo.com/picpac/datasets/cifar/cifar10-test.picpac

./cls-train.py --db cifar10-train.picpac --val_db cifar10-test.picpac --classes 10
