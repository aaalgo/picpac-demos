#!/bin/bash

./download.sh http://www.aaalgo.com/picpac/datasets/nmist/nmist-train.picpac
./download.sh http://www.aaalgo.com/picpac/datasets/nmist/nmist-test.picpac

./cls-train.py --db nmist-train.picpac --val_db nmist-test.picpac --classes 10
