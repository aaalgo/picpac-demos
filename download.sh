#!/bin/bash

if [ -z "$1" ]; then exit; fi

BN=`basename $1`
if [ ! -f $BN ]
then
    wget "$1" -O $BN
fi

if [ ! -f picpac-explorer ]
then
    wget http://www.aaalgo.com/picpac/binary/picpac-explorer
    chmod +x picpac-explorer
fi

