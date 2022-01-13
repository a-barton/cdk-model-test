#!/bin/bash

if [ $1 = "train" ]; then
    python ./train.py
elif [ $1 = "serve" ]; then
    python ./serve.py
elif [ $1 = "debug" ]; then 
    /bin/bash
else
    echo "You haven't supplied a valid command. Aborting."
    exit 1
fi