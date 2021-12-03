#!/bin/bash

sudo docker run --rm -v pytorch/torchserve:latest torch-model-archiver --model-name caption --version 1.0  --serialized-file checkpoints/checkpoint_num_39__21_11_2021__16_33_06.pth.tar --extra-files ./index_to_name.json --handler custom_handler --export-path model-store -f