#!/bin/bash
tail -n 4000000 data/train.csv > data/temp.csv
head -1 data/train.csv | cat - data/temp.csv > data/train_small.csv && rm data/temp.csv
