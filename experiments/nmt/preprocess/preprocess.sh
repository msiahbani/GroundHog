#!/bin/bash

W_DIR="$1"
VocabSize=$2

echo "Process of input files"
python preprocess.py -d $W_DIR/vocab.en.pkl -v $VocabSize -b $W_DIR/binarized_text.en.pkl -p $W_DIR/*.en -r

python invert-dict.py $W_DIR/vocab.en.pkl $W_DIR/ivocab.en.pkl

python convert-pkl2hdf5.py $W_DIR/binarized_text.en.pkl $W_DIR/binarized_text.en.h5

echo "Process of output files"
python preprocess.py -d $W_DIR/vocab.fr.pkl -v $VocabSize -b $W_DIR/binarized_text.fr.pkl -p $W_DIR/*.fr

python invert-dict.py $W_DIR/vocab.fr.pkl $W_DIR/ivocab.fr.pkl

python convert-pkl2hdf5.py $W_DIR/binarized_text.fr.pkl $W_DIR/binarized_text.fr.h5

python shuffle-hdf5.py $W_DIR/binarized_text.en.h5 $W_DIR/binarized_text.fr.h5 $W_DIR/binarized_text.en.shuf.h5 $W_DIR/binarized_text.fr.shuf.h5

