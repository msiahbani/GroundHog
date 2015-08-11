#!/bin/bash

W_DIR="$1"
VocabSize=$2
src=cs
tgt=en

echo "Process of input files"
python preprocess.py -d $W_DIR/vocab.$src.pkl -v $VocabSize -b $W_DIR/binarized_text.$src.pkl -p $W_DIR/*.$src -r # --embd-file $W_DIR/vectors.bin 

python invert-dict.py $W_DIR/vocab.$src.pkl $W_DIR/ivocab.$src.pkl

python convert-pkl2hdf5.py $W_DIR/binarized_text.$src.pkl $W_DIR/binarized_text.$src.h5

echo "Process of output files"
python preprocess.py -d $W_DIR/vocab.$tgt.pkl -v $VocabSize -b $W_DIR/binarized_text.$tgt.pkl -p $W_DIR/*.$tgt

python invert-dict.py $W_DIR/vocab.$tgt.pkl $W_DIR/ivocab.$tgt.pkl

python convert-pkl2hdf5.py $W_DIR/binarized_text.$tgt.pkl $W_DIR/binarized_text.$tgt.h5

python shuffle-hdf5.py $W_DIR/binarized_text.$src.h5 $W_DIR/binarized_text.$tgt.h5 $W_DIR/binarized_text.$src.shuf.h5 $W_DIR/binarized_text.$tgt.shuf.h5

