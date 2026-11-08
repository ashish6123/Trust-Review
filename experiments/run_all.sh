#!/bin/bash
# Runs every GPU experiment in sequence. PY = python with CUDA torch + transformers.
set -e
cd "$(dirname "$0")"
PY="${PY:-python}"
export PYTHONIOENCODING=utf-8
$PY dl.py --name base
$PY dl.py --name weighted --weighted
$PY dl.py --name tagged --tag
$PY mlm.py
$PY dl.py --name mlm --init cache/mlm_yelp
$PY dl.py --name loso_yelp --exclude yelp
$PY dl.py --name all --weighted --tag --init cache/mlm_yelp
$PY fusion.py base all
echo ALL_DONE
