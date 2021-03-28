#!/usr/bin/env bash

echo "=== Acquiring datasets ==="
echo "---"

if [ -d "$PWD/data/" ]
then
  echo "Directory /data/ exists."
else
  echo "Create Directory /data."
  mkdir data
fi

echo "Downloading Glue Datasets..."
if [ -d "$PWD/data/MRPC" ]
then
  echo "Glue data already exist."
else
  python src/glue/download_glue_2.py
#  python src/glue/download_glue.py
fi
echo "---"

echo "Downloading TREC-6..."
if [ -d "$PWD/data/TREC-6" ]
then
  echo "Directory /data/TREC-6 exists."
else
  echo "Directory /data/TREC-6 does not exist."
  cd data
  mkdir TREC-6
  cd TREC-6
  wget -O train.txt https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label?fbclid=IwAR3jRGuCdhqctyGK6SYvOcYqSYO3LGLp9nkFara24JbO8SZrRoHlE6-qFzo
  wget -O test.txt https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label?fbclid=IwAR1Ka5-NV_86uXVerqRcDTr7QjSSTJDIg2lrKbqTAhe0WLkOICmKv86JKFs
  cd ../../
fi
echo "---"
#
echo "Downloading AG_NEWS..."
# I have to fix that
if [ -d "$PWD/data/AG_NEWS" ]
then
  echo "Directory /data/AG_NEWS exists."
else
  echo "Directory /data/AG_NEWS does not exist."
  cd data
  mkdir AG_NEWS
  cd AG_NEWS
  https://drive.google.com/file/d/1X8hXEEpVscCVPsQnBKZZjozoUP7QL-mI/view?usp=sharing
  wget https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms #-O "ag_news_csv.tar.gz"
  tar -xzvf "ag_news_csv.tar.gz" -C "${DATADIR}"
  cd ../../
fi
echo "---"

#echo "Downloading DBPedia..."
#if [ -d "$PWD/datasets/DBPEDIA" ]
#then
#  echo "Directory /datasets/DBPEDIA exists."
#else
#  cd datasets
#  mkdir DBPEDIA
#  cd DBPEDIA
#  wget https://drive.google.com/file/d/0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k
#fi

echo "---"
echo "The End :)"
