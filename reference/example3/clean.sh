#!/usr/bin/env bash

echo "Cleaning the raw train and test CSVs"
mkdir -p data/cleaned_quora
python clean_quora.py data/train.csv data/cleaned_quora/
# python clean_quora.py data/test.csv data/cleaned_quora/


echo "Splitting the train CSV into train and val and moving it to data/processed/quora"
mkdir -p data/processed_data
python data_split.py 0.1 data/cleaned_quora/train_cleaned.csv data/processed_data/


# echo "Copying the cleaned test CSV into data/processed/quora"
# cp data/cleaned_quora/test_cleaned.csv data/processed_data/test_final.csv
