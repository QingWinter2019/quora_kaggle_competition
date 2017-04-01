source activate py35
python src/feature_engineering.py > logs/feature_engineering.txt 2>&1
python src/modelling.py > logs/modelling.txt 2>&1
