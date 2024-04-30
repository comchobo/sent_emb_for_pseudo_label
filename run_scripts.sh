huggingface-cli login --token 'your_token'
jupyter nbconvert --to script notebook_code.ipynb
python notebook_code.py --yaml_path configs/baseline.yaml --mode mini_local
python notebook_code.py --yaml_path configs/baseline_token_max_fill.yaml --mode mini_local
rm notebook_code.py