# Time Series Prediction with MIXER-MLP

## Requirements
All requirements are in the `requirements.txt`

you can install with:
```
$ pip install -r requirements.txt
```

## Training
The complete code for training is in `trainer.py`
To train a model you can use the following: (it by default using apple dataset from yfinance, you can change it at your will)
```
$ python3 trainer.py --exp_name exp1 --log_dir ./logs --lr 0.01 --minmax_scalar 1 --batch_size 8 --n_threads 4 --save_model_interval 1000 --dataset_params Close Open Volume
```

In the above keep the param you want predict first for `--dataset_params` field

## Testing

For testing you can use the `inference.ipynb` jupyter notebook


For Any Queries you can reach me: `Prabhat Chellingi` (CS20BTECH11038@iith.ac.in)