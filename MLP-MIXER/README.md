# Time Series Prediction with MIXER-MLP

## Requirements
All requirements are in the `requirements.txt`

You can install it with the following:
```
$ pip install -r requirements.txt
```

## Training
The complete code for training is in `trainer.py`
To train a model, you can use the following: (it by default uses the Apple dataset from yfinance; you can change it at your will)
```
$ python3 trainer.py --exp_name exp1 --log_dir ./logs --lr 0.01 --minmax_scalar 1 --batch_size 8 --n_threads 4 --save_model_interval 1000 --dataset_params Close Open Volume
```

In the above, keep the param you want to predict first for the `--dataset_params` field.

## Testing

You can use the `inference.ipynb` jupyter notebook for testing.


For Any Queries, you can reach: `Prabhat Chellingi` (CS20BTECH11038@iith.ac.in), Avula Mohana Durga Dinesh Reddy (
CS20BTECH11005@iith.ac.in)
