# Amazon Delivery Time Prediction

a machine learning project where i predict how long a delivery will take (in minutes) based on things like distance, traffic, weather, the agent, and the vehicle.

its a regression problem (predicting a number, not a category). i used a random forest. did this to practice the full ML workflow: load data -> clean -> feature engineering -> train -> evaluate.

## the data

the real dataset is the **Amazon Delivery** dataset on Kaggle (search "amazon delivery time prediction"). it has columns like agent age/rating, store + drop GPS coordinates, weather, traffic, vehicle, area, product category, and the delivery time.

> ⚠️ i did NOT commit the real dataset (it's big and not mine to redistribute). instead there's a small **synthetic sample** in `data/sample_deliveries.csv` so the code runs out of the box. the numbers below are from that sample - real data will be different. to run on the real thing: `python train.py --data path/to/amazon_delivery.csv`

## how to run

```bash
pip install -r requirements.txt
python generate_sample.py     # creates the synthetic sample csv
python train.py               # trains + prints results
```

## what i did

1. **cleaning** - dropped rows with no target value
2. **feature engineering** - the biggest one: turned the store + drop GPS coordinates into a single `distance_km` feature using the haversine formula. distance turned out to be by far the most important feature (makes sense).
3. **encoding** - one-hot encoded the categorical columns (weather, traffic, vehicle, area, category) with `pd.get_dummies`
4. **model** - RandomForestRegressor (200 trees), 80/20 train-test split
5. **evaluation** - MAE, RMSE and R²

## results (on the synthetic sample)

```
MAE  : 6.88 minutes
RMSE : 8.46 minutes
R2   : 0.838

top features:
  distance_km    0.682
  Vehicle_van    0.097
  Traffic_Jam    0.055
```

so on average the prediction is off by ~7 minutes, and the model explains ~84% of the variance. distance dominates, then vehicle type and heavy traffic - which all makes intuitive sense.

## things i'd improve

- [ ] try other models (XGBoost, gradient boosting) and compare
- [ ] tune the hyperparameters instead of just using defaults
- [ ] use the actual pickup/order timestamps (the real dataset has them) to get a "prep time" feature
- [ ] cross-validation instead of a single split
- [ ] a notebook with charts (distance vs time scatter etc)

## stack
python, pandas, scikit-learn, numpy

MIT license.
