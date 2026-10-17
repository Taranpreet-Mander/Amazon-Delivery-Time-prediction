# notes

## what this project is

practice regression project. predict delivery time from delivery details.
chose it because delivery ETA is a real thing companies care about and the
dataset has a nice mix of numeric + categorical + geo features.

## what i learned

- the GPS columns are useless on their own to the model, but turning store+drop
  into a haversine distance made it the single most predictive feature. feature
  engineering > throwing raw columns at the model.
- random forests dont need feature scaling, which is nice.
- get_dummies for the categorical columns. have to be a bit careful the real
  data might have categories the sample doesnt.
- MAE is easier to explain than RMSE ("off by ~7 minutes on average") so i report both.

## honest notes

- the committed csv is SYNTHETIC - i wrote generate_sample.py to make fake but
  realistic rows so the repo runs without downloading anything. real kaggle data
  will give different (probably worse) numbers because real data is messier.
- default hyperparameters, single train/test split. good enough to learn the
  workflow, not a tuned production model.

## todo
- xgboost comparison
- proper CV + hyperparameter tuning
- eda notebook with plots
