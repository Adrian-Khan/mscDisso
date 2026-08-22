lstm/cnnlstm/cnnlstm with attention

created both the baseline with typical params/intuitive params and then optimised this model using optuna to find the best params for these 3 models. this forms the baseline for adding more feature engineering, other models (BiLSTMS) and other improvements outside of hyperparams


Added more features to the best baseline model (based on avg r2):
- lagged features
- 2nd order derivative
- rolling windows
