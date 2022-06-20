import pandas as pd

Y_test = pd.Series()
Y_test_p = pd.Series()

exec(open('TA_features.py').read())

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)

Capital = 100000
Earned = 0
for i in range(1, Y_test_p.shape[0]-1):
    if Y_test_p[i] > Y_test[i-1]:
        Earned = Earned + Y_test[i+1] - Y_test[i]
        print('Buy ' + str(Earned))
    else: 
        Earned = Earned + Y_test[i] - Y_test[i+1]
        print('Sell ' + str(Earned))
print(Earned)
