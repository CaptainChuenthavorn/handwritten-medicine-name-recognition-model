import pandas as pd # for CSV file I/O operations
train = pd.read_csv('01_init\\train_test_data\\train.csv')
test = pd.read_csv('01_init\\train_test_data\\test.csv')

# x is features, y is your target column
x_train = train.iloc[:,0]
y_train = train.iloc[:,0]

x_test = test.iloc[:,0]
y_test = test.iloc[:,0]

print(x_train)
x_train.to_csv("01_init\\train_test_data\\x_train.csv")
y_train.to_csv("01_init\\train_test_data\\x_train.csv")
x_train.to_csv("01_init\\train_test_data\\x_train.csv")
x_train.to_csv("01_init\\train_test_data\\x_train.csv")