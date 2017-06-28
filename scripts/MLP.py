from sklearn.neural_network import MLPRegressor
#x is your training series as a matrix
#t is your testing series as a matrix
#Training
X = []
y = []
for i in np.arange(25, x.shape[0]):
    lags = []
    for k in np.arange(1, 25):
        lags.append(x[i-k])
    X.append(lags)
    y.append(x[i])
X = np.squeeze(np.array(X))
y = np.array(y)


#Change parameters to MLP
MLP = MLPRegressor()
MLP.fit(X, y)

#Testing
X = []
y = []
for i in np.arange(25, t.shape[0]):
    lags = []
    for k in np.arange(1, 25):
        lags.append(t[i-k])
    X.append(lags)
    y.append(t[i])
X = np.squeeze(np.array(X))
y = np.array(y)

y_pred = MLP.predict(X)

print np.mean(np.abs(y_pred-y))
plt.plot(y_pred, label='predicted')
plt.plot(y, label='actual')
plt.legend()
plt.show()