import numpy as np

def activation(x):
   return 1 / (1 + np.exp(-x))


def deltaK(y,x):
   return -x*(1-x)*(y-x)

def deltaJ(x,W,D):
    sum = np.dot(D, W)
    return x*(1-x)*sum

def nnTrain(X,z,N,M,K,a):
    np.random.seed(10)
    x0 = np.zeros(N)
    w1 = np.random.uniform(-1, 1, (K, N))
    w2 = np.random.uniform(-1, 1, (M, K + 1))
    err = 0
    for j in range(10000):
        print(err)
        err = 0
        for i in range(100):
            x0[0] = 1
            for q in range(N - 1):
                x0[q + 1] = X[i][q]

            z1 = np.dot(w1, x0)
            z1 = activation(z1)
            x1 = np.insert(z1, 0, 1)
            z2 = np.dot(w2, x1)
            z2 = activation(z2)
            err += sum((z[i] - z2) ** 2)
            d2 = deltaK(z[i],z2)
            d1 = deltaJ(x1,w2,d2)
            d1 = d1[1:]
            w2 -= a * np.dot(d2.reshape(len(d2), 1), x1.reshape(1, len(x1)))
            w1 -= a * np.dot(d1.reshape(len(d1), 1), x0.reshape(1, len(x0)))


def nn2():
    X = np.loadtxt("nndata2.txt")
    z = np.loadtxt("nndata2_ans.txt")
    nnTrain(X,z,65,4,100,0.8)

def nn1():
    X = np.empty([0,2])
    z = np.empty([0,1])
    f = open("nndata1.txt")
    for line in f:
        a = line.split(" ")
        X = np.append(X, [[float(a[0]),float(a[1])]], axis = 0)
        z = np.append(z, float(a[2]))
    nnTrain(X,z,3,1,20,0.8)

nn2()