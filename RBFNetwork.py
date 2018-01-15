import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def h(X):
    return 0.5+0.4*np.sin(2*np.pi*X)

def noise(X):
    return X+np.random.uniform(-0.1,0.1, X.shape)

def getTrainData():
    X = np.random.uniform(0, 1, (75, 1))
    D = noise(h(X))
    return X,D

def intializeWeights(clusters):
    return np.random.uniform(-1, 1, (clusters, 1))

def gaussian(param, x):
    mean = param[:,0]
    sd = param[:,1]
    return np.exp((-1*(x-mean)**2)/(2*sd**2)).reshape(param.shape[0],1)

def euc_dist(x1, x2):
    return np.abs(x1-x2)

def same_width(centers,clusters):
    sd = (np.amax(centers)-np.amin(centers))/np.sqrt(2*clusters)
    sd = np.repeat(sd,clusters).reshape(clusters,1)
    return sd

def diff_width(centers,X,predicted_clusters,clusters):
    sd = np.sqrt(np.array([np.mean((centers[i,] - X[predicted_clusters == i]) ** 2, axis=0) for i in range(clusters)]))
    for i in range(sd.shape[0]):
        if sd[i, 0] == 0.0:
            sd[i, 0] = np.mean(sd[np.arange(len(sd)) != i])
    return sd

def kmeans(X, clusters, isSameWidth):
    centers = np.random.choice(np.squeeze(X), clusters, False).reshape(clusters,1)

    while True:
        dist_from_centers = np.squeeze(np.array([euc_dist(X, centers[i,]) for i in range(clusters)])).T
        predicted_clusters = np.argmin(dist_from_centers, axis=1)
        centers_new =  np.array([np.mean(X[predicted_clusters == i], axis=0) for i in range(clusters)])
        if np.array_equal(centers_new, centers):
           break
        centers = centers_new

    if not isSameWidth:
        sd = diff_width(centers,X,predicted_clusters,clusters)
    else:
        sd = same_width(centers,clusters)

    return np.squeeze(np.dstack((centers, sd)))

def lms(W, g, d, y, eta, b):
    return W+eta*g*(d-y), b+eta*(d-y)

def getOutput(gaussian_params, x, W, b):
    return np.squeeze(np.dot((gaussian(gaussian_params, x)).T, W))+b

def rbs(X, D, eta, bases, isSameWidth):
    W = intializeWeights(bases)
    b = np.random.uniform(-1, 1)
    params = kmeans(X, bases, isSameWidth)

    for i in range(100):
        Y = []
        for j in range(X.shape[0]):
            y = getOutput(params,X[j, 0],W,b)
            W, b = lms(W, gaussian(params, X[j, 0]), D[j, 0], y, eta, b)
            Y.append(np.squeeze(y))
    return np.array(Y),W,b,params

def plotGraph(gaussian_params,X, D, bias, eta, bases, isSameWidth):
    if not os.path.exists('graphs/'):
        os.makedirs('graphs/')

    X1 = np.linspace(0, 1, 150)
    Y1 = np.array([getOutput(gaussian_params,x,W,bias) for x in X1])
    width = "Different"
    if isSameWidth:
        width = "Same"

    plt.scatter(X, D, label="Desired ouput")
    plt.plot(X1,Y1, label="RBF network", color="#52D017")
    plt.plot(X1,h(X1), label="Original function", color="#4B0082")
    plt.title("Eta = {}, Bases = {}, Gaussian Width = {}".format(eta,bases,width))
    plt.legend()
    plt.savefig('graphs/' + str(eta) + "_" + str(bases) + "_" + str(width) + '.jpg')
    plt.clf()


if __name__ == "__main__":
    X, D = getTrainData()
    bases = [2,4,7,11,16]
    eta = [0.01,0.02]
    gaussian_width=[False,True]

    for b in bases:
        for e in eta:
            for isSameWidth in gaussian_width:
                Y,W,bias,g = rbs(X,D,e,b,isSameWidth)
                plotGraph(g,X,D,bias,e,b,isSameWidth)