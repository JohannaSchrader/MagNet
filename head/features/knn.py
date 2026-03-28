import sklearn.base
import sklearn.neighbors
import numpy as np


class KNNDistanceFeature(sklearn.base.BaseEstimator,
                         sklearn.base.TransformerMixin):

    def __init__(self, ks=(1, 2, 4), metric='euclidean'):
        self.ks = ks
        self.metric = metric
        self.knns = None
        self.n_classes = None

    def fit(self, x, y):
        self.n_classes = y.max() + 1
        self.knns = []
        max_k = max(self.ks)

        for c in range(self.n_classes):
            #if c in y:
                print('c: ',c)
                print('n_ classes: ',self.n_classes)
                print('x: ',x)
                print('y: ',y)
                print('c in y: ',c in y)
                x_selected = x[y == c]
                print('x_selected: ',x_selected)
                print('max_k: ',max_k)
                print('metric: ',self.metric)
                knn = sklearn.neighbors.NearestNeighbors(
                    n_neighbors=max_k, metric=self.metric, n_jobs=4)
                knn.fit(x_selected)
                self.knns.append(knn)
            #else:
                print('c [',c,'] is not in y: ',y)

                print('c: ',c)
                print('n_ classes: ',self.n_classes)
                print('x: ',x)
                print('y: ',y)
                x_selected = x[y == c]
                print('x_selected: ',x_selected)
                print('max_k: ',max_k)
                print('metric: ',self.metric)
                print('else ende')

    def transform(self, x):
        ret = np.vstack(tuple(
            knn.kneighbors(x, min(k, knn._fit_X.shape[0]))[0].mean(axis=1) #knn.kneighbors(x, k)[0].mean(axis=1)
            for k in self.ks
            for knn in self.knns
        ))
        print(ret)
        print(ret.shape)
        ret = ret.transpose(1, 0)
        print(ret.shape)
        print('len ks * classes: ',len(self.ks) * self.n_classes)
        print('len ks: ',len(self.ks))
        print('classes: ', self.n_classes)
        print('len ret 0: ',len(ret[0]))
        assert len(ret) == len(x)
        assert len(ret[0]) == len(self.ks) * self.n_classes
        return ret
