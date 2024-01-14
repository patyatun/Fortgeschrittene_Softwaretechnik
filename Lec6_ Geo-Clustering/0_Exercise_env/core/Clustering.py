
from sklearn.cluster      import Birch
import core.HelperTools   as ht

#------------------------------------------------------------------------------
@ht.timer
def birchen(X, bfac, nclusters, thres):
    """BIRCH-Clustering Algorithm"""
    brc = Birch(branching_factor=bfac, n_clusters=nclusters,threshold=thres)
    brc.fit(X)
    return brc.predict(X)


