import numpy as np
import time
import sys
import logging
import pandas as pd
sys.path.append('../../../../Code/')
from plot_grid import load_h5, data
from kmeans_hops import getConstraints, normalize, doMutualInformation
from rfml.core import Experiment
from sklearn.cluster import k_means, KMeans


dbg = True
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logger.handlers = []
ch = logging.StreamHandler()
fmt = logging.Formatter("%(thread)d %(levelname)s --"
                        "%(filename)s (line %(lineno)s):%(funcName)s: "
                        "%(message)s")
ch.setFormatter(fmt)
ch.setLevel(logging.DEBUG if dbg else logging.INFO)
logger.addHandler(ch)

n_runs = 25 

class MonteCarlo(Experiment):

    def __init__(self, config, dbg):
        super(MonteCarlo, self).__init__('MonteCarlo', config, dbg)

        self.pandalist = []
    def run(self, trials, errors):
        for t in trials:
            for e in errors:

                hops, key, params = load_h5('PerfectTimeEstimates.h5',['pulseshape','centerfreq'], t, e)
                hops = hops.astype(np.float)
                params = params.tolist()
                cl = getConstraints(hops, params)
                
                discard = []
                discard.append(params.index("endtime"))
                discard.append(params.index("starttime"))

                hops=np.delete(hops, discard,1)
                params = np.delete(params,discard)

                logger.info('hops {}'.format(hops.shape))
                params = params.tolist()

                logger.info('CopK clustering on {} parameters'.format(params))

                hops = normalize(hops, 'param_config.yml', params)

                start = time.time()
                kmeans = KMeans(algorithm = 'full',n_clusters=5, random_state=0, verbose = False, constraints = cl, n_init=n_runs, n_jobs = -1).fit(hops)
                km_labels= kmeans.predict(hops)
                end = time.time()
                logger.info('It took {} to run {} kmeans iterations'.format(end-start, n_runs))

                start = time.time()
                copkmeans = KMeans(algorithm = 'cop',n_clusters=5, random_state=0, verbose = False, constraints = cl, n_init=n_runs, n_jobs = -1).fit(hops)
                cop_labels = copkmeans.cop_predict(hops,0)
                end = time.time()
                logger.info('It took {} to run {} cop iterations'.format(end-start, n_runs))

                logger.info('lengths of unassigned points: {}'.format([len(val) for val in copkmeans.unassigned_]))
                logger.info('inertia of unassigned points: {}'.format([val for val in copkmeans.inertia_]))

                km_score = doMutualInformation(km_labels, key)

                cop_score = doMutualInformation(cop_labels, key)

                result = {'Trial': t, 'Error': e, 'Hops': hops.shape[0], 'KMScore': km_score, 'CopScore': cop_score}
                
                self.pandalist.append(result)
        return(self.pandalist)

r = MonteCarlo(config = {"resultsSavePath": "results/"}, dbg= True)
pandalist = r.run(range(100),range(20))

df = pd.DataFrame(pandalist)
df.to_csv('ClusterResults.csv')






