import numpy as np
import time
import yaml
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

n_runs = 20 
max_iter = 50

class MonteCarlo(Experiment):

    def __init__(self, config, dbg):
        super(MonteCarlo, self).__init__('MonteCarlo', config, dbg)
        
        with open('param_config.yml','r') as stream:
            self.param_dict = (yaml.load(stream))
        print(self.param_dict['grid']['duration'][2])


    def run(self, trials, errors):
        numWritesToFile=0
        for kclusters in range(3,11,2):
            for t in trials:
                for e in errors:
                    
                    max_endtime = self.param_dict['grid']['duration'][2]
                    hops, key, oparams = load_h5('PerfectTimeEstimates' + str(kclusters)+'.h5',['pulseshape','centerfreq'], t, e) 

                    #Exit loop if hops count is very large due to computation time.
                    if len(key) >800:
                        break
                    constraintVector=[0,5,10,20,40,80,120,160,200]
                    for numConstraints in constraintVector:
                        heldout=.05
                        while heldout < .6:

                            train_hops = np.copy(hops)
                            train_keys = np.copy(key)
                            params = np.copy(oparams)
                            

                            #hop indices that will be held out for testing
                            test_hop_idx = np.where(np.asfarray(hops[:,np.where(params == 'starttime')[0]]) >(max_endtime*(1-heldout)))[0]
                            logger.debug('held out test hops {}'.format(test_hop_idx))

                            test_hops = hops[test_hop_idx]
                            test_keys = [key[indx] for indx in test_hop_idx]

                            test_hops = test_hops.astype(np.float)
                            train_hops = train_hops.astype(np.float)

                            train_hops=np.delete(train_hops,test_hop_idx,axis=0)
                            train_keys=np.delete(train_keys,test_hop_idx)


                            logger.debug('hops {} trainh {} testh{}'.format(hops.shape,train_hops.shape,test_hops.shape))
                            logger.debug('hops {} trainh {} testh{}'.format(len(key),len(train_keys),len(test_keys)))
                            
                            params = params.tolist()
                            train_cl = getConstraints(train_hops, params)
                            test_cl = getConstraints(test_hops, params)

                            if (len(train_cl)<numConstraints) or (len(test_cl)<numConstraints):
                                break
                            train_deleted = np.random.randint(0,len(train_cl),len(train_cl)-numConstraints)
                            test_deleted = np.random.randint(0,len(test_cl),len(test_cl)-numConstraints)

                            train_cl = np.delete(train_cl,(train_deleted),axis=0)
                            test_cl = np.delete(test_cl,(test_deleted),axis=0)

                            logger.debug('deleted constraints are {}'.format(train_deleted))
                            
                            
                            discard = []
                            discard.append(params.index("endtime"))
                            discard.append(params.index("starttime"))

                            train_hops=np.delete(train_hops, discard,1)
                            test_hops=np.delete(test_hops, discard,1)
                            params = np.delete(params,discard)

                            logger.info('hops {}'.format(train_hops.shape))
                            params = params.tolist()

                            logger.info('CopK clustering on {} parameters'.format(params))
                            train_hops = normalize(train_hops, 'param_config.yml', params)
                            test_hops = normalize(test_hops, 'param_config.yml', params)

                            logger.debug('hops {} trainh {} testh{}'.format(hops.shape,train_hops.shape,test_hops.shape))
                            # Conventional K-means algorithm used as a baseline comparison 
                            start = time.time()
                            kmeans = KMeans(algorithm='full',n_clusters=kclusters, random_state=0, verbose=False, constraints=train_cl, n_init=n_runs, n_jobs=-1, max_iter=max_iter).fit(train_hops)
                            km_labels = kmeans.labels_
                            km_inertia = kmeans.inertia_
                            km_iter = kmeans.n_iter_
                            end = time.time()
                            kmrun=end-start
                            logger.debug('It took {} to run {} kmeans iterations'.format(end-start, n_runs))
                            
                            
                            start = time.time()
                            #Run cop algorithm and return list of results in terms of minimum inertia
                            copkmeans = KMeans(algorithm='cop',n_clusters=kclusters, random_state=0,  verbose=False, constraints=train_cl, n_init=n_runs, n_jobs = -1, max_iter=max_iter, force_add=True).fit(train_hops)
                            cop_labels = copkmeans.labels_
                            cop_iter = copkmeans.n_iter_
                            cop_inertia = copkmeans.inertia_
                            end = time.time()
                            coprun=end-start
                            logger.debug('It took {} to run {} cop iterations'.format(end-start, n_runs))


                            #Run cop but excluding unassigned points
                            start = time.time()
                            copleftout = KMeans(algorithm='cop', n_clusters=kclusters, random_state=0, verbose=False, constraints=train_cl,force_add=False, n_init=n_runs, n_jobs=-1, max_iter=max_iter ).fit(train_hops)
                            copleftout_unassigned = copleftout.unassigned_
                            copleftout_inertia = copleftout.inertia_
                            end=time.time()
                            coplorun=end-start

                            logger.debug('It took {} to run {} cop left out iterations'.format(end-start, n_runs))

                            points_unassigned = [len(un) for un in copleftout_unassigned]
                            min_cop_inertia = np.argmin(copleftout_inertia) #best result in terms of inertia
                            min_cop_unassigned = np.argmin(points_unassigned) # best result in terms of # points assigned

                            coplo_labels_inertia = copleftout.labels_[min_cop_inertia] #prediction of best inertia run
                            coplo_labels_unassigned= copleftout.labels_[min_cop_unassigned] #prediction of best unassigned run

                            coplo_iter_unassigned = copleftout.n_iter_[min_cop_unassigned] #num iterations best unassigned run
                            coplo_iter_inertia= copleftout.n_iter_[min_cop_inertia] #num iterations best inertia run

                            coplo_inertia_unassigned=copleftout_inertia[min_cop_unassigned]
                            coplo_inertia_inertia = copleftout_inertia[min_cop_inertia]

                            coplo_pointunassigned_unassigned = points_unassigned[min_cop_unassigned] #Need to add these to csv
                            coplo_pointunassigned_inertia = points_unassigned[min_cop_inertia]


                            
                            #Held out set labels
                            km_labels_test= kmeans.predict(test_hops)
                            cop_labels_test_constraints = copkmeans.cop_predict(test_hops,0,test_cl,True) # assign the test hops with constraints
                            cop_labels_test = copkmeans.cop_predict(test_hops,0)# assign the test hops without constraints

                            coplo_labels_inertia_constraints = copleftout.cop_predict(test_hops,min_cop_inertia,test_cl,True) #prediction of best inertia run with force constraints
                            coplo_labels_unassigned_constraints = copleftout.cop_predict(test_hops,min_cop_unassigned,test_cl,True) #prediction of best unassigned run with force constraints

                            #coplo_labels_inertia_none = copleftout.cop_predict(test_hops,min_cop_inertia) #prediction of best inertia run without force constraints
                            #coplo_labels_unassigned_none= copleftout.cop_predict(test_hops,min_cop_unassigned) #prediction of best unassigned run without force constraints
                    
                            km_train_score = doMutualInformation(km_labels, train_keys)

                            cop_train_score = doMutualInformation(cop_labels[0], train_keys)

                            copiner_train_score = doMutualInformation(coplo_labels_inertia, np.delete(train_keys,copleftout_unassigned[min_cop_inertia]))

                            copass_train_score = doMutualInformation(coplo_labels_unassigned, np.delete(train_keys,copleftout_unassigned[min_cop_unassigned]))


                            km_test_score = doMutualInformation(km_labels_test, test_keys)
                            cop_test_score_cons = doMutualInformation(cop_labels_test_constraints, test_keys)
                            copiner_test_score_cons = doMutualInformation(coplo_labels_inertia_constraints, test_keys)
                            copass_test_score_cons = doMutualInformation(coplo_labels_unassigned_constraints, test_keys)

                            #cop_test_score_nocons = doMutualInformation(cop_labels_test, test_keys)
                            #copiner_test_score_nocons = doMutualInformation(coplo_labels_inertia_none, test_keys)
                            #copass_test_score_nocons = doMutualInformation(coplo_labels_unassigned_none, test_keys)
                            
                            logger.info('On Trial {} Error {} and cluster {}'.format(t,e,kclusters))

                            result = {'Trial': t, 'Error': e,'NumConstraints': numConstraints, 'Heldout':heldout,\
                                    'Hops': hops.shape[0],'k': kclusters, 'copNumUnassignedUnassigned':coplo_pointunassigned_unassigned,\
                                    'copNumUnassignedInertia': coplo_pointunassigned_inertia,\
                                    'kmTrainScore':km_train_score,'kmInertia':km_inertia,'kmIter':km_iter,'kmTestScore':km_test_score,\
                                    'copTrainScore':cop_train_score,'copInertia':cop_inertia,'copIter':cop_iter,\
                                    'copTestScoreCons':cop_test_score_cons,\
                                    'copInerScore':copiner_train_score,'copInerInertia':coplo_inertia_inertia,'copInerIter':coplo_iter_inertia,\
                                    'copInerTestScoreCons':copiner_test_score_cons,\
                                    'copUnassScore':copass_train_score,'copUnassInertia':coplo_inertia_unassigned,'copUnassIter':coplo_iter_unassigned,\
                                    'copUnassTestScoreCons':copass_test_score_cons,\
                                    'kmRuntime': kmrun, 'copRuntime': coprun,'coploRuntime':coplorun, 'traincl': len(train_cl),'testcl': len(test_cl) }
                    
                            df=pd.DataFrame(result)
                            header=False
                            if numWritesToFile ==0:
                                header = True
                            numWritesToFile +=1
                            df.to_csv('FinalClusterResultsLessConstraints.csv',mode='a',header=header)
                            heldout=heldout+1.2*heldout

r = MonteCarlo(config = {"resultsSavePath": "results/"}, dbg= True)
#r.run(range(25),range(0,20,2))
r.run(range(10),range(0,40,3))


