import numpy as np
import time
import yaml
import sys
import logging
import pandas as pd
sys.path.append('../../../../Code/')
from plot_grid import load_h5, data
from kmeans_hops import getConstraints, normalize, doMutualInformation, getCollisions
from rfml.core import Experiment
from sklearn.cluster import k_means, KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from statistics import mode
from collections import Counter


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

n_runs = 1 
max_iter = 50


def reassignClusterLabels(keylabels,guesslabels):
    guesslabels=np.asarray(guesslabels)
    originalGuessLabels=np.copy(guesslabels)
    numberOfClusters=max(keylabels)+1
    if numberOfClusters != len(set(keylabels)):
        return False
    for clusterNumber in range(numberOfClusters):
        locations = np.where(np.asarray(keylabels)==clusterNumber)
        guessMajority = max(set(list(originalGuessLabels[locations])), key=list(originalGuessLabels[locations]).count) 
        guesslabels[np.where(np.asarray(originalGuessLabels)==guessMajority)[0]]=clusterNumber
    return list(guesslabels)

def getClusterAccuracy(keylabels,guesslabels):
    if not guesslabels:
        return None
    correct = (np.asarray(keylabels)==np.asarray(guesslabels))
    accuracy = correct.sum()/correct.size
    return accuracy

def getCollisionAccuracy(keylabels,guesslabels,collisions):
    if not guesslabels:
        return None
    if not collisions:
        return None
    guesslabels = reassignClusterLabels(keylabels,guesslabels)
    logger.debug("collisions {}".format(collisions))
    flatList=[item for sublist in collisions for item in sublist]
    logger.debug("flatlist {}".format(flatList))
    uniqueElements=set(flatList)
    tally=[]
    collisionAccuracy=0
    for elem in uniqueElements:
        if (keylabels[elem] == guesslabels[elem]):
            tally+=[1]
        else:
            tally+=[0]
    collisionAccuracy=sum(tally)/len(tally)
    return collisionAccuracy

def getSilhouetteCoeff(data,labels):
    silCoeff=silhouette_score(data,labels,metric='euclidean')
    return silCoeff


def getMinClusters(hops,params):
    endind = params.index("endtime")
    startind = params.index("starttime")
    numHops = hops.shape[0]
    overlaps=[]
    for hopidx,hop in enumerate(hops):
        start=hop[startind]
        end=hop[endind]
        #print(np.equal(np.delete(hops,hopidx,0)[:,startind]<start, np.delete(hops,hopidx,0)[:,endind]>start))

        overlaps.append(np.sum(np.equal(np.delete(hops,hopidx,0)[:,startind]<start, np.delete(hops,hopidx,0)[:,endind]>start)))
    return np.max(overlaps)+1

class MonteCarlo(Experiment):

    def __init__(self, config, dbg):
        super(MonteCarlo, self).__init__('MonteCarlo', config, dbg)
        with open('param_config.yml','r') as stream:
            self.param_dict = (yaml.load(stream))


    def run(self, trials, errors,runnum):
        numWritesToFile=0
        ktorun=[3,4,5,6,7,8]
        for kclusters in ktorun:
            for t in trials:
                for e in errors:
                    logger.info("================= On Trial {} and Error {} ==============".format(t,e))
                    max_endtime = self.param_dict['grid']['duration'][2]
                    hops, key, oparams = load_h5('../../../../Code/TimeHops/ZeroError' + str(kclusters)+'Hops.h5',['pulseshape'], t, e) 
                    logger.debug("key {}".format(key))

                    #PercentConstraintsDropped=[0,.25,.50,.75,.99]
                    PercentConstraintsDropped=[0]
                    for ConstraintsDropped in PercentConstraintsDropped:
                        heldoutTrainPoints=[.5]
                        for heldout in heldoutTrainPoints:
                            train_hops = np.copy(hops)
                            train_keys = np.copy(key)
                            params = np.copy(oparams)
                            print(max_endtime*.5)

                            #hop indices that will be held out for testing
                            test_hop_idx = np.where(np.asfarray(hops[:,np.where(params == 'starttime')[0]]) >(max_endtime*(heldout)))[0]
                            test_hop_last50_idx = np.where(np.asfarray(hops[:,np.where(params == 'starttime')[0]]) >(max_endtime*(.5)))[0]
                            logger.debug('held out test hops {}'.format(test_hop_idx))

                            test_hops = hops[test_hop_last50_idx]
                            test_keys = [key[indx] for indx in test_hop_last50_idx]

                            test_hops = test_hops.astype(np.float)
                            train_hops = train_hops.astype(np.float)

                            train_hops=np.delete(train_hops,test_hop_idx,axis=0)
                            train_keys=np.delete(train_keys,test_hop_idx)

                            if (len(set(train_keys))<kclusters) or (len(set(test_keys))<kclusters):
                                break

                            #np.set_printoptions(threshold=sys.maxsize)

                            if (train_hops.shape[0] < kclusters) or (test_hops.shape[0] <kclusters):
                                break

                            sweepMax = int(kclusters+kclusters/2.0)
                            if (train_hops.shape[0] <= sweepMax) or (test_hops.shape[0] <=sweepMax):
                                break

                            logger.debug('hops {} trainh {} testh{}'.format(len(key),len(train_keys),len(test_keys)))
                            
                            params = params.tolist()
                            train_cl,mink = getConstraints(train_hops, params)

                            #Test loop to determing where the extra collision is coming from
                            '''
                            for collisionPair in train_cl:
                                if train_keys[collisionPair[0]] == train_keys[collisionPair[1]]:
                                    print("false collision at hops {} and {} belonging to {} and {} ".format(collisionPair[0],collisionPair[1],train_keys[collisionPair[0]],train_keys[collisionPair[1]]))
                                    print(params)
                                    print("violating hops are {} and {}".format(train_hops[collisionPair[0]],train_hops[collisionPair[1]]))
                                    print(train_hops[collisionPair[0]:collisionPair[1]+1])
                                    sys.exit()
                            '''

                            test_cl,_ = getConstraints(test_hops, params)

                            train_collisions = getCollisions(train_hops, params,train_cl)

                            test_collisions = getCollisions(test_hops, params, test_cl)
                            #TODO plot and check if the collision are correct

                            #logger.debug("train collisions, {} test collisions {}".format(train_collisions, test_collisions))

                            train_deleted = np.random.randint(0,len(train_cl),int(len(train_cl)*ConstraintsDropped))
                            test_deleted = np.random.randint(0,len(test_cl),int(len(test_cl)*ConstraintsDropped))


                            train_cl = np.delete(train_cl,(train_deleted),axis=0)
                            test_cl = np.delete(test_cl,(test_deleted),axis=0)

                            
                            
                            discard = []
                            discard.append(params.index("endtime"))
                            discard.append(params.index("starttime"))
                            discard.append(params.index("centerfreq"))

                            train_hops=np.delete(train_hops, discard,1)
                            test_hops=np.delete(test_hops, discard,1)
                            params = np.delete(params,discard)

                            logger.info('hops {}'.format(train_hops.shape))
                            params = params.tolist()

                            logger.info('CopK clustering on {} parameters'.format(params))
                            train_hops = normalize(train_hops, 'param_config.yml', params)
                            test_hops = normalize(test_hops, 'param_config.yml', params)

                            logger.debug('hops {} trainh {} testh{}'.format(hops.shape,train_hops.shape,test_hops.shape))

                            kmrun=0
                            coplorun=0

                            def execSpectral():
                                spectral=SpectralClustering(n_jobs=-1,n_clusters=kclusters,n_init=n_runs,).fit(train_hops)
                                return spectral
                            #spectral = execSpectral()
                            #spectral.labels_=spectral.labels_[0]

                            
                            def execDBSCAN():
                                dbscan = DBSCAN(n_jobs=-1,).fit(train_hops)
                                return dbscan
                            #dbscan = execDBSCAN()

                            def execAgglom():
                                agglom=AgglomerativeClustering(n_clusters=kclusters).fit(train_hops)
                                return agglom
                            #agglom=execAgglom()

                            # Conventional K-means algorithm used as a baseline comparison 
                            def execKmeans():
                                start = time.time()
                                kmeans = KMeans(algorithm='full',n_clusters=kclusters, verbose=False, constraints=train_cl, n_init=n_runs, n_jobs=-1, max_iter=max_iter).fit(train_hops)
                                end = time.time()
                                kmrun=end-start
                                logger.debug('It took {} to run {} kmeans iterations'.format(end-start, n_runs))
                                return kmeans
                            kmeans = execKmeans()

                            """    
                            kSweepMin = int(kclusters-kclusters/2.0)
                            silCo=[]
                            for numK in range(kSweepMin,sweepMax):
                                def runtk():
                                    tempkmeans = KMeans(algorithm='full', n_clusters=numK, random_state=0, verbose=False, constraints=train_cl, n_init=n_runs, n_jobs=-1, max_iter=max_iter).fit(train_hops)
                                    return tempkmeans
                                tempkmeans=runtk()
                                if max(tempkmeans.labels_) ==0:
                                    continue
                                templabels = tempkmeans.labels_
                                silCo.append(getSilhouetteCoeff(train_hops,templabels))
                            kmGuessK=(np.argmax(silCo)+kSweepMin)
                            """
                            kmGuessK=0
                            
                            #Run cop algorithm and return list of results in terms of minimum inertia
                            def execCopForce():
                                start = time.time()
                                copkmeans = KMeans(algorithm='cop',n_clusters=kclusters, verbose=False, constraints=train_cl, n_init=n_runs, n_jobs = -1, max_iter=max_iter, force_add=True).fit(train_hops)
                                end = time.time()
                                coprun=end-start
                                logger.debug('It took {} to run {} cop iterations'.format(end-start, n_runs))
                                return copkmeans


                            #Run cop but excluding unassigned points
                            def execCoplo():
                                start = time.time()
                                print("CLUSTERING WITH {} CONSTRAINTS".format(len(train_cl)))
                                copleftout = KMeans(algorithm='cop', n_clusters=kclusters, verbose=False, constraints=train_cl,force_add=False, n_init=n_runs, n_jobs=-1, max_iter=max_iter ).fit(train_hops)
                                end=time.time()
                                coplorun=end-start
                                logger.debug('It took {} to run {} cop left out iterations'.format(end-start, n_runs))
                                return copleftout

                            copleftout = execCoplo()

                            """
                            silCo=[]
                            silCoPenalty=[]
                            for numK in range(mink,sweepMax):
                                logger.debug("------------------------see this ---------------")
                                logger.debug("cop Guess k bug")
                                temptrainhops=np.copy(train_hops)
                                def runcop():

                                    tempcopleftout = KMeans(algorithm='cop', n_clusters=numK, random_state=0, verbose=False, constraints=train_cl,force_add=False, n_init=n_runs, n_jobs=-1, max_iter=max_iter ).fit(train_hops)
                                    return tempcopleftout
                                tempcopleftout=runcop()
                                temppoints_unassigned = [len(un) for un in tempcopleftout.unassigned_]
                                print(temppoints_unassigned)

                                if(temppoints_unassigned):
                                    temptrainhops=np.delete(temptrainhops,tempcopleftout.unassigned_[np.argmin(temppoints_unassigned)],axis=0)
                                    templabels = tempcopleftout.labels_[np.argmin(temppoints_unassigned)] 
                                else:
                                    templabels = tempcopleftout.labels_[0] 

                                #if max(tempcopleftout.labels_[np.argmin(temppoints_unassigned)]) ==0:
                                #    continue
                                print("temptrainhops {} templabels {}".format(temptrainhops,templabels))

                                sil= getSilhouetteCoeff(temptrainhops,templabels)
                                silCo.append(sil)
                                if numK ==mink:
                                    silCoPenalty.append(sil)
                                else:
                                    silCoPenalty.append(sil*1.0/float(numK-mink))
                            copGuessK=(np.argmax(silCo)+mink)
                            copGuessKPenalty=(np.argmax(silCoPenalty)+mink)
                            """

                            copGuessK=0
                            copGuessKPenalty=0

                            #logger.info("Kmeans guessed k correctly {}, and cop guessed k correctly {}".format(kmGuessK==kclusters,copGuessK==kclusters))

                            points_unassigned = [len(un) for un in copleftout.unassigned_]
                            min_cop_inertia = np.argmin(copleftout.inertia_) #best result in terms of inertia
                            min_cop_unassigned = np.argmin(points_unassigned) # best result in terms of # points assigned

                            coplo_labels_inertia = copleftout.labels_[min_cop_inertia] #prediction of best inertia run
                            coplo_labels_unassigned= copleftout.labels_[min_cop_unassigned] #prediction of best unassigned run

                            coplo_iter_unassigned = copleftout.n_iter_[min_cop_unassigned] #num iterations best unassigned run
                            coplo_iter_inertia= copleftout.n_iter_[min_cop_inertia] #num iterations best inertia run

                            coplo_inertia_unassigned=copleftout.inertia_[min_cop_unassigned]
                            coplo_inertia_inertia = copleftout.inertia_[min_cop_inertia]

                            coplo_pointunassigned_unassigned = points_unassigned[min_cop_unassigned] #Need to add these to csv
                            coplo_pointunassigned_inertia = points_unassigned[min_cop_inertia]

                            #Held out set labels
                            km_labels_test= kmeans.predict(test_hops)
                            #cop_labels_test_constraints = copkmeans.cop_predict(test_hops,0,test_cl,True) # assign the test hops with constraints
                            #cop_labels_test = copkmeans.cop_predict(test_hops,0)# assign the test hops without constraints

                            coplo_labels_inertia_constraints = copleftout.cop_predict(test_hops,min_cop_inertia,test_cl,True) #prediction of best inertia run with force constraints
                            coplo_labels_unassigned_constraints = copleftout.cop_predict(test_hops,min_cop_unassigned,test_cl,True) #prediction of best unassigned run with force constraints
                    
                            logger.info("true labels {}, predicted labels {}".format(train_keys, kmeans.labels_))


                            # ------------------------------- Train Score ---------------------------------------------
                            #dbscan_train_score=doMutualInformation(dbscan.labels_,train_keys)
                            #spectral_train_score=doMutualInformation(spectral.labels_, train_keys)
                            #agglom_train_score=doMutualInformation(agglom.labels_, train_keys)
                            km_train_score = doMutualInformation(kmeans.labels_, train_keys)
                            km_train_accuracy=getClusterAccuracy(train_keys,reassignClusterLabels(train_keys,kmeans.labels_))
                            #cop_train_score = doMutualInformation(copkmeans.labels_[0], train_keys)

                            copiner_train_score = doMutualInformation(coplo_labels_inertia, np.delete(train_keys,copleftout.unassigned_[min_cop_inertia]))
                            copiner_accuracy=getClusterAccuracy(np.delete(train_keys,copleftout.unassigned_[min_cop_inertia]),reassignClusterLabels(np.delete(train_keys,copleftout.unassigned_[min_cop_inertia]),coplo_labels_inertia))

                            copass_train_score = doMutualInformation(coplo_labels_unassigned, np.delete(train_keys,copleftout.unassigned_[min_cop_unassigned]))
                            copass_accuracy=getClusterAccuracy(np.delete(train_keys,copleftout.unassigned_[min_cop_unassigned]),reassignClusterLabels(np.delete(train_keys,copleftout.unassigned_[min_cop_unassigned]),coplo_labels_unassigned))


                            # ------------------------------- Test Score ---------------------------------------------
                            km_test_score = doMutualInformation(km_labels_test, test_keys)
                            km_test_accuracy=getClusterAccuracy(test_keys,reassignClusterLabels(test_keys,km_labels_test))
                            #cop_test_score_cons = doMutualInformation(cop_labels_test_constraints, test_keys)

                            copiner_test_score_cons = doMutualInformation(coplo_labels_inertia_constraints, test_keys)
                            copiner_test_accuracy=getClusterAccuracy(test_keys,reassignClusterLabels(test_keys,coplo_labels_inertia_constraints))

                            copass_test_score_cons = doMutualInformation(coplo_labels_unassigned_constraints, test_keys)
                            copass_test_accuracy=getClusterAccuracy(test_keys,reassignClusterLabels(test_keys,coplo_labels_unassigned_constraints))

                            # ------------------------------- Collision Accuracy ---------------------------------------------
                            #km_train_collision_accuracy=getCollisionAccuracy(train_keys,reassignClusterLabels(train_keys,kmeans.labels_),train_collisions)
                            km_test_collision_accuracy=getCollisionAccuracy(test_keys,reassignClusterLabels(test_keys,km_labels_test),test_collisions)


                            #copiner_train_collision_accuracy=getCollisionAccuracy(np.delete(train_keys,copleftout.unassigned_[min_cop_inertia]),reassignClusterLabels(np.delete(train_keys,copleftout.unassigned_[min_cop_inertia]),coplo_labels_inertia),train_collisions)
                            copiner_test_collision_accuracy=getCollisionAccuracy(test_keys,reassignClusterLabels(test_keys,coplo_labels_inertia_constraints),test_collisions)

                            #copass_train_collision_accuracy=getCollisionAccuracy(np.delete(train_keys,copleftout.unassigned_[min_cop_unassigned]),reassignClusterLabels(np.delete(train_keys,copleftout.unassigned_[min_cop_unassigned]),coplo_labels_unassigned),train_collisions)
                            copass_test_collision_accuracy=getCollisionAccuracy(test_keys,reassignClusterLabels(test_keys,coplo_labels_unassigned_constraints),test_collisions)

                            logger.info("km, copiner, copunass accuracies, {}, {}, {}".format(km_train_accuracy, copiner_accuracy,copass_accuracy))


                            
                            logger.info('On Trial {} Error {} and cluster {}'.format(t,e,kclusters))

                            result = {'Trial': t, 'Error': e,'PercentDroppped':ConstraintsDropped , 'Heldout':heldout,\
                                    'Hops': hops.shape[0],'k': kclusters,'copNumUnassignedUnassigned':coplo_pointunassigned_unassigned,\
                                                'copNumUnassignedInertia': coplo_pointunassigned_inertia,\
                                                'kmTrainScore':km_train_score,'kmInertia':[kmeans.inertia_],'kmIter':[kmeans.n_iter_],'kmTestScore':km_test_score,\
                                                #'copInertia':copkmeans.inertia_,'copIter':copkmeans.n_iter_,'copRuntime': coprun,\
                                                #'copTestScoreCons':cop_test_score_cons,\
                                                'copInerScore':copiner_train_score,'copInerInertia':coplo_inertia_inertia,'copInerIter':coplo_iter_inertia,\
                                                'copInerTestScoreCons':copiner_test_score_cons,\
                                                'copUnassScore':copass_train_score,'copUnassInertia':coplo_inertia_unassigned,'copUnassIter':coplo_iter_unassigned,\
                                                'copUnassTestScoreCons':copass_test_score_cons,\
                                                'kmRuntime': kmrun, 'coploRuntime':coplorun, 'traincl': len(train_cl),'testcl': len(test_cl),\
                                                'kmTrainAccuracy': km_train_accuracy, 'copinerTrainAccuracy': copiner_accuracy, 'copassTrainAccuracy': copass_accuracy,\
                                                'kmTestAccuracy':km_test_accuracy, 'copinerTestAccuracy':copiner_test_accuracy, 'copassTestAccuracy': copass_test_accuracy,\
                                                'TrainCollisions':len(train_collisions),
                                                #'kmTrainCollisionAccuracy':km_train_collision_accuracy,
                                                'kmTestCollisionAccuracy':km_test_collision_accuracy,
                                                #'copInerCollisionAccuracy':copiner_train_collision_accuracy, 
                                                'copInterTestCollisionAccuracy': copiner_test_collision_accuracy,
                                                #'copUnassCollisionAccuracy':copass_train_collision_accuracy, 
                                                'copUnassTestCollisionAccuracy': copass_test_collision_accuracy,
                                                'copGuessK': copGuessK,
                                                'copGuessKPenalty': copGuessKPenalty,
                                                'kmGuessK': kmGuessK,
                                                'mink':mink,
                                                #'spectralTrainScore':spectral_train_score,
                                                #'agglomTrainScore':agglom_train_score,
                                                #'dbscanTrainScore': dbscan_train_score,

                                                }
                    
                            df=pd.DataFrame(result)
                            header=False
                            if numWritesToFile ==0:
                                header = True
                            numWritesToFile +=1
                            df.to_csv('ZeroTimeErrorRandomIter'+str(runnum)+'.csv',mode='a',header=header)

if __name__ == "__main__":
    for runnum in range(20):
        r = MonteCarlo(config = {"resultsSavePath": "results/"}, dbg= True)
        r.run(range(0,100),range(0,40,5),runnum)


