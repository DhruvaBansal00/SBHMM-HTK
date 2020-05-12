from HMMUtil import HMMUtil
from gait_datareader import GaitDataReader
from adaboost_ensembles import AdaboostEnsembles
import numpy as np
import sys
# read dataset
np.set_printoptions(threshold=sys.maxsize)
gait_reader = GaitDataReader()
gait_reader.read()
temp_arr = np.array(gait_reader.train_data)
print(temp_arr.shape)
# fit hmms
hmm_util_raw_data = HMMUtil()
hmm_util_raw_data.fit(gait_reader.train_data,num_gaussians=1,num_states=3)
for class_index,frames in enumerate(gait_reader.test_data):
    pred_class_indices = hmm_util_raw_data.predict(frames)
    print("Prediction = " + str(pred_class_indices) + " Ground Truth = " + str(class_index))
# construct ensembles
ensembles = AdaboostEnsembles()
numpy_arr = np.array(hmm_util_raw_data.mapping_observation_state)
# print(numpy_arr.shape)
ensembles.fit(hmm_util_raw_data.mapping_observation_state[:,:-1], hmm_util_raw_data.mapping_observation_state[:,-1])
# train HMMs on new features got from the ensembles
new_train_data = []
# print(gait_reader.train_data)
for frames in gait_reader.train_data:
    new_train_data_inner_list = []
    for frame in frames:
        new_train_data_inner_list.append(ensembles.ensemble_scores(frame))
    new_train_data.append(new_train_data_inner_list)

temp_arr = np.array(new_train_data)
print(temp_arr.shape)
hmm_ensemble_feats = HMMUtil()
hmm_ensemble_feats.fit(new_train_data,num_gaussians=1,num_states=3)

for class_index,frames in enumerate(gait_reader.test_data):
    new_test_inner_list = []
    for frame in frames:
        new_test_inner_list.append(ensembles.ensemble_scores(frame))
    #store predictions
    print(np.array(new_test_inner_list).shape)
    pred_class_indices = hmm_ensemble_feats.predict(new_test_inner_list)
    print("Prediction = " + str(pred_class_indices) + " Ground Truth = " + str(class_index))













