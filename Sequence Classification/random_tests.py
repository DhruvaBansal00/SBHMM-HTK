from HMMUtil import HMMUtil
from gait_datareader import GaitDataReader
from adaboost_ensembles import AdaboostEnsembles
# read dataset
gait_reader = GaitDataReader()
gait_reader.read()
# fit hmms
hmm_util_raw_data = HMMUtil()
hmm_util_raw_data.fit(gait_reader.train_data,num_gaussians=1,num_states=3)
for class_index,frames in enumerate(gait_reader.test_data):
    pred_class_indices = hmm_util_raw_data.predict(frames)
    print("Prediction = " + str(pred_class_indices) + " Ground Truth = " + str(class_index))


