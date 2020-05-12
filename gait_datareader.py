import numpy as np
import os

class GaitDataReader:

    def read(self):
        # directory = 'C:\\Users\\vishvak murahari\\Documents\\SegmentedHMMs\\GaitData'
        directory = 'gaitData'
        # read the first 5 subjects
        folders = ['sub 1','sub 2','sub 3','sub 4','sub 5']
        # each folder represents a class of data
        # each text file in the folders is a possible sequence
        self.train_data= []
        self.test_data = []
        for folder in folders:
            train_data_inner_list = []
            test_data_inner_list = []
            inner_dir = os.path.join(directory,folder)
            # choose 5 files for each speed -- we have 4 speeds
            speeds = ["0-7","1-0","1-3","1-6"]
            for i,speed in enumerate(speeds):
                train_file_endings = np.random.choice(np.arange(9),size = (5,),replace=False)
                test_file_endings =  list(set(np.arange(9)) - set(train_file_endings))
                for train_ending in np.nditer(train_file_endings):
                    train_data_inner_list.append(self.get_data_from_file(train_ending,inner_dir,folder,speed))
                for test_ending in test_file_endings:
                    test_data_inner_list.append(self.get_data_from_file(test_ending,inner_dir,folder,speed))
            # append to test data
            self.train_data.append(train_data_inner_list)
            self.test_data.append(test_data_inner_list)

    def get_data_from_file(self,fileindex,inner_dir,folder,speed):
            session = (fileindex % 3) + 1
            trial = int(fileindex / 3) + 1
            # reading file
            raw_data = np.genfromtxt(os.path.join(inner_dir,folder + " " + str(session) + " " + speed + " " + str(trial) + ".txt"),skip_header=1)
            # delete the first two rows and covert it to a 6 dimensional feature vector by calculating angles
            raw_data = raw_data[:,2:]

            left_bwt = raw_data[:,6:9]
            left_knee = raw_data[:,21:24]
            left_fwt = raw_data[:,15:18]
            left_ank = raw_data[:,0:3]
            left_mt5 = raw_data[:,24:27]

            right_bwt = raw_data[:,39:42]
            right_knee = raw_data[:,54:57]
            right_fwt = raw_data[:,48:51]
            right_ank = raw_data[:,33:36]
            right_mt5 = raw_data[:,57:60]
            left_torso_femur = np.arccos(np.sum((left_fwt - left_bwt) * (left_bwt - left_knee),axis=1) /
                                          (np.linalg.norm(left_fwt - left_bwt, axis=1) * np.linalg.norm(left_bwt - left_knee,axis=1)))
            left_femur_tibia = np.arccos(np.sum((left_bwt - left_knee) * (left_knee - left_ank),axis=1) /
                                          (np.linalg.norm(left_bwt - left_knee, axis=1) * np.linalg.norm(left_knee - left_ank,axis=1)))
            left_tibia_foot = np.arccos(np.sum((left_knee - left_ank) * (left_ank - left_mt5),axis=1) /
                                          (np.linalg.norm(left_knee - left_ank, axis=1) * np.linalg.norm(left_ank - left_mt5,axis=1)))

            right_torso_femur = np.arccos(np.sum((right_fwt - right_bwt) * (right_bwt - right_knee),axis=1) /
                                          (np.linalg.norm(right_fwt - right_bwt, axis=1) * np.linalg.norm(right_bwt - right_knee,axis=1)))
            right_femur_tibia = np.arccos(np.sum((right_bwt - right_knee) * (right_knee - right_ank),axis=1) /
                                          (np.linalg.norm(right_bwt - right_knee, axis=1) * np.linalg.norm(right_knee - right_ank,axis=1)))
            right_tibia_foot = np.arccos(np.sum((right_knee - right_ank) * (right_ank - right_mt5),axis=1) /
                                          (np.linalg.norm(right_knee - right_ank, axis=1) * np.linalg.norm(right_ank - right_mt5,axis=1)))

            new_data_feats = np.column_stack((left_torso_femur,left_femur_tibia,left_tibia_foot,
                                         right_torso_femur,right_femur_tibia,right_tibia_foot))
            # getting the starting point of the sequence
            length = 80
            start =  int(np.random.uniform(0,new_data_feats.shape[0]-length))
            # TODO change this to adhere to the paper by adding variance
            final_data = new_data_feats[start:(start+length),:]
            return final_data

reader = GaitDataReader()
reader.read()
