import os
import glob
import argparse
import numpy as np
import sys
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.use('TkAgg')
import json
import cv2
from filterpy.kalman import KalmanFilter
import math
np.random.seed(0)
import pandas as pd

from feature_selection import select_features

def _load_json(json_file):
    with open(json_file, 'r') as data_file:
        data = json.loads(data_file.read())
    return data


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou(bb_test, bb_gt):
    """
    Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """

    if 0 in bb_test or 0 in bb_gt: return -1
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return(o)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
    """
    
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    #print(w, h, x, y)
    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """

    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

'''
def flatten_cluster(cluster):

    # flatten cluster of points


    return np.array(cluster).flatten()

def produce_cluster(cluster):

    # unflatten list of point coords

    
    return np.array(cluster).reshape((len(cluster) / 2, 2))
'''

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0
    def __init__(self,bbox):
        """
        Initialises a tracker using initial bounding box.
        """

        #define constant velocity model
        self.kf = KalmanFilter(dim_x=10, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0,.001,0,0],[0,1,0,0,0,1,0,0,.001,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,.001,0,0],[0,0,0,0,0,1,0,0,.001,0],[0,0,0,0,0,0,1,0,0,0],
        [0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """

        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """

        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
            self.kf.predict()
            self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """

        return convert_x_to_bbox(self.kf.x)


class KalmanClusterTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as point clusters.
    """

    count = 0
    def __init__(self,cluster):
        """
        Initialises a tracker using initial bounding box.
        """

        #define constant velocity model
        dim_x = 3*len(cluster)
        dim_z = len(cluster)
        self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.kf.F = np.eye(dim_x) + np.array([[int(x==y+3) for x in range(dim_x)] for y in range(dim_x)]) 
        + np.array([[int(x==y+6) for x in range(dim_x)] for y in range(dim_x)])
        self.kf.H = np.array([[int(x==y) for x in range(dim_x)] for y in range(dim_z)])

        self.kf.P[len(cluster):,len(cluster):] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P[len(cluster)*2:,len(cluster)*2:] *= 100000. #give very high uncertainty to the unobservable accelerations
        self.kf.P *= 10.
        self.kf.Q[len(cluster):,len(cluster):] *= 0.01

        self.time_since_update = 0
        self.id = KalmanClusterTracker.count
        KalmanClusterTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.cluster = cluster

    def update(self):
        """
        Updates the state vector with observed cluster.
        """

        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.kf.x[:len(self.cluster)])

    def predict(self):
        """
        Advances the state vector and returns the predicted cluster estimate.
        """

        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def get_state(self):
        """
        Returns the current cluster estimate.
        """

        return self.kf.x


def associate_detections_to_boxes(detections,trackers,iou_threshold = 0.4):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """

    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            iou_matrix[d,t] = iou(det,trk)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

'''
def associate_detections_to_points(detections, trackers):

    # Assigns detections to tracked face, each represented as cluster of points

    # Returns list of matched trackers

    print(detections[0])
    print(trackers[0])
    matched_list = []
    for det in detections:
        min = float("inf")
        for trk in trackers:
            dist = (trk[0] - det[0])**2 + (trk[1] - det[1])**2
            if dist < min: min = trk

        if math.isinf(min): matched_list.append(0)
        else: matched_list.append(min)

    return matched_list
'''

class BoxKF(object):
    def __init__(self, max_age=5, min_hits=3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        
        self.frame_count += 1

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_boxes(dets,trks)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]][:])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i][:])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]

            if (trk.time_since_update <= self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)

        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,5))

'''
class ClusterKF(object):

    # This class represents internal state of point objects being tracked by filter.


    def __init__(self, max_age=3, feature='landmarks', cluster_len=42):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.trackers = []
        self.frame_count = 0
        self.feature = feature
        self.cluster_len = cluster_len

    def update(self, dets=np.empty((0, 0))):
        """
        Params:
        dets - a numpy array of detections in the format [[[x1,y1],[x1,y1],...][[x2,y2],[x2,y2],...]]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """

        if dets == np.empty((0,0)): dets = np.empty((0, self.cluster_len))
        
        self.frame_count += 1
        cluster_len = 0
        trks = np.zeros((len(self.trackers), cluster_len))

        # get predicted locations from existing trackers.
        if self.feature == 'faces':
            trks = np.zeros((len(self.trackers), 12))
            cluster_len = 6
        if self.feature == 'landmarks':
            trks = np.zeros((len(self.trackers), 42))
            cluster_len = 42
        to_del = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[:cluster_len]]

        matched = associate_detections_to_points(dets,trks)

        # update matched trackers with assigned detections
        print(matched)
        self.trackers.append(KalmanClusterTracker(matched))
        
        print(self.trackers)
        for tracker in self.trackers:
            tracker.update()

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1

            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.remove(trk)

        return self.trackers[0]
'''

def load_data(data_file):
    curr_data = _load_json(data_file)

    if not curr_data:
        return None
    curr_data = {int(key): value for key, value in curr_data.items()}

    return curr_data


def kalman_box_tracking(features_filepath):
    data_all = load_data(features_filepath)

    frames = len(data_all)

    dict_list = []
    for frame in range(frames):
        dict_list.append(data_all[frame]['boxes'])

    data = []
    for frame in dict_list:
        data.append(list(frame.values()))

    kalman_box_data = [[[0 for i in range(4)] for j in range(2)] for k in range(frames)]

    tracker = BoxKF()
    for frame in range(frames):
        dets = data[frame]
        for det in dets:
            if det[0] == 0.: dets.remove(det)
            if len(dets) == 0: dets.append(data[frame - 1][0])
        dets[:][2:4] += dets[:][0:2] #convert [x1,y1,w,h] to [x1,y1,x2,y2]
        trackers = tracker.update(dets) # new data output
        for d in trackers:
            kalman_box_data[frame].append([d[0],d[1],d[2]-d[0],d[3]-d[1]])
    return kalman_box_data

'''
def kalman_face_tracking(features_filepath):
    data_all = load_data(features_filepath)

    frames = len(data_all)

    dict_list = []
    for frame in range(frames):
        dict_list.append(data_all[frame]['faces'])

    data = []
    for frame in dict_list:
        face_list = list(frame.values())
        for face in face_list:
            data.append([list(face.values())])

    kalman_face_data = [[[0 for i in range(2)] for j in range(6)] for k in range(frames)]

    tracker = ClusterKF(feature='faces', cluster_len=12)
    for frame in range(frames):
        dets = data[frame]
        dets = flatten_cluster(dets)
        d = tracker.update(dets) # new data output
        kalman_face_data[frame].append([d.get_state()[:12].reshape((6, 2))])
    
    return kalman_face_data

def kalman_landmark_tracking(features_filepath):
    data_all = load_data(features_filepath)

    frames = len(data_all)

    dict_list = []
    for frame in range(frames):
        dict_list.append(data_all[frame]['landmarks'])

    data = []
    for frame in dict_list:
        data.append(list(frame.values()))

    kalman_landmark_data = [[[[0 for i in range(3)] for j in range(21)] for k in range(2)] for l in range(frames)]

    tracker_1 = ClusterKF(feature='landmarks', cluster_len=42)
    tracker_2 = ClusterKF(feature='landmarks', cluster_len=42)
    for frame in range(frames):
        dets = data[frame]
        dets_1 = flatten_cluster(dets[0])
        dets_2 = flatten_cluster(dets[1])
        trackers_1 = tracker_1.update(dets_1) # new data output
        trackers_2 = tracker_2.update(dets_2) # new data output
        for i, d in enumerate(trackers_1):
            kalman_landmark_data[frame].append([[d.reshape((21, 2))][trackers_2[i].reshape((21, 2))][np.zeros(21)]])
    
    return kalman_landmark_data
'''

def kalman_feature_data(features_filepath, features, drop_na: bool = True):
    
    unordered_hands = kalman_box_tracking(features_filepath)

    # face = kalman_face_tracking(features_filepath)

    # ordered_landmarks = kalman_landmark_tracking(features_filepath)
    
    num_frames = len(unordered_hands)

    # Create Dataframe from data
    hand_0 = []
    hand_1 = []
    for i in range(num_frames):
        frame_hand_data_0 = unordered_hands[i][0]
        frame_hand_data_1 = unordered_hands[i][1]
        hand_0.append(frame_hand_data_0)
        hand_1.append(frame_hand_data_1)

    hand_0 = np.array(hand_0)
    hand_1 = np.array(hand_1)

    landmarks_0 = []
    landmarks_1 = []

    for i in range(num_frames):
        frame_landmark_data_0 = [[0 for i in range(3)] for j in range(21)]
        frame_landmark_data_1 = [[0 for i in range(3)] for j in range(21)]
        data_0 = []
        data_1 = []
        for j in range(0, 21):
            data_0.append(frame_landmark_data_0[j][0])
            data_0.append(frame_landmark_data_0[j][1])
            data_1.append(frame_landmark_data_1[j][0])
            data_1.append(frame_landmark_data_1[j][1])
        landmarks_0.append(data_0)
        landmarks_1.append(data_1)

    landmarks_0 = np.array(landmarks_0)
    landmarks_1 = np.array(landmarks_1)

    face_0 = []
    for i in range(num_frames):
        frame_face_data_0 = [[0 for i in range(2)] for j in range(6)]
        data_0 = []
        for j in range(0, 6):
            data_0.append(frame_face_data_0[j][0])
            data_0.append(frame_face_data_0[j][1])
        face_0.append(data_0)
    face_0 = np.array(face_0)

    hands_ = ['left_hand', 'right_hand']
    coordinates = ['x', 'y', 'w', 'h']
    hand_cols = [f'{hand}_{coordinate}' 
                for hand 
                in hands_ 
                for coordinate 
                in coordinates]

    hands_ = ['left', 'right']
    landmarks_ = ['landmark_{}'.format(i) for i in range(21)]
    coordinates = ['x', 'y']
    landmark_cols = ['{}_{}_{}'.format(hand, landmark, coordinate) 
                    for hand 
                    in hands_ 
                    for landmark 
                    in landmarks_ 
                    for coordinate 
                    in coordinates]


    faces_ = ['face_{}'.format(i) for i in range(6)]
    coordinates = ['x', 'y']
    face_cols = ['{}_{}'.format(face, coordinate)
                for face
                in faces_
                for coordinate
                in coordinates]

    cols = hand_cols + landmark_cols + face_cols

    all_features = np.concatenate([hand_0, hand_1, landmarks_0, landmarks_1, face_0], axis=1)

    df = pd.DataFrame(all_features, columns=cols)

    if drop_na: df = df.dropna(axis=0)

    # print("Kalman DataFrame: ")
    # print(df)

    return df



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', default = '/home/thad/Desktop/AndroidCaptureApp/DATA/Prerna_04_07_20/alligator_in_box/1582398952685')
    parser.add_argument('--features_filepath', default = '/home/thad/Desktop/AndroidCaptureApp/mp_feats_20-03-25_prerna/alligator_in_box/1582398952685/Prerna.alligator_in_box.1582398952685.data')
    parser.add_argument('--save_dir', default='/home/thad/copycat/copycat-ml/main/projects/prerna_20-03-25/visualization/test_alligator_in_box/1582398952685')
    parser.add_argument('--features', default=[])
    parser.add_argument('--table_video', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print("Making Directory ", args.save_dir)
    
    kalman_feature_data(args.features_filepath, args.features)



