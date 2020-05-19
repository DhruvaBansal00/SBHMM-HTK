import json

import numpy as np


with open('test.json', 'r') as data_file:
    #data = data_file.read()
    data = json.loads(data_file.read())

data = {int(key): value for key, value in data.items()}

hands = np.zeros((len(data), 20))
landmarks = np.zeros((len(data), 126))
faces = []

print(data[2]['landmarks'])

for key, value in sorted(data.items()):
    
    if data[key].get('boxes') is not None:
        n_boxes = len(data[key]['boxes'])
    else:
        n_boxes = 0

    if data[key].get('landmarks') is not None:
        n_landmarks = len(data[key]['landmarks'])
    else:
        n_landmarks = 0
        #print(n_landmarks)

    if n_boxes == 1:
        hand = data[key]['boxes']['0']
        hands[key] = hand * 4
    elif n_boxes == 2:
        hand_0 = data[key]['boxes']['0']
        hand_1 = data[key]['boxes']['1']
        if hand_0[0] < hand_1[0]:
            hands[key][0:5] = hand_0
            hands[key][5:10] = hand_1
        else:
            hands[key][0:5] = hand_1
            hands[key][5:10] = hand_0

        if hand_0[1] < hand_1[1]:
            hands[key][10:15] = hand_0
            hands[key][15:20] = hand_1
        else:
            hands[key][10:15] = hand_1
            hands[key][15:20] = hand_0

    
    if n_landmarks == 1:
        landmark = []
        for landmark_point in data[key]['landmarks']['0']:
            landmark += data[key]['landmarks']['0'][landmark_point]
        landmarks[key] = landmark * 2

    elif n_landmarks == 2:

        landmark_0 = []
        for landmark_point in data[key]['landmarks']['0']:
            landmark_0 += data[key]['landmarks']['0'][landmark_point]
        landmarks[key] = landmark_0 * 2

        landmark_1 = []
        for landmark_point in data[key]['landmarks']['1']:
            landmark_1 += data[key]['landmarks']['1'][landmark_point]
        landmarks[key] = landmark_1 * 2

        if landmark_0[0] < landmark_1[0]:
            landmarks[key] = landmark_0 + landmark_1

        else:
            landmarks[key] = landmark_1 + landmark_0

print(hands)

print(landmarks)
