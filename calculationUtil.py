import numpy as np
import logging
from tf_pose import common



#     Nose = 0
#     Neck = 1
#     RShoulder = 2
#     RElbow = 3
#     RWrist = 4
#     LShoulder = 5
#     LElbow = 6
#     LWrist = 7
#     RHip = 8
#     RKnee = 9
#     RAnkle = 10
#     LHip = 11
#     LKnee = 12
#     LAnkle = 13
#     REye = 14
#     LEye = 15
#     REar = 16
#     LEar = 17

def getBodyPartString(num):
    switcher = {
        0: "nose",
        1: "neck",
        2: "right shoulder",
        3: "right elbow",
        4: "right wrist",
        5: "left shoulder",
        6: "left elbow",
        7: "left wrist",
        8: "right hip",
        9: "right knee",
        10: "right ankle",
        11: "left hip",
        12: "left knee",
        13: "left ankle",
        14: "right eye",
        15: "left eye",
        16: "right ear",
        17: "left ear"
    }
    return switcher.get(num, "invalid body part")

#squat2
#left hip: (188, 152)
#left knee: (172, 196)
#left ankle: (162, 264)

def calculate_angles(point1, point2, point3):
    vector1 = np.array(point1) - np.array(point2)
    vector2 = np.array(point3) - np.array(point2)
    angle = np.math.atan2(np.linalg.det([vector1, vector2]), np.dot(vector1, vector2))
    return angle


def getBodyPartPoints(npimg, humans, imgcopy=False):
    if imgcopy:
        npimg = np.copy(npimg)
    image_h, image_w = npimg.shape[:2]
    centers = {}
    bodyPoints = {}
    #draw point
    for i in range(common.CocoPart.Background.value):
        if i not in humans[0].body_parts.keys():
            continue
        body_part = humans[0].body_parts[i]

        center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
        centers[i] = center

        bodyPoints[getBodyPartString(i)] = center


    return bodyPoints