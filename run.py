import argparse
import logging
import sys
import time

import tf_pose.common as common
import cv2
import numpy as np
import tf_pose.estimator as estimator
from tf_pose.networks import get_graph_path, model_wh
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tf_pose.calculationUtil import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--workout', type=str, default='pullups',
                        help='if provided, appropriate exercise will measured. default=pullups')

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = estimator.TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = estimator.TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # estimate human poses from a single image !
    image = common.read_imgfile(args.image, None, None)
    if image is None:
        print('Image can not be read, path=%s' % args.image)
        sys.exit(-1)
    t = time.time()
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    elapsed = time.time() - t

    print('inference image: %s in %.4f seconds.' % (args.image, elapsed))

    image = estimator.TfPoseEstimator.draw_humans(image, humans, imgcopy=False)



    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.set_title('Result')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # squat2
    # left hip: (188, 152)
    # left knee: (172, 196)
    # left ankle: (162, 264)
    p0 = [3.5, 6.7]
    p1 = [7.9, 8.4]
    p2 = [10.8, 4.8]


    #
    # logger.debug("angle between thigh and lower leg: ")
    # angle2 = calculate_angles((188,152), (172, 196), (162, 264))
    # logger.debug(np.degrees(angle2))

    bodypartpoints = getBodyPartPoints(image, humans, imgcopy=False)

    if(args.workout == 'pullups'):
        print("Calculate pullup arm angle: ")
        angle1 = calculate_angles(bodypartpoints["left wrist"], bodypartpoints["left elbow"], bodypartpoints["left shoulder"])
        print(np.degrees(angle1))
    elif(args.workout == 'squats'):
        print("Calculate squat angle: ")
        angle1 = calculate_angles(bodypartpoints["left hip"], bodypartpoints["left knee"],
                                  bodypartpoints["left ankle"])
        print(np.degrees(angle1))

        # a = fig.add_subplot(2, 2, 2)
    # a.set_title('Original Image')
    # plt.imshow(cv2.cvtColor(common.read_imgfile(args.image, None, None), cv2.COLOR_BGR2RGB))

    # bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    # bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)
    #
    # # show network output
    # # a = fig.add_subplot(2, 2, 2)
    # # plt.imshow(bgimg, alpha=0.5)
    # tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
    # # plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    # plt.colorbar()
    #
    # tmp2 = e.pafMat.transpose((2, 0, 1))
    # tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    # tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)
    #
    # a = fig.add_subplot(2, 2, 3)
    # a.set_title('Vectormap-x')
    # # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    # # plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
    # plt.colorbar()
    #
    # a = fig.add_subplot(2, 2, 4)
    # a.set_title('Vectormap-y')
    # # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    # # plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()
    plt.show()
