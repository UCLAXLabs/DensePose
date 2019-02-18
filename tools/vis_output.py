# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Generate visualizations of inferred poses in a video, given a folder
containing per-frame inference results in joblib .pkl format.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

# PMB extra imports for video processing
import numpy as np
import json
import pycocotools.mask as mask_util
import joblib
from PIL import Image
#from sklearn.externals import joblib

import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.vis as vis_utils
import detectron.utils.keypoints as keypoint_utils

#c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='visualize inference output')
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization files (default: /srv/Top_N_Kpop/inferenceVis)',
        default='/srv/Top_N_Kpop/inferenceVis',
        type=str
    )
    parser.add_argument(
        '--video',
        dest="video",
        help='input video file (needed to get original image for overlays)', 
        default=None,
        type=str
    )
    parser.add_argument(
        '--video-data',
        dest="video_path",
        help='path to folder with sub-folders containing pickled inference output', 
        default=None,
        type=str
    )
    parser.add_argument(
        '--average-frames',
        help='average figure detections in grayscale and output at end',
        action='store_true',
        dest="average_frames",
        default=False
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

# PMB
def mask_to_polygon(mask):
    mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # before opencv 3.2
    # contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,
    #                                                    cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []

    for contour in contours:
        contour = contour.flatten().tolist()
        segmentation.append(contour)
        #if len(contour) > 4:
        #    segmentation.append(contour)

    return segmentation

def main(args):
    logger = logging.getLogger(__name__)
    # Detectron is only able to use 1 GPU for inference
    #cfg.NUM_GPUS = 1
    #assert_and_infer_cfg(cache_urls=False)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if (not os.path.isfile(args.video)):
        print("Unable to find video file",args.video)
        sys.exit(1)

    videoID = ".".join(args.video.split('/')[-1].split('.')[0:-1]).replace('video-', '')

    video_path = os.path.join(args.video_path, videoID)

    if (not os.path.isdir(video_path)):
        print("Unable to find video output files",video_path)
        sys.exit(1)

    out_dir = os.path.join(args.output_dir, videoID)
    if (not os.path.isdir(out_dir)):
        os.mkdir(out_dir)

    cap = cv2.VideoCapture(args.video)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    dataset_keypoints, _ = keypoint_utils.get_keypoints()
    kp_lines = vis_utils.kp_connections(dataset_keypoints)

    targetFramerate = 30
    thresh = .9 # minimum likelihood threshold for a box

    fps = cap.get(cv2.CAP_PROP_FPS)

    if (targetFramerate > fps):
        targetFramerate = fps

    print("Target output frame rate:",targetFramerate)

    skipRatio = int(round(float(fps) / float(targetFramerate)))   

    outputFrameDuration = 1 / float(targetFramerate) # .0333333 ...
    sourceFrameDuration = 1 / float(fps) # .016672224 ...

    #with open(figuresFilename, "w") as figuresFile:
    #    figuresFile.write("[\n")
    #figuresFile.close()

    firstFrame = True

    outputTimecode = 0
    sourceTimecode = 0

    frameCount = 0

    i = 0

    while(cap.isOpened() and (frameCount < total_frames)):
        ret_val, im = cap.read()

        imHeight, imWidth, imChannels = im.shape

        if (firstFrame and args.average_frames):
            average_frame = np.zeros((imWidth,imHeight,3),np.float)

        frameCount += 1

        sourceTimecode += sourceFrameDuration

        frameId = int(round(cap.get(1)))

        if ((frameCount % skipRatio) != 0):

            if ((sourceTimecode - outputTimecode) > outputFrameDuration):
                print("sourceTimecode",sourceTimecode,"outputTimecode",outputTimecode,"outputFrameDuration",outputFrameDuration,"not skipping")
            else:
                print("skipping frame",frameId,"count",frameCount,"skip rato",skipRatio)
                continue

        print("processing frame",frameId,"count",frameCount)
        
        outputTimecode += outputFrameDuration
        
        #fps_time = time.time()
        #im_name = str(fps_time)
        im_name = str(frameId).zfill(4) + "_" + str(outputTimecode) + "_" + videoID
        
        out_path = os.path.join(
            out_dir, '{}'.format(im_name + '.jpg')
        )
        
        datafile_name = video_path + '/' + str(outputTimecode) + "_" + str(frameId) + "_" + videoID + '.joblib'

        logger.info('Processing {}'.format(im_name))
            
        cls_boxes, cls_segms, cls_keyps, cls_bodys = joblib.load(datafile_name)

        i += 1

        figBoxes = []
        figOutlines = []
        figKeypoints = []

        boxes, segms, keyps, classes = vis_utils.convert_from_cls_format(cls_boxes, cls_segms, cls_keyps)

        if ((boxes is not None) and (boxes.shape[0] != 0) and (max(boxes[:, 4]) >= thresh)):
        
            if (len(boxes) != len(segms) != len(keyps)):
                print("WARNING: Different numbers of boxes, segment masks, and keypoint sets")
                print(len(boxes),len(segms), len(keyps))

            for bodyi in range(0,len(boxes)):
                box = boxes[bodyi]
                score = box[-1]
                if (score < thresh):
                    continue

                figBoxes.append([str(box[0]), str(box[1]), str(box[2]), str(box[3])])
                if (len(segms) > bodyi):
                    seg = segms[bodyi]

                    figOutlines.append(str(seg))

                if (len(keyps) > bodyi):
                    keypts = keyps[bodyi]
                    keypXs = keypts[0]
                    keypYs = keypts[1]
                    if (len(keypXs) != len(keypYs)):
                      print("WARNING: different length of keypoint X and Y coord rows:",len(keypXs),len(keypYs))
                   
                    keypointInfo = {}
                    for kpi in range(0, len(keypXs)):
                      kpname = dataset_keypoints[kpi]
                      keypointInfo[kpname] = [str(keypXs[kpi]), str(keypYs[kpi])]

                    figKeypoints.append(keypointInfo)

        #timeFigures = {str(outputTimecode): {'frameID': str(frameId), 'boxes': figBoxes, 'outlines': figOutlines, 'keypoints': figKeypoints}}

        #with open(figuresFilename, "a") as figuresFile:
        #    outStr = json.dumps(timeFigures)
        #    if (not firstFrame):
        #        figuresFile.write("," + outStr + "\n")
        #    else:
        #        figuresFile.write(outStr + "\n")
        #        firstFrame = False
        """
        vis_utils.vis_only_figure(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            out_path,
            cls_boxes,
            cls_segms,
            cls_keyps,
            cls_bodys,
            dataset=dummy_coco_dataset,
            box_alpha=0.5,
            show_class=True
        )
        """
        if (args.average_frames):
          show_class = False
          show_box = False
          show_background = False
          return_grayscale = True
        else:
          show_class = True
          show_box = True
          show_background = True
          return_grayscale = False

        vis_image = vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            out_path,
            boxes,
            segms,
            keyps,
            cls_bodys,
            classes,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=show_class,
            show_box=show_box,
            show_background=show_background,
            thresh=0.7,
            kp_thresh=2,
            grayscale=return_grayscale
        )

        if (args.average_frames):
            average_frame = average_frame + vis_image/total_frames



        #cv2.putText(im,
        #            "FPS: %f" % (1.0 / (time.time() - fps_time)),
        #            (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #            (0, 255, 0), 2)
        #    #cv2.imshow('tf-pose-estimation result', image)
        #cv2.imwrite(out_name, kp_img)

        #fps_time = time.time()
    #cv2.destroyAllWindows()

    cap.release()
    #with open(figuresFilename, "a") as figuresFile:
    #    figuresFile.write("]")

    if (args.average_frames):
        average_frame=numpy.array(numpy.round(average-frames),dtype=numpy.uint8)
        out=Image.fromarray(average_frame,mode="RGB")
        out.save(videoID + "_average.png")
        

if __name__ == '__main__':
    args = parse_args()
    main(args)
