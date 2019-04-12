# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
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

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils
import detectron.utils.keypoints as keypoint_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

save_work = False
vis_figures = True
# 132556_4422.95186666
start_time_in_secs = 0 # Number of seconds to skip at start of video
start_frame = 0 # Start ON this frame, not after

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization files (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--video',
        dest="video",
        help='video file', 
        default=None,
        type=str
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
    merge_cfg_from_file(args.cfg)
    # Detectron is only able to use 1 GPU for inference
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()
    model = infer_engine.initialize_model_from_cfg(args.weights)

    if (not os.path.isfile(args.video)):
        print("Unable to find video",args.video)
        sys.exit(1)

    videofile = args.video
    videoID = os.path.basename(videofile).split('.')[0].replace('video-','')

    if (save_work and (not os.path.isdir('DensePoseData/figuresRawOutput/' + videoID))):
        os.mkdir('DensePoseData/figuresRawOutput/' + videoID)

    cap = cv2.VideoCapture(videofile)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    dataset_keypoints, _ = keypoint_utils.get_keypoints()
    kp_lines = vis_utils.kp_connections(dataset_keypoints)

    figuresFilename = 'figuresJSON/' + '.'.join(os.path.basename(videofile).replace('video-','').split('.')[0:-1]) + "_figures.json"
    print("figures JSON file is",figuresFilename)

    targetFramerate = 30
    thresh = .9 # minimum likelihood threshold for a box

    fps = cap.get(cv2.CAP_PROP_FPS)

    if (targetFramerate > fps):
        targetFramerate = fps

    print("Target output frame rate:",targetFramerate)

    skipRatio = int(round(float(fps) / float(targetFramerate)))   

    outputFrameDuration = 1 / float(targetFramerate) # .0333333 ...
    sourceFrameDuration = 1 / float(fps) # .016672224 ...

    firstFrame = False
    if (not os.path.exists(figuresFilename)):
        with open(figuresFilename, "w") as figuresFile:
            figuresFile.write("[\n")
        firstFrame = True

    outputTimecode = 0
    sourceTimecode = 0

    frameCount = 0

    i = 0

    in_retry = False

    while(cap.isOpened() and (frameCount < total_frames)):
        ret_val, im = cap.read()

        if (im is None):
            in_retry = True
            ret_val, im = cap.read()
            if (im is None):
                print("Consecutive failures to retrieve frame, quitting.")
                break
            print("Missed a frame, continuing...")
            in_retry = False

        imHeight, imWidth, imChannels = im.shape

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
        
        if (outputTimecode <= start_time_in_secs):
            continue

        if (frameCount < start_frame):
            continue

        #fps_time = time.time()
        #im_name = str(fps_time)
        im_name = str(frameId).zfill(4) + "_" + str(outputTimecode) + "_" + videoID
        
        out_path = os.path.join(
            args.output_dir, '{}'.format(im_name + '.jpg')
        )

        datafile_name = 'DensePoseData/figuresRawOutput/' + videoID + '/' + str(outputTimecode) + "_" + str(frameId) + "_" + videoID + '.joblib'

        logger.info('Processing {}'.format(im_name))
        
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(model, im, None, timers=timers)

        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )
        i += 1

        if (save_work):
            joblib.dump([cls_boxes, cls_segms, cls_keyps, cls_bodys], datafile_name, compress=True)

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

                    # PMB This code converts the segmentation mask into a
                    # regular (bitmap) mask, and then converts the mask into
                    # a polygon that "lassos" (or outlines) the region -- but
                    # both of these representations are rather cumbersome.
                    # This is why COCO uses the weird segmentation mask format
                    # (technically it's "compressed RLE") in the first place.
                    # So we'll keep using it for now.

                    #segHeight = int(seg['size'][0])
                    #segWidth = int(seg['size'][1])
                    #segMask = mask_util.decode(seg)

                    #segPolygon = mask_to_polygon(segMask.copy())[0]
                    #thisPolygon = []
                    #for a,b in zip(segPolygon[0::2], segPolygon[1::2]):
                    #    thisPolygon.append([str(a), str(b)])
                    #figOutlines.append(thisPolygon)
                    
                    figOutlines.append(str(seg))

                if (len(keyps) > bodyi):
                    keypts = keyps[bodyi]
                    print("Keypoints shape is",str(keypts.shape))
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
        timeFigures = {str(outputTimecode): {'frameID': str(frameId), 'boxes': figBoxes, 'keypoints': figKeypoints}}

        with open(figuresFilename, "a") as figuresFile:
            outStr = json.dumps(timeFigures)
            if (not firstFrame):
                figuresFile.write("," + outStr + "\n")
            else:
                figuresFile.write(outStr + "\n")
                firstFrame = False
        
        if (not vis_figures): 
            continue

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

        vis_utils.vis_one_image(
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
            show_class=True,
            thresh=0.7,
            kp_thresh=2
        )

        #cv2.putText(im,
        #            "FPS: %f" % (1.0 / (time.time() - fps_time)),
        #            (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #            (0, 255, 0), 2)
        #    #cv2.imshow('tf-pose-estimation result', image)
        #cv2.imwrite(out_name, kp_img)

        #fps_time = time.time()
    #cv2.destroyAllWindows()

    cap.release()
    with open(figuresFilename, "a") as figuresFile:
        figuresFile.write("]")

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
