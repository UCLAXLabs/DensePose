#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

count=1

for vfile in `ls -dr -1 /srv/Top_N_Kpop/videos/video-*`; do
    ((count++))
    #if [[ $count -gt 100 ]]; then
    #    exit 1;
    #fi 
    videoFile=$(basename -- $vfile)
    videoName="${videoFile%.*}"
    #videoID="${videoName/video-/''}"
    prefix="video-"
    videoID=${videoName#"$prefix"}
    videoDir=/srv/Top_N_Kpop/figuresRawOutput/$videoID
    if [[ -d $videoDir ]]; then
        echo "$videoID already processed, skipping"
        continue
    fi
    echo "Processing $videoFile"
    /usr/bin/python2 tools/vis_output.py --video $vfile --video-data /srv/Top_N_Kpop/figuresRawOutput/ --average-frames > /dev/null 2>&1
    echo "Finished processing $videoFile"
done
