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
    if [ "$videoID" != "pBuZEGYXA6E" ]; then
        continue
    fi
    videoDir=/srv/Top_N_Kpop/figuresRawOutput/$videoID
    durInSeconds=$(ffprobe -i $vfile -show_entries format=duration -v quiet -of csv="p=0")
    if [[ ! -d $videoDir ]]; then
        echo "no data files for $videoID found, skipping"
        continue
    fi
    frameFiles=$(ls $videoDir | wc -l)
    outputDir=/srv/Top_N_Kpop/inferenceVis/$videoID
    if [[ -d $outputDir ]]; then
        echo "vis files for $videoID already exist, skipping"
        continue
    fi
    echo "Processing $videoFile"
    /usr/bin/python2 tools/vis_output.py --video-data /srv/Top_N_Kpop/figuresRawOutput/ --average-frames --video $vfile > /dev/null 2>&1
    echo "Finished creating vis frames for $videoFile"
    actualFrameRate=(bc -l <<< 'scale=2; $frameFiles/$durInSeconds')

    # Create video from vis frames
    $(ffmepg -framerate $actualframeRate -pattern_type glob -i '$outputDir/*.jpg' -c:v libx264 -pix_fmt yuv420p -vf scale=960x540 $videoID.mp4)
    $(ffmpeg -nostdin -i $vfile -vn -acodec libmp3lame $videoID.mp3
    $(ffmpeg -nostdin -i $videoID.mp4 -i $videoID.mp3 -codec copy -shortest $videoID_figs_audio.mp4)
    $(rm $videoID.mp3 $videoID.mp3)

    echo "Finished processing $videoFile"
done
