#Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

EXAMPLE=/work/04340/tushar_n/cartoon-colorization/data/lmdb
DATA=/work/04340/tushar_n/cartoon-colorization/data/raw
TOOLS=/work/04340/tushar_n/packages/caffe/build/tools

FRAME_DATA_ROOT=/work/04340/tushar_n/cartoon-colorization/data/raw/frames/
SKETCH_DATA_ROOT=/work/04340/tushar_n/cartoon-colorization/data/raw/sketch/
REF_DATA_ROOT=/work/04340/tushar_n/cartoon-colorization/data/raw/frames/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

#echo "Creating train frame lmdb..."
#GLOG_logtostderr=1 $TOOLS/convert_imageset \
#    --resize_height=$RESIZE_HEIGHT \
#    --resize_width=$RESIZE_WIDTH \
#    $FRAME_DATA_ROOT \
#    $DATA/ltrain.txt \
#    $EXAMPLE/train_frames_lmdb

#echo "Creating val frame lmdb..."
#GLOG_logtostderr=1 $TOOLS/convert_imageset \
#    --resize_height=$RESIZE_HEIGHT \
#    --resize_width=$RESIZE_WIDTH \
#    $FRAME_DATA_ROOT \
#    $DATA/lval.txt \
#    $EXAMPLE/val_frames_lmdb

#echo "Creating train sketch lmdb..."
#GLOG_logtostderr=1 $TOOLS/convert_imageset --gray\
#    --resize_height=$RESIZE_HEIGHT \
#    --resize_width=$RESIZE_WIDTH \
#    $FRAME_DATA_ROOT \
#    $DATA/ltrain.txt \
#    $EXAMPLE/train_sketch_lmdb

echo "Creating val sketch lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_imageset --gray\
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $FRAME_DATA_ROOT \
    $DATA/lval.txt \
    $EXAMPLE/val_sketch_lmdb

echo "Done."
