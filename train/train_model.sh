#train_model.sh <solver> <log>
./caffe-colorization/build/tools/caffe train -solver $1 -weights ./models/colorization_release_v2.caffemodel -gpu 0 2>&1 | tee -a $2
