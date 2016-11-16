#convert all spaces to underscores
#find -name "* *" -print0 | sort -rz | \
#  while read -d $'\0' f; do mv -v "$f" "$(dirname "$f")/$(basename "${f// /_}")"; done

#split all videos to frames
#COUNT=0;
#for f in $(find cartoons/ -name '*.avi' -or -name '*.mp4'); do
#	ffmpeg -i $f -r 1/4 -vf scale=256:256 frames/"$COUNT"_%03d.png
#	let COUNT++
#done

#convert all frames to greyscale (do this in python)
#for f in $(find frames/ -name '*.png'); do
#	IMGNAME=`basename $f`
#	convert $f -colorspace Gray greyscale/$IMGNAME
#done

#generate decolorized sketches for all frames
cp frames/*.png sketch/
gimp -i -b '(batch-cart2sketch "sketch/*.png")' -b '(gimp-quit 0)'

