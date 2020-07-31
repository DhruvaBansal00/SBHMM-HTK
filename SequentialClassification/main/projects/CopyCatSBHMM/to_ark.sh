#/bin/sh

mkdir -p data/ark
for f in $(find $1 -name '*.json')
do
	filename="${f##*/}"
	filenameWithoutExtension="${filename%.*}" 
	# change comma separated values based on which features you want
	# python feature_extraction_kinect.py $f "data/kinect/ark/${filenameWithoutExtension}.ark" "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
	python feature_extraction_kinect.py $f "data/ark/${filenameWithoutExtension}.ark" "6,7,8,9,10,13,14,15,16,17,27"
done