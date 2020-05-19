###############################################################################
#
# Argument 1: name of the feature sets
# Argument 2: feature vector dimension
#
# run like so - ./training_script.sh tip_thumb_hand_dist 16 ravi
#Mediapipe has 20 features!
###############################################################################

###
# NOTE: To run all of this:
# First run this file
# Then run the testing_script.sh file to get timstamped mlf files in results directory
# Then copy over the timestamps mlf file to top-level directory
# Modify the HRest and HERest code below: change all -I flags to use the new .mlf file that has timestamps
# Then run this same training file again and then run testing again
###

# Training Script for Verification System
# Aditya Vishwanath

FEATURESET=$1
FVECTOR=$2
USER=$3
# MIXTURE=$3

# Prepare Prototype for tip_thumb_hand_dist
python2 python_scripts/gen_prototype.py 6 ${FVECTOR} models/prototype

echo "-------------- Training HMM --------------"

echo "Training starts now!"
trFile=lists/${FEATURESET}.train
# tsFile=lists/${FEATURESET}.test

###############################################################################
# HMM training
###############################################################################

# Prepare the folder for model
MODELFOLDER=models
if [ ! -e "${MODELFOLDER}" ]; then
  mkdir $MODELFOLDER
fi

#These work the best (Feb 14)
restimation_init=25
restimation_second=20

sum=$((restimation_init+restimation_second))

for((k=0; k<=${sum}; k++))
#for k in {0..$sum}
do
  HMMFOLDER=${MODELFOLDER}/hmm${k}
  echo "Making folder ${HMMFOLDER}"
  if [ ! -e "$HMMFOLDER" ]; then
    mkdir $HMMFOLDER
  fi
done

# Initialization of the model (Flat Start)
#orig - HCompV -A -T 2 -C configs/hcompv.conf -v 2.0 -f 0.01 -m -S ${trFile} -M ${MODELFOLDER}/hmm0 models/prototype >> log/${FEATURESET}.log
HCompV -A -T 2 -C configs/hcompv.conf -v 2.0 -f 0.01 -m -S ${trFile} -M ${MODELFOLDER}/hmm0 models/prototype >> log/${FEATURESET}.log
# HCompV -A -T 2 -C configs/hp.conf -v 2.0 -f 0.01 -m -S ${trFile} -M ${MODELFOLDER}/hmm0 models/prototype >> log/${FEATURESET}.log
python2 python_scripts/gen_init_models_each_word.py ${MODELFOLDER}/hmm0 wordList_${USER}
echo "Initialized model (Flat Start)"

cat wordList_${USER} | while read n
do
  echo $n
  # HRest -i 60 -l $n -C configs/hcompv.conf -m 1 -t -v 0.2 -A -L train-labels -M ${MODELFOLDER}/hmm1 -S ${trFile} -T 1 ${MODELFOLDER}/hmm0/$n >> log/${FEATURESET}.log
  HRest -A -i 60 -C configs/hrest.conf -v 0.1 -A -I all_labels.mlf -M ${MODELFOLDER}/hmm1 -S ${trFile} ${MODELFOLDER}/hmm0/$n >> log/${FEATURESET}.log
  # HRest -A -i 60 -C configs/hp.conf -v 0.1 -A -I all_labels.mlf -M ${MODELFOLDER}/hmm1 -S ${trFile} ${MODELFOLDER}/hmm0/$n >> log/${FEATURESET}.log
done

echo "HRest Iteration done"
# HERest -d ${MODELFOLDER}/hmm1 -m 1 -v 0.001 -A -L train-labels -M ${MODELFOLDER}/hmm2 -S ${trFile} -T 1 wordList_${USER} >> log/${FEATURESET}.log
#orig - HERest -A -d ${MODELFOLDER}/hmm1 -c 500.0 -v 0.0005 -A -I all_labels.mlf -M ${MODELFOLDER}/hmm2 -S ${trFile} -T 1 wordList_${USER} >> log/${FEATURESET}.log

HERest -A -d ${MODELFOLDER}/hmm1 -c 500.0 -v 0.0005 -A -I all_labels.mlf -M ${MODELFOLDER}/hmm2 -S ${trFile} -T 1 wordList_${USER} >> log/${FEATURESET}.log
echo "Completed HERest Iteration 1"
count=2
# Re-estimate 6 more times
while [ $count -lt $restimation_init ]
#for i in {$count..$restimation_init-1}
  do
    # echo "HERest Iterating ${count}"
    # HERest -v 0.001 -m 1 -A -H ${MODELFOLDER}/hmm${count}/newMacros -L train-labels -M ${MODELFOLDER}/hmm$((count+1)) -S ${trFile} -T 1 wordList_${USER} >> log/${FEATURESET}.log
    #org -     HERest -A  -c 500.0 -v 0.0005 -A -H ${MODELFOLDER}/hmm$((count))/newMacros -I all_labels.mlf -M ${MODELFOLDER}/hmm$((count+1)) -S ${trFile} -T 1 wordList_${USER} >> log/${FEATURESET}.log
    HERest -A  -c 500.0 -v 0.0005 -A -H ${MODELFOLDER}/hmm$((count))/newMacros -I all_labels.mlf -M ${MODELFOLDER}/hmm$((count+1)) -S ${trFile} -T 1 wordList_${USER} >> log/${FEATURESET}.log
    count=$((count+1))
  done

# increase mixture to 2
HHEd -A -H ${MODELFOLDER}/hmm${count}/newMacros -M ${MODELFOLDER}/hmm$((count+1)) configs/hhed.conf wordList_${USER}


count=$((count+1))
# Re-estimate 9 more times
while [ $count -lt $sum ]
  do
    echo "HERest Iterating ${count}"
    # HERest -v 0.001 -m 1 -A -H ${MODELFOLDER}/hmm${count}/newMacros -L train-labels -M ${MODELFOLDER}/hmm$((count+1)) -S ${trFile} -T 1 wordList >> log/${FEATURESET}.log
    #org -     HERest -A  -c 500.0 -v 0.0005 -A -H ${MODELFOLDER}/hmm$((count))/newMacros -I all_labels.mlf -M ${MODELFOLDER}/hmm$((count+1)) -S ${trFile} -T 1 wordList >> log/${FEATURESET}.log
    HERest -A  -c 500.0 -v 0.0005 -A -H ${MODELFOLDER}/hmm$((count))/newMacros -I all_labels.mlf -M ${MODELFOLDER}/hmm$((count+1)) -S ${trFile} -T 1 wordList_${USER} >> log/${FEATURESET}.log
    count=$((count+1))
  done







# # Create the triphones and biphones
# HLEd -A -n triphones -i wintri1.mlf configs/mktri.led all_labels.mlf

# sort wordList triphones | uniq > allphones

# # Increase context dependency to bi/triphones
# HHEd -A -T 7 -H ${MODELFOLDER}/hmm18/newMacros -M ${MODELFOLDER}/hmm19 configs/mktri.hed wordList

# # Fix the directory address format in the new wintri1.mlf file and create a new wintri.mlf file
# python add-star.py

# echo "successful so far!"

# count=19
# # Re-estimate 2 more times
# while [[ $count -lt 21 ]]
#   do
#     # echo "HERest Iterating ${count}"
#     # HERest -v 0.001 -m 1 -A -H ${MODELFOLDER}/hmm${count}/newMacros -L train-labels -M ${MODELFOLDER}/hmm$((count+1)) -S ${trFile} -T 1 wordList >> log/${FEATURESET}.log
#     HERest -A -v 0.001 -A -H ${MODELFOLDER}/hmm${count}/newMacros -I wintri.mlf -M ${MODELFOLDER}/hmm$((count+1)) -S ${trFile} -T 1 allphones >> log/${FEATURESET}.log
#     count=$((count+1))
#   done

# # Final HERest that also outputs the stats file
# HERest -A -v 0.001 -A -H ${MODELFOLDER}/hmm21/newMacros -I wintri.mlf -M ${MODELFOLDER}/hmm22 -S ${trFile} -s stats -T 1 allphones >> log/${FEATURESET}.log

# Make new triphone dictionary -- HDMan is probably (?) acting weird so I also wrote my own Python script
# python make-triphone-dict.py
# HDMan -b sil0 -b sil1 -n fulllist -g global.ded tri.dct dict.txt
# HDMan -T 1 -g mktri.hdman tri.dct dict.txt

echo "Training complete!"

echo "Now we will create the word net"
HParse -A -T 1 grammar_${USER}.txt wordNet.txt
