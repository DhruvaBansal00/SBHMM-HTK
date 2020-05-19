###############################################################################
#
# Argument 1: name of the feature sets 
# Argument 2: feature vector dimension
# Argument 3: test sentence
# Argument 4: ground truth sentence  

#./testing_script Mediapipe_all 16 Mediapipe_all Prerna abc

# ./testing_script.sh tip_thumb_hand_dist_all 16 tip_thumb_hand_dist_all abc  ----> Use this only when line with all_labels.mlf is uncommented

#RECOGNITION WORKS!!!!!!!!!
# ./testing_script.sh tip_thumb_hand_dist_all 16 tip_thumb_hand_dist_all abc

##There are two options for verification available (as labled below). The command to run option 1 is above. I haven't figured out how to run 
#option 2 yet. Also how do we interpret results?
#
###############################################################################

# Testing Script for Verification System
# Aditya Vishwanath

FEATURESET=$1
FVECTOR=$2
SENTENCE=$3
USER=$4
GROUNDTRUTH=$5  #make abc for Recognition - GroundTruth not needed as input for Recognition but needed for verification (only for the second verification command. the first one does it on all_lables)

################################################################################
# Recognition & Statistics
###############################################################################
# echo " -------------- Performing Recognition for hmm[9-18] -------------- "
# echo "Corr    Sub    Del    Ins    Err    S. Err"

# For each model
for((i=35; i<=45; i++))
#for i in {35..45}
do

  rstFile=projects/test/hresults/res_${FEATURESET}_hmm${i}.txt
  if [ -f $rstFile ];
    then rm $rstFile
  fi

  # Force alignment / verification
  MODELFOLDER=projects/test/models ##The error was here. models had a / after it causing it to look for the wrong mmf (hmm created from training) files. 
  
  rcgFile=projects/test/results/res_${FEATURESET}_hmm${i}.mlf
  # rcgFile=results/${GROUNDTRUTH}/res_${FEATURESET}_hmm${i}.mlf

  # CHANGE THIS IF YOU WANT TO TEST AGAINST ONLY SOME SENTENCES!!! (Else use the default of ALL testing files.)
  #tsFile=projects/lists/${SENTENCE}.test
  tsFile=projects/test/lists/test.data
  # For recognition, uncomment the line below
  HVite -A -H ${MODELFOLDER}/hmm${i}/newMacros -m -S $tsFile -i $rcgFile -w wordNet.txt -s 25 dict_${USER}.txt wordList_${USER}

  # HVite with forced alignment
  # For verification, uncomment one of the lines below
  
  # This is the regular HVite NOTE TO ADD -b sil0 -b sil1 to this  -> added actually. ignore!!!
  #Use this command to run -> ./testing_script.sh tip_thumb_hand_dist_all 16 tip_thumb_hand_dist_all abc
  #Ground truth isn't needed here
  ####VERIFICATION OPTION 1##########
  #HVite -a -o N -T 1 -b sil0 -b sil1 -H ${MODELFOLDER}/hmm${i}/newMacros -S $tsFile -i $rcgFile -m -y lab -t 250.0 -s 1.0 -p 0.0 -I all_labels.mlf -s 25 dict.txt wordList 

  # This is the special HVite for a specific sentence verification (Here: Alligator Behind Black Wall)

  ####VERIFICATION OPTION 2##########
  # HVite -a -T 1 -o N -H ${MODELFOLDER}/hmm${i}/newMacros -S $tsFile -i $rcgFile -m -y lab -t 250.0 -s 1.0 -p 0.0 -I mlf_files/${GROUNDTRUTH}_test_labels.mlf -s 25 dict.txt wordList 

  #For recognition uncomment this line as well
  HResults -A -h -e \?\?\? sil0 -e \?\?\? sil1 -p -t -I all_labels.mlf wordList_${USER} $rcgFile >> $rstFile

# -k 'data/tip_thumb_hand_dist/htk/%%%%*'
  echo "***************************************"
  echo "***************************************"

  # Accuracy calculation
  # python python_scripts/calculate_accuracy.py ${rstFile}

done


# UNCOMMENT STUFF BELOW FOR BIPHONE AND TRIPHONE VERIFICATION/RECOGNITION

# for((i=19; i<=22; i++))
# do

#   rstFile=hresults/${SENTENCE}/res_${FEATURESET}_hmm${i}.txt
#   if [ -f $rstFile ];
#     then rm $rstFile
#   fi

#   # Force alignment / verification
#   MODELFOLDER=models/
#   rcgFile=results/${SENTENCE}/res_${FEATURESET}_hmm${i}.mlf
#   # tsFile=lists/test_file_lists/${SENTENCE}.test
#   tsFile=lists/${SENTENCE}.test
#   # tsFile=lists/tip_thumb_hand_dist.train

#   # For recognition, uncomment the line below
#   HVite -A -C configs/hvite.conf -m -s 25 -H ${MODELFOLDER}/hmm${i}/newMacros -S $tsFile -i $rcgFile -w wordNet.txt dict.txt allphones

#   # HVite with forced alignment
#   # For verification, uncomment one of the lines below
  
#   # This is the regular HVite
#   # HVite -a -o N -T 1 -H ${MODELFOLDER}/hmm${i}/newMacros -S $tsFile -i $rcgFile -m -y lab -t 250.0 -s 1.0 -p 0.0 -I all_labels.mlf -s 25 dict.txt wordList 

#   # This is the special HVite for a specific sentence verification (Here: Alligator Behind Black Wall)
#   # HVite -a -T 1 -o N -H ${MODELFOLDER}/hmm${i}/newMacros -S $tsFile -i $rcgFile -m -y lab -t 250.0 -s 1.0 -p 0.0 -I alligator_above_black_wall_test_labels.mlf -s 25 dict.txt allphones 

#   HResults -h -s -e \?\?\? sil0 -e \?\?\? sil1 -p -t -I all_labels.mlf wordList $rcgFile >> $rstFile

#   echo "***************************************"
#   echo "***************************************"

#   # Accuracy calculation
#   # python python_scripts/calculate_accuracy.py ${rstFile}

# done