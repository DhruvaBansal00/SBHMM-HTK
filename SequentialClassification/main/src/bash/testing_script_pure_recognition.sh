###############################################################################
#
# Argument 1: name of the feature sets
# Argument 2: feature vector dimension
# Argument 3: test sentence (can be 'tip_thumb_hand_dist_all' as well)
#
###############################################################################

# Testing Script for Verification System
# Aditya Vishwanath

FEATURESET=$1
FVECTOR=$2
SENTENCE=$3

# For each model
for((i=9; i<=18; i++))
do

  rstFile=hresults/${SENTENCE}/res_${FEATURESET}_hmm${i}.txt
  if [ -f $rstFile ];
    then rm $rstFile
  fi

  MODELFOLDER=models/
  rcgFile=results/${SENTENCE}/res_${FEATURESET}_hmm${i}.mlf
  tsFile=lists/${SENTENCE}.test

  HVite -A -H ${MODELFOLDER}/hmm${i}/newMacros -m -S $tsFile -i $rcgFile -w wordNet.txt -s 25 dict.txt wordList

  HResults -A -h -e \?\?\? sil0 -e \?\?\? sil1 -p -t -I all_labels.mlf wordList $rcgFile >> $rstFile

  echo "***************************************"
  echo "***************************************"

done

echo ""
echo ""
echo "Recognition is complete: Please see the "${SENTENCE}" directory within the hresults directory for the recognition results!"