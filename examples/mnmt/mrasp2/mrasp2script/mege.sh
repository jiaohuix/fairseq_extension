infolder=$1
outfolder=$2
domains=("bible" "ccmatrix" "ikcest" "qed" "ted" "tico" "opsub" "un")
for mid in  ${domains[@]}
  do
     cat $infolder/train.${mid}.zh > $outfolder/train.${mid}.zh
     cat  $infolder/train.${mid}.ar > $outfolder/train.${mid}.ar
  done
