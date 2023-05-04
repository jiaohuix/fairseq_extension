# mv  infolder/sub/* to  outfolder/sub/* , except checkpoint
infolder=$1
outfolder=$2

mkdir -p $outfolder
subfolders=`ls $infolder`

ckpt=checkpoint
for sub in ${subfolders[@]}
  do
      for file in `ls $infolder/$sub`
        do
            # 如果名字不含ckpt，把它挪到out/sub/file
            if [[ $file ==  *$ckpt* ]]
                then
                  echo "has ckpt: $file "
                else
#                  echo $file
                   mkdir -p $outfolder/$sub/
                   cp -r $infolder/$sub/$file   $outfolder/$sub/
            fi
        done

  done
