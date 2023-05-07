infile=$1
outfile=$2
grep "valid on" $infile | cut -f4 -d"|" | cut -f3 -d" " > $outfile.epoch
grep "valid on" $infile | cut -f8 -d"|" | cut -f3 -d" " > $outfile.ppl
paste -d"," $outfile.epoch $outfile.ppl > $outfile
rm  $outfile.epoch && rm $outfile.ppl
echo "write epoch and ppl to $outfile!"

