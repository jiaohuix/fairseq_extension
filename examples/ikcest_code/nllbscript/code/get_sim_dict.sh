echo "<ptm_dict> <ft_dict> <model_path> <outfile>"
ptm_dict=$1
ft_dict=$2
model_path=$3 # fasttext
outfile=$4

# get words
python get_zh_word.py $ptm_dict  ptm.zh.txt
python get_unk_zh_word.py $ptm_dict $ft_dict unk.zh.txt
# get vector
python text2vec.py ptm.zh.txt  $model_path ptm.zh.npy
python text2vec.py unk.zh.txt $model_path unk.zh.npy
# get sim words
python get_sim_words.py   ptm.zh.txt unk.zh.txt ptm.zh.npy unk.zh.npy $outfile
echo "write to $outfile success."
