echo "bash extract.sh <file> <lang>"
FILE=$1
LANG=$2 # ZH/AR
cat  $FILE | grep -P "^D" | sort -V | cut -f 3- > result.txt
sed -i "s/LANG_TOK_${LANG} //g"  result.txt

