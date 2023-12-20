echo "bash extract.sh <file> <lang>"
FILE=$1
LANG=$2 # ZH/AR

LANG_Code=zho_Hans
TGT_Code=

if [ "$LANG"x == "ar"x ];then
    LANG_Code=arb_Arab
fi


cat  $FILE | grep -P "^D" | sort -V | cut -f 3- > result.txt
sed -i "s/$LANG_Code //g"  result.txt

