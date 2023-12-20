#!/bin/bash

if [ $# -lt 5 ];then
  echo "usage: bash $0 <workers> <src> <tgt> <inprefix> <spm_model>"
  exit
fi

workers=$1
lang=$2
inprefix=$3
spm_model=$4
dict=$5

source my_tools/shard_func.sh

function paral_spm(){
    local workers=$1
    local lang=$2
    local inprefix=$3
    local spm_model=$4
    local dict=$5

    # 1.shard [inprefix.lang->inprefix.lang.worker_id]
    echo "---------------------- Sharding files.----------------------"
    func_shard $workers $inprefix.${lang}

    # 2.parallel process [inprefix.lang.worker_id -> inprefix.spm.lang.worker_id]
    for i in $(seq 0 $(($workers-1)))
    do
      (
      echo "---------------------- Apply spm to shard${i}'s.----------------------"
      # weather to use jieba for punctuation and split unk
      use_jieba="--use-jieba"
      dict_path="--dict-path $dict"
      if [ "$lang"x == "ar"x ];then
           use_jieba=""
           dict_path=""
      fi
      echo "use_jieba= ${use_jieba}, dict_path= ${dict_path}"
      # spm encode
      python spm_encode.py \
             --model $spm_model \
             --output_format=piece \
             --inputs=$inprefix.$lang.${i} \
             --outputs=$inprefix.spm.$lang.${i} $use_jieba $dict_path

      rm $inprefix.$lang.${i}

      )&
    done
    wait
    echo "-------- merge --------"
    # 3.merge files [inprefix.spm.lang.worker_id -> inprefix.spm.lang]
#    touch $inprefix.spm.${lang}
    rm $inprefix.spm.${lang}
    for i in $(seq 0 $(($workers-1)))
      do
          cat  $inprefix.spm.${lang}.${i} >> $inprefix.spm.${lang}
          rm $inprefix.spm.${lang}.${i}
      done

}

paral_spm $workers $lang $inprefix $spm_model $dict
echo "all done!"