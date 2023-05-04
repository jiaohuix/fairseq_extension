src=de
tgt=en
ARCH=${model}_iwslt_de_en
DATAPATH=data-bin/iwslt14.tokenized.$src-$tgt
mkdir -p $SAVE
python train.py $DATAPATH \
-a $ARCH --optimizer adam --lr 0.0005 -s $src -t $tgt --label-smoothing 0.1 \
--dropout 0.3 --max-tokens 4000 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --warmup-updates 4000 --warmup-init-lr '1e-07' --keep-last-epochs 10 \
--adam-betas '(0.9,0.98)' --save-dir $SAVE --share-all-embeddings   \
--encoder-bert-dropout --encoder-bert-dropout-ratio 0.5 \
--user-dir my --no-progress-bar --max-epoch 40 --fp16 \
--ddp-backend=no_c10d \
| tee -a $SAVE/training.log
#
## for model
#if cfg.share_all_embeddings:  # hare encoder, decoder and output embeddings
#    encoder_embed_tokens = cls.build_embedding(
#        cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
#    )
#    decoder_embed_tokens = encoder_embed_tokens
#    cfg.share_decoder_input_output_embed = True
#    share_input_output_embed = True
#    share_decoder_input_output_embed = True
#
## for decoder
#if self.share_input_output_embed:
#    self.output_projection = nn.Linear(
#        self.embed_tokens.weight.shape[1],
#        self.embed_tokens.weight.shape[0],
#        bias=False,)
#    self.output_projection.weight = self.embed_tokens.weight
#else:
#    self.output_projection = nn.Linear(
#        self.output_embed_dim, len(dictionary), bias=False
#    )
## ???--share-all-embeddings 指的是bert-jam是共享？？？ bert-jam三个阶段都是共享的！  bert-nmt也是共享--share-all-embeddings
#
#data=$1
#save=$2
#CUDA_VISIBLE_DEVICES=0 fairseq-train \
#    $data --save-dir $save \
#    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
#    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#    --dropout 0.3 --weight-decay 0.0001 \
#    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#    --max-tokens 4096 \
#    --eval-bleu \
#    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#    --eval-bleu-detok moses \
#    --eval-bleu-remove-bpe \
#    --eval-bleu-print-samples \
#    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric