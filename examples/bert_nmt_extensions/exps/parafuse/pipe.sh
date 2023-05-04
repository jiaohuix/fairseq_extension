# 1.demose，还原文本  data_demose/
bash demose.sh
# 2.相似句子查找，train，valid，test
cp -r data_demose nmt_data_tools/LASER/

# 3.拼接src1 src2 tgt2 ->prefix.bert.de
# 已经在sim完成了


# 4.放一起，二值化
#合并laser下的data_demose和iws14
cp -r  iwslt14.tokenized.de-en  iwslt14_parafuse
cp data_demose/*.bert.* iwslt14_parafuse/
bash bin.sh
# 7m->28m

# 5. 测试训练
bash train_parafuse.sh iwslt_de_en_parafuse/ ckpt/para/

#
bash train_parafuse.sh iwslt_de_en_parafuse/ ckpt/para/
bash para/train_dbmdz2.sh iwslt_de_en_parafuse_dbmdz/ ckpt/para_dbmdz_fuse2/
bash para/train_m.sh iwslt_de_en_parafuse_mbert/ ckpt/para_mbert/