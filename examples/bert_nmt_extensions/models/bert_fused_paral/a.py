def forward(self, x, bert_out):
    x1 = self.self_attn(x)
    if self.bert_gate:
        x2 = self.bert_attn(q=x, k=bert_out, v=bert_out)
        x2 = GradMultiply.apply(x2, self.cfg.fuse_lr_multiply)
        x = x1 + x2
    else:
        x = x1
    return x