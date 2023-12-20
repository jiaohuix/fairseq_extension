from fairseq.models import register_model_architecture
'''
transformer_12_12
transformer_12_12_wide
transformer_12_12_small
'''

@register_model_architecture('transformer', 'transformer_12_12')
def transformer_12_12(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.activation_dropout = getattr(args, "activation_dropout", 0.)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'transformer_12_12_wide')
def transformer_12_12(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1536)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1536)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1536 * 4)
    args.encoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1536 * 4)
    args.activation_dropout = getattr(args, "activation_dropout", 0.)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)



@register_model_architecture('transformer', 'transformer_12_12_small')
def transformer_12_12(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    # args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    # args.encoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.activation_dropout = getattr(args, "activation_dropout", 0.)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)

