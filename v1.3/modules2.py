import torch 
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size : int, heads : int):
        '''
            embed_size : maintained both in encoder and decoder
        '''
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.vlinear = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim, bias = False) for i in range(heads)])
        self.klinear = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim, bias = False) for i in range(heads)])
        self.qlinear = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim, bias = False) for i in range(heads)])

        self.feedforward = nn.Linear(embed_size, embed_size, bias = False)

    def forward(self, query : torch.Tensor, key : torch.Tensor, value : torch.Tensor, mask : torch.Tensor):
        '''
            current shapes:
            query, key, value : (batch_dates, num_companies, embed_size ) || (batch_dates, num_companies, heads, head_dim)
            mask : (batch_dates, num_companies)
        '''
        ## query size = (N, length, embed_size)
        ## paralle size = N, len, heads, head_dim
        queries_parallel = query.reshape(query.shape[0], query.shape[1], self.heads, self.head_dim)
        keys_parallel = key.reshape(key.shape[0], key.shape[1], self.heads, self.head_dim)
        values_parallel = value.reshape(value.shape[0], value.shape[1], self.heads, self.head_dim )
        
        Scaled_Attention = []

        for i in range(self.heads):
            query_slice = self.qlinear[i](queries_parallel[:,:,i]) ## shape = N, qlen, head_dim
            key_slice = self.klinear[i](keys_parallel[:,:,i]) ## shape = N, klen, head_dim
            value_slice = self.vlinear[i](values_parallel[:,:,i]) ## shape = N, vlen, head_dim
            qk_product = torch.einsum("nqd,nkd->nqk",[query_slice, key_slice]) / (self.embed_size**(1/2))
            # qk_product = torch.einsum("nqd,nqk->ndk",[query_slice, key_slice]) / (self.embed_size**(1/2))
            if mask is not None:
                # exp_mask = torch.unsqueeze(mask, dim = 2) 
                qk_product =  qk_product.masked_fill(mask == torch.tensor(0), float("-1e28"))
            qk_probs = torch.softmax(qk_product, dim = 2)
            attention = torch.einsum("nqk,nkh->nqh",[qk_probs, value_slice])
            # attention = torch.einsum("ndk,nqd->nqk",[qk_probs, value_slice])
            # print(attention.shape)
            Scaled_Attention.append(attention)
        # stack along the heads
        Scaled_Attention = torch.stack(Scaled_Attention,dim = 2).reshape(query.shape[0], query.shape[1],-1)
        out = self.feedforward(Scaled_Attention)
        return out 

class BlockLayer(nn.Module):
    def __init__(self, embed_size : int, inner_dim : int, heads : int, dropout : int):
        super(BlockLayer, self).__init__()

        self.attention = MultiHeadAttention(embed_size, heads)
        self.layernorm1 = nn.LayerNorm(embed_size)
        self.layernorm2 = nn.LayerNorm(embed_size)

        self.feedforward = nn.Sequential(
                nn.Linear(embed_size, inner_dim, bias = True),
                nn.ReLU(),
                nn.Linear(inner_dim, embed_size, bias = True)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        out1 = self.dropout(self.attention(query, key, value, mask))
        out1 = self.layernorm1(out1 + query)

        output = self.dropout(self.feedforward(out1))
        output = self.layernorm2(output + out1)
        return output
    


class Encoder(nn.Module):
    def __init__(self, embed_size : int, num_returns : int, inner_dim : int,  heads : int, repeats : int, dropout : float):
        super(Encoder, self).__init__()
        self.sqlayer = nn.Linear(num_returns, 16) #### Edit this later
        self.return_embeds = nn.Linear(1, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.Transformer_Blocks = nn.ModuleList([BlockLayer(embed_size, inner_dim, heads, dropout) for i in range(repeats)])

    def forward(self, returns : torch.Tensor, return_mask : torch.BoolTensor = None):
        
        returns = self.sqlayer(returns)
        returns = returns.unsqueeze(dim = -1)

        return_embeds = self.return_embeds(returns)
        encoded = self.dropout(return_embeds)

        for transformer_block in self.Transformer_Blocks:
            encoded = transformer_block(encoded, encoded, encoded, return_mask)
            
        return encoded
    

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, inner_dim, heads, dropout):
        super(DecoderBlock, self).__init__()
        self.masked_attention = MultiHeadAttention(embed_size, heads)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.encoder_decoder_attention = BlockLayer(embed_size, inner_dim, heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_out, return_mask = None, company_mask = None):
        decoder_attention_out = self.dropout(self.masked_attention(trg, trg, trg, return_mask))
        decoder_attention_out = self.layer_norm1(decoder_attention_out + trg)
        enc_dec_attention = self.encoder_decoder_attention(decoder_attention_out, enc_out, enc_out, company_mask)
        return enc_dec_attention
    


class Decoder(nn.Module):
    def __init__(self, embed_size : int, num_returns : int, inner_dim : int,  heads : int, repeats : int, dropout : int):
        super(Decoder, self).__init__()
        self.return_encoding = nn.Linear(1, embed_size)
        self.num_returns = num_returns
        self.embed_size = embed_size

        self.Decoder_Blocks = nn.ModuleList([DecoderBlock(embed_size, inner_dim, heads, dropout) for i in range(repeats)])
        self.final_linear_layer = nn.Linear(embed_size, 1, bias = True)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, enc_out : torch.Tensor, returns : torch.Tensor, return_mask : torch.BoolTensor = None, company_mask = None):
        num_batches = returns.shape[0]
        returns = returns.unsqueeze(dim = -1)
        return_encodings = self.return_encoding(returns) #### (batch_dates, num_companies, embed_size)
        decoder_outs = self.dropout(return_encodings)

        for decoder_block in self.Decoder_Blocks:
            decoder_outs = decoder_block(decoder_outs, enc_out, return_mask, company_mask)
        
        outputs = self.final_linear_layer(decoder_outs)
        return outputs
    

class FinTransformer(nn.Module):
    def __init__(self, embed_size : int, inner_dim : int, num_returns : int, num_decoder_returns : int, heads : int, repeats : int, dropout : float):
        super(FinTransformer, self).__init__()
        self.encoder = Encoder(embed_size, num_returns, inner_dim, heads, repeats, dropout)
        self.decoder = Decoder(embed_size, num_decoder_returns, inner_dim, heads, repeats, dropout)

    def forward(self, decoder_inputs : torch.FloatTensor,
                 return_inputs : torch.FloatTensor, company_mask : torch.LongTensor = None, 
                 return_mask : torch.LongTensor = None):
        encoder_output = self.encoder(return_inputs, company_mask)
        decoder_output = self.decoder(encoder_output, decoder_inputs, return_mask, company_mask)

        return decoder_output