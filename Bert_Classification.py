##############################################################
#
# Bert_Classification.py
# This file contains the code for fine-tuning BERT using a
# simple classification head.
#
##############################################################

import math
import torch.nn.functional as F
import torch
import networkx as nx
import torch.nn as nn
from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx
import numpy as np
import transformers
from transformers import RobertaTokenizer, BertTokenizer, RobertaModel, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
import time
from torch.nn.utils.rnn import pad_sequence

from utils import kronecker_generator

'''
This is our own transformer layer implementation

https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial
code from :https://github.com/codertimo/BERT-pytorch/tree/d10dc4f9d5a6f2ca74380f62039526eb7277c671/bert_pytorch

'''

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super(TokenEmbedding,self).__init__(vocab_size, embed_size, padding_idx=0)

class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super(SegmentEmbedding,self).__init__(3, embed_size, padding_idx=0)

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super(PositionalEmbedding,self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]



class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super(BERTEmbedding,self).__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention,self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super(TransformerBlock,self).__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):


        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=None))


        x = self.output_sublayer(x, self.feed_forward)



        return self.dropout(x)


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super(BERT, self).__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        # self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        # x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x)



        'same shape'
        # torch.Size([2, 3, 128])

        return x


import argparse



def train():
    parser = argparse.ArgumentParser()

    # parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset for train bert")
    # parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    # parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
    # parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model")

    parser.add_argument("-hs", "--hidden", type=int, default=128, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=2, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=20, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    print("Building BERT model")
    bert = BERT(hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads).to(device)


    test = torch.rand([2, 3, 128]).to(device)

    output = bert(test)
    import pdb;
    pdb.set_trace()

    print ('BERT model...', bert)


if __name__ == "__main__":
    train()


class Bert_Classification_Model(nn.Module):
    """ A Model for bert fine tuning """

    def __init__(self):
        super(Bert_Classification_Model, self).__init__()
        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        # self.bert_drop=nn.Dropout(0.2)
        # self.fc=nn.Linear(768,256)
        # self.out=nn.Linear(256,10)
        self.out = nn.Linear(768, 10)
        # self.relu=nn.ReLU()

    def forward(self, ids, mask, token_type_ids):
        """ Define how to perfom each call

        Parameters
        __________
        ids: array
            -
        mask: array
            - 
        token_type_ids: array
            -

        Returns
        _______
            - 
        """
        import pdb;pdb.set_trace()
        'original'
        results = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)


        return self.out(results[1])


class Hi_Bert_Classification_Model(nn.Module):
    """ A Model for bert fine tuning """

    def __init__(self,num_class,device,pooling_method='mean'):
        super(Hi_Bert_Classification_Model, self).__init__()
        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.out = nn.Linear(768, num_class)
        self.device = device
        self.pooling_method=pooling_method


    def forward(self, ids, mask, token_type_ids):



        if self.pooling_method == "mean":
            emb_pool = torch.stack([torch.mean(x.float(), 0) for x in ids]).long().to(self.device)
        elif self.pooling_method == "max":
            emb_pool = torch.stack([torch.max(x.float(), 0)[0] for x in ids]).long().to(self.device)
        emb_mask = torch.stack([x[0] for x in mask]).long().to(self.device)
        emb_token_type_ids = torch.stack([x[0] for x in token_type_ids]).long().to(self.device)

        'original'
        results = self.bert(emb_pool, attention_mask=emb_mask, token_type_ids=emb_token_type_ids)


        return self.out(results[1]) # (batch_size, class_number)


class Hi_Bert_Classification_Model_LSTM(nn.Module):
    """ A Model for bert fine tuning, put an lstm on top of BERT encoding """

    def __init__(self,num_class,device,pooling_method='mean'):
        super(Hi_Bert_Classification_Model_LSTM, self).__init__()
        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)

        self.lstm_layer_number = 2
        self.lstm_hidden_size = 128

        self.bert_lstm = nn.Linear(768, self.lstm_hidden_size)
        self.device = device
        self.pooling_method=pooling_method


        self.lstm = nn.LSTM(
            input_size=self.lstm_hidden_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layer_number,
            dropout=0.2,
        )
        self.out = nn.Linear(self.lstm_hidden_size, num_class)

    def forward(self, ids, mask, token_type_ids):

        'encode bert'
        bert_ids = pad_sequence(ids).permute(1, 0, 2).long().to(self.device)
        bert_mask = pad_sequence(mask).permute(1, 0, 2).long().to(self.device)
        bert_token_type_ids = pad_sequence(token_type_ids).permute(1, 0, 2).long().to(self.device)
        batch_bert = []
        for emb_pool, emb_mask, emb_token_type_ids in zip(bert_ids, bert_mask, bert_token_type_ids):
            results = self.bert(emb_pool, attention_mask=emb_mask, token_type_ids=emb_token_type_ids)
            batch_bert.append(results[1])

        sent_bert = self.bert_lstm(torch.stack(batch_bert, 0)) # (batch, step, 128)


        'lstm starts'
        batch_size = sent_bert.shape[0]
        lstm_input = sent_bert.permute(1,0,2)

        h0 = c0 = torch.zeros(self.lstm_layer_number, batch_size, self.lstm_hidden_size).to(self.device)

        outputs, (ht, ct) = self.lstm(lstm_input, (h0, c0))

        lstm_out = self.out(outputs[-1]) # shape torch.Size([batch, 128])
        'lstm ends'


        return lstm_out # (batch_size, class_number)


class Hi_Bert_Classification_Model_BERT(nn.Module):
    """ A Model for bert fine tuning, put an lstm on top of BERT encoding """

    def __init__(self,num_class,device,pooling_method='mean'):
        super(Hi_Bert_Classification_Model_BERT, self).__init__()
        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)

        self.lstm_layer_number = 2
        self.lstm_hidden_size = 128

        # self.bert_lstm = nn.Linear(768, self.lstm_hidden_size)
        self.device = device
        self.pooling_method=pooling_method

        self.mapping = nn.Linear(768, self.lstm_hidden_size).to(device)
        self.BERTLayer = BERT(hidden=self.lstm_hidden_size, n_layers=1, attn_heads=8).to(device)
        self.out = nn.Linear(self.lstm_hidden_size, num_class).to(device)

    def forward(self, ids, mask, token_type_ids):

        'encode bert'
        bert_ids = pad_sequence(ids).permute(1, 0, 2).long().to(self.device)
        bert_mask = pad_sequence(mask).permute(1, 0, 2).long().to(self.device)
        bert_token_type_ids = pad_sequence(token_type_ids).permute(1, 0, 2).long().to(self.device)
        batch_bert = []
        for emb_pool, emb_mask, emb_token_type_ids in zip(bert_ids, bert_mask, bert_token_type_ids):
            results = self.bert(emb_pool, attention_mask=emb_mask, token_type_ids=emb_token_type_ids)
            batch_bert.append(results[1])

        sent_bert = torch.stack(batch_bert, 0)

        'BERT starts'
        lstm_input = sent_bert.permute(1,0,2)


        lstm_input = self.mapping(lstm_input)
        lstm_output = self.BERTLayer(lstm_input)
        'lstm ends'

        # import pdb;
        # pdb.set_trace()
        return self.out(lstm_output[-1]) # (batch_size, class_number)


from Graph_Models import GCN,GAT,GraphSAGE,SimpleRank,LinearFirst,DiffPool,HiPool
class Hi_Bert_Classification_Model_GCN(nn.Module):
    """ A Model for bert fine tuning, put an lstm on top of BERT encoding """

    def __init__(self,args,num_class,device,adj_method,pooling_method='mean'):
        super(Hi_Bert_Classification_Model_GCN, self).__init__()
        self.args = args
        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)

        self.lstm_layer_number = 2
        'default 128 and 32'
        self.lstm_hidden_size = args.lstm_dim
        self.hidden_dim = args.hid_dim

        # self.bert_lstm = nn.Linear(768, self.lstm_hidden_size)
        self.device = device
        self.pooling_method=pooling_method

        self.mapping = nn.Linear(768, self.lstm_hidden_size).to(device)

        'start GCN'
        if self.args.graph_type == 'gcn':
            self.gcn = GCN(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.args.graph_type == 'gat':
            self.gcn = GAT(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.args.graph_type == 'graphsage':
            self.gcn = GraphSAGE(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.args.graph_type == 'linear':
            self.gcn = LinearFirst(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.args.graph_type == 'rank':
            self.gcn = SimpleRank(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.args.graph_type == 'diffpool':
            self.gcn = DiffPool(self.device,max_nodes=10,input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.args.graph_type == 'hipool':
            self.gcn = HiPool(self.device,input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)

        self.adj_method = adj_method


    def forward(self, ids, mask, token_type_ids):

        # import pdb;pdb.set_trace()
        'encode bert'
        bert_ids = pad_sequence(ids).permute(1, 0, 2).long().to(self.device)
        bert_mask = pad_sequence(mask).permute(1, 0, 2).long().to(self.device)
        bert_token_type_ids = pad_sequence(token_type_ids).permute(1, 0, 2).long().to(self.device)
        batch_bert = []
        for emb_pool, emb_mask, emb_token_type_ids in zip(bert_ids, bert_mask, bert_token_type_ids):
            results = self.bert(emb_pool, attention_mask=emb_mask, token_type_ids=emb_token_type_ids)
            batch_bert.append(results[1])

        sent_bert = torch.stack(batch_bert, 0)



        'GCN starts'
        sent_bert = self.mapping(sent_bert)
        node_number = sent_bert.shape[1]



        'random, using networkx'

        if self.adj_method == 'random':
            generated_adj = nx.dense_gnm_random_graph(node_number, node_number)
        elif self.adj_method == 'er':
            generated_adj = nx.erdos_renyi_graph(node_number, node_number)
        elif self.adj_method == 'binom':
            generated_adj = nx.binomial_graph(node_number, p=0.5)
        elif self.adj_method == 'path':
            generated_adj = nx.path_graph(node_number)
        elif self.adj_method == 'complete':
            generated_adj = nx.complete_graph(node_number)
        elif self.adj_method == 'kk':
            generated_adj = kronecker_generator(node_number)
        elif self.adj_method == 'watts':
            if node_number-1 > 0:
                generated_adj = nx.watts_strogatz_graph(node_number, k=node_number-1, p=0.5)
            else:
                generated_adj = nx.watts_strogatz_graph(node_number, k=node_number, p=0.5)
        elif self.adj_method == 'ba':
            if node_number - 1>0:
                generated_adj = nx.barabasi_albert_graph(node_number, m=node_number-1)
            else:
                generated_adj = nx.barabasi_albert_graph(node_number, m=node_number)
        elif self.adj_method == 'bigbird':

            # following are attention edges
            attention_adj = np.zeros((node_number, node_number))
            global_attention_step = 2
            attention_adj[:, :global_attention_step] = 1
            attention_adj[:global_attention_step, :] = 1
            np.fill_diagonal(attention_adj,1) # fill diagonal with 1
            half_sliding_window_size = 1
            np.fill_diagonal(attention_adj[:,half_sliding_window_size:], 1)
            np.fill_diagonal(attention_adj[half_sliding_window_size:, :], 1)
            generated_adj = nx.from_numpy_matrix(attention_adj)

        else:
            generated_adj = nx.dense_gnm_random_graph(node_number, node_number)


        nx_adj = from_networkx(generated_adj)
        adj = nx_adj['edge_index'].to(self.device)

        'combine starts'
        # generated_adj2 = nx.dense_gnm_random_graph(node_number,node_number)
        # nx_adj = from_networkx(generated_adj)
        # adj = nx_adj['edge_index'].to(self.device)
        # nx_adj2 = from_networkx(generated_adj2)
        # adj2 = nx_adj2['edge_index'].to(self.device)
        # adj = torch.cat([adj2, adj], 1)
        'combine ends'

        if self.adj_method == 'complete':
            'complete connected'
            adj = torch.ones((node_number,node_number)).to_sparse().indices().to(self.device)

        if self.args.graph_type.endswith('pool'):
            'diffpool only accepts dense adj'
            adj_matrix = nx.adjacency_matrix(generated_adj).todense()
            adj_matrix = torch.from_numpy(np.asarray(adj_matrix)).to(self.device)
            adj = (adj,adj_matrix)
        # if self.args.graph_type == 'hipool':

        # sent_bert shape torch.Size([batch_size, 3, 768])
        gcn_output_batch = []
        for node_feature in sent_bert:


            # import pdb;pdb.set_trace()

            gcn_output=self.gcn(node_feature, adj)

            'graph-level read out, summation'
            gcn_output = torch.sum(gcn_output,0)
            gcn_output_batch.append(gcn_output)

        # import pdb;
        # pdb.set_trace()

        gcn_output_batch = torch.stack(gcn_output_batch, 0)

        'GCN ends'

        # import pdb;
        # pdb.set_trace()
        return gcn_output_batch,generated_adj # (batch_size, class_number)




class Hi_Bert_Classification_Model_GCN_tokenlevel(nn.Module):
    """ A Model for bert fine tuning, put an lstm on top of BERT encoding """

    def __init__(self,num_class,device,adj_method,pooling_method='mean'):
        super(Hi_Bert_Classification_Model_GCN_tokenlevel, self).__init__()
        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)

        self.lstm_layer_number = 2
        self.lstm_hidden_size = 128
        self.max_len = 1024

        # self.bert_lstm = nn.Linear(768, self.lstm_hidden_size)
        self.device = device
        self.pooling_method=pooling_method

        self.mapping = nn.Linear(768, self.lstm_hidden_size).to(device)

        'start GCN'
        # self.gcn = GCN(input_dim=self.lstm_hidden_size,hidden_dim=32,output_dim=num_class).to(device)
        self.gcn = GAT(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        self.adj_method = adj_method


    def forward(self, ids, mask, token_type_ids):


        batch_size = len(ids)



        reshape_ids = pad_sequence(ids).permute(1, 0, 2).long().to(self.device)
        reshape_mask = pad_sequence(mask).permute(1, 0, 2).long().to(self.device)
        reshape_token_type_ids = pad_sequence(token_type_ids).permute(1, 0, 2).long().to(self.device)

        # reshape_ids = torch.stack(ids, 0).reshape(batch_size, -1).to(self.device)
        # reshape_mask = torch.stack(mask, 0).reshape(batch_size, -1).to(self.device)
        # reshape_token_type_ids = torch.stack(token_type_ids, 0).reshape(batch_size, -1).to(self.device)



        batch_bert = []
        for emb_pool, emb_mask, emb_token_type_ids in zip(reshape_ids, reshape_mask, reshape_token_type_ids):
            results = self.bert(emb_pool, attention_mask=emb_mask, token_type_ids=emb_token_type_ids)
            batch_bert.append(results[0]) # results[0] shape: (length,chunk_len, 768)


        sent_bert = torch.stack(batch_bert, 0).reshape(batch_size,-1,768)[:,:self.max_len,:]

        # import pdb;pdb.set_trace()
        # res,not_use = self.bert(reshape_ids,attention_mask=reshape_mask, token_type_ids=reshape_token_type_ids)
        # sent_bert shape: (batch_size, seq_len, 768)



        'encode bert'
        # bert_ids = pad_sequence(ids).permute(1, 0, 2).long().to(self.device)
        # bert_mask = pad_sequence(mask).permute(1, 0, 2).long().to(self.device)
        # bert_token_type_ids = pad_sequence(token_type_ids).permute(1, 0, 2).long().to(self.device)
        # batch_bert = []
        # for emb_pool, emb_mask, emb_token_type_ids in zip(bert_ids, bert_mask, bert_token_type_ids):
        #     results = self.bert(emb_pool, attention_mask=emb_mask, token_type_ids=emb_token_type_ids)
        #     batch_bert.append(results[1])
        #
        # sent_bert = torch.stack(batch_bert, 0)



        'GCN starts'
        sent_bert = self.mapping(sent_bert)
        node_number = sent_bert.shape[1]



        'random, using networkx'

        if self.adj_method == 'random':
            generated_adj = nx.dense_gnm_random_graph(node_number, node_number)
        elif self.adj_method == 'er':
            generated_adj = nx.erdos_renyi_graph(node_number, node_number)
        elif self.adj_method == 'binom':
            generated_adj = nx.binomial_graph(node_number, p=0.5)
        elif self.adj_method == 'path':
            generated_adj = nx.path_graph(node_number)
        elif self.adj_method == 'complete':
            generated_adj = nx.complete_graph(node_number)
        elif self.adj_method == 'kk':
            generated_adj = kronecker_generator(node_number)
        elif self.adj_method == 'watts':
            if node_number-1 > 0:
                generated_adj = nx.watts_strogatz_graph(node_number, k=node_number-1, p=0.5)
            else:
                generated_adj = nx.watts_strogatz_graph(node_number, k=node_number, p=0.5)
        elif self.adj_method == 'ba':
            if node_number - 1>0:
                generated_adj = nx.barabasi_albert_graph(node_number, m=node_number-1)
            else:
                generated_adj = nx.barabasi_albert_graph(node_number, m=node_number)
        else:
            generated_adj = nx.dense_gnm_random_graph(node_number, node_number)


        nx_adj = from_networkx(generated_adj)
        adj = nx_adj['edge_index'].to(self.device)

        'combine starts'
        # generated_adj2 = nx.dense_gnm_random_graph(node_number,node_number)
        # nx_adj = from_networkx(generated_adj)
        # adj = nx_adj['edge_index'].to(self.device)
        # nx_adj2 = from_networkx(generated_adj2)
        # adj2 = nx_adj2['edge_index'].to(self.device)
        # adj = torch.cat([adj2, adj], 1)
        'combine ends'



        if self.adj_method == 'complete':
            'complete connected'
            adj = torch.ones((node_number,node_number)).to_sparse().indices().to(self.device)

        # sent_bert shape torch.Size([batch_size, 3, 768])
        gcn_output_batch = []
        for node_feature in sent_bert:
            gcn_output=self.gcn(node_feature, adj)

            'graph-level read out, summation'
            gcn_output = torch.sum(gcn_output,0)
            gcn_output_batch.append(gcn_output)



        gcn_output_batch = torch.stack(gcn_output_batch, 0)

        'GCN ends'

        # import pdb;
        # pdb.set_trace()
        return gcn_output_batch,generated_adj # (batch_size, class_number)
##############################################################
#
# RoBERT.py
# This file contains the implementation of the RoBERT model
# An LSTM is applied to a segmented document. The resulting
# embedding is used for document-level classification
#
##############################################################
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import transformers
# get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, BertTokenizer, RobertaModel, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
import time


class RoBERT_Model(nn.Module):
    """ Make an LSTM model over a fine tuned bert model.

    Parameters
    __________
    bertFineTuned: BertModel
        A bert fine tuned instance

    """

    def __init__(self, bertFineTuned):
        super(RoBERT_Model, self).__init__()
        self.bertFineTuned = bertFineTuned
        self.lstm = nn.LSTM(768, 100, num_layers=1, bidirectional=False)
        self.out = nn.Linear(100, 10)

    def forward(self, ids, mask, token_type_ids, lengt):
        """ Define how to performed each call

        Parameters
        __________
        ids: array
            -
        mask: array
            - 
        token_type_ids: array
            -
        lengt: int
            -

        Returns:
        _______
        -
        """
        _, pooled_out = self.bertFineTuned(
            ids, attention_mask=mask, token_type_ids=token_type_ids)
        chunks_emb = pooled_out.split_with_sizes(lengt)

        seq_lengths = torch.LongTensor([x for x in map(len, chunks_emb)])

        batch_emb_pad = nn.utils.rnn.pad_sequence(
            chunks_emb, padding_value=-91, batch_first=True)
        batch_emb = batch_emb_pad.transpose(0, 1)  # (B,L,D) -> (L,B,D)
        lstm_input = nn.utils.rnn.pack_padded_sequence(
            batch_emb, seq_lengths.cpu().numpy(), batch_first=False, enforce_sorted=False)

        packed_output, (h_t, h_c) = self.lstm(lstm_input, )  # (h_t, h_c))
#         output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, padding_value=-91)

        h_t = h_t.view(-1, 100)

        return self.out(h_t)


