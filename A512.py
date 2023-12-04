import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx

import torch.nn.functional as F
import pandas as pd
import numpy as np

import re
pd.set_option('display.max_colwidth', None)

import math

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import transformers
from transformers import RobertaTokenizer, BertTokenizer, RobertaModel, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils.rnn import pad_sequence

import pprint
import time
import timeit

from graphModels import *

MAX_LEN = 1024
CHUNK_LEN = 200
OVERLAP_LEN = int(CHUNK_LEN/2)

TRAIN_BATCH_SIZE = 32
EPOCH = 10
lr=1e-5


class Preprocess:
    
    def __init__(self, trainPath, devPath, testPath):
        self.train_df = pd.read_csv(trainPath, sep = '\t', header=0)
        self.train_df['review'] = self.train_df['headline'].str.cat(self.train_df['text'], sep=' ')
        
        self.valid_df = pd.read_csv(devPath, sep = '\t', header=0)
        self.valid_df['review'] = self.valid_df['headline'].str.cat(self.valid_df['text'], sep=' ')
        
        self.test_df = pd.read_csv(testPath, sep = '\t', header=0)
        self.test_df['review'] = self.test_df['headline'].str.cat(self.test_df['text'], sep=' ')
        
    
    def clean_text(self, sentence):
        cleaned_sentence = re.sub(r'[^a-zA-Z0-9\s]', ' ', sentence)
        cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence).strip()
        return cleaned_sentence.lower()
        
    def get_clean(self):
        self.train_df['cleaned_text'] = self.train_df['review'].apply(self.clean_text)
        self.valid_df['cleaned_text'] = self.valid_df['review'].apply(self.clean_text)
        self.test_df['cleaned_text'] = self.test_df['review'].apply(self.clean_text)
        return self.train_df[['cleaned_text', 'label']], self.valid_df[['cleaned_text', 'label']], self.test_df[['cleaned_text', 'label']]


pr = Preprocess("/scratch/smanduru/NLP/project/data/amazon_512/amazon-books-512-train.tsv",
               "/scratch/smanduru/NLP/project/data/amazon_512/amazon-books-512-dev.tsv",
               "/scratch/smanduru/NLP/project/data/amazon_512/amazon-books-512-test.tsv")

train, valid, test = pr.get_clean()

print(valid.shape, test.shape, train.shape)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


class CustomDataset(Dataset):
    
    def __init__(self, tokenizer, max_len, df, chunk_len=200, overlap_len=50, approach="all", max_size_dataset=None, min_len=249):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.overlap_len = overlap_len
        self.chunk_len = chunk_len
        self.approach = approach
        self.min_len = min_len
        self.max_size_dataset = max_size_dataset
        self.data, self.label = self.process_data(df)
        
    def process_data(self, df):
        self.num_class = len(set(df['label'].values))
        return df['cleaned_text'].values, df['label'].values
    
    def long_terms_tokenizer(self, data_tokenize, targets):
        long_terms_token = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        targets_list = []

        previous_input_ids = data_tokenize["input_ids"].reshape(-1)
        previous_attention_mask = data_tokenize["attention_mask"].reshape(-1)
        previous_token_type_ids = data_tokenize["token_type_ids"].reshape(-1)
        remain = data_tokenize.get("overflowing_tokens")
        targets = torch.tensor(targets, dtype=torch.int)
        
        start_token = torch.tensor([101], dtype=torch.long)
        end_token = torch.tensor([102], dtype=torch.long)

        total_token = len(previous_input_ids) -2 # remove head 101, tail 102
        stride = self.overlap_len - 2
        number_chunks = math.floor(total_token/stride)

        mask_list = torch.ones(self.chunk_len, dtype=torch.long)
        type_list = torch.zeros(self.chunk_len, dtype=torch.long)
        
        for current in range(number_chunks-1):
            input_ids = previous_input_ids[current*stride:current*stride+self.chunk_len-2]
            input_ids = torch.cat((start_token, input_ids, end_token))
            input_ids_list.append(input_ids)

            attention_mask_list.append(mask_list)
            token_type_ids_list.append(type_list)
            targets_list.append(targets)

        if len(input_ids_list) == 0:
            input_ids = torch.ones(self.chunk_len-2, dtype=torch.long)
            input_ids = torch.cat((start_token, input_ids, end_token))
            input_ids_list.append(input_ids)

            attention_mask_list.append(mask_list)
            token_type_ids_list.append(type_list)
            targets_list.append(targets)

        return({
            'ids': input_ids_list,
            'mask': attention_mask_list,
            'token_type_ids': token_type_ids_list,
            'targets': targets_list,
            'len': [torch.tensor(len(targets_list), dtype=torch.long)]
        })
    
    def __getitem__(self, idx):
        
        review = str(self.data[idx])
        targets = int(self.label[idx])
        data = self.tokenizer.encode_plus(
            review,
            max_length=self.max_len,
            pad_to_max_length=False,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_overflowing_tokens=True,
            return_tensors='pt')
        
        long_token = self.long_terms_tokenizer(data, targets)
        return long_token
    
    def __len__(self):
        return self.label.shape[0]

train_dataset = CustomDataset(
    tokenizer = bert_tokenizer,
    max_len = MAX_LEN,
    chunk_len = CHUNK_LEN,
    overlap_len = OVERLAP_LEN,
    df = train)


valid_dataset = CustomDataset(
    tokenizer = bert_tokenizer,
    max_len = MAX_LEN,
    chunk_len = CHUNK_LEN,
    overlap_len = OVERLAP_LEN,
    df = valid)

    
test_dataset = CustomDataset(
    tokenizer = bert_tokenizer,
    max_len = MAX_LEN,
    chunk_len = CHUNK_LEN,
    overlap_len = OVERLAP_LEN,
    df = test)


def my_collate1(batches):
    return [{key: torch.stack(value) for key, value in batch.items()} for batch in batches]

# train_loader = DataLoader(train_dataset,
#                           batch_size = TRAIN_BATCH_SIZE, 
#                           shuffle = True, 
#                           collate_fn = my_collate1)

# valid_loader = DataLoader(valid_dataset,
#                           batch_size = 8, 
#                           shuffle = False, 
#                           collate_fn = my_collate1)

# test_loader = DataLoader(test_dataset,
#                           batch_size = 8, 
#                           shuffle = False, 
#                           collate_fn = my_collate1)

# Define the size of the subset you want to sample for each epoch
subset_size = 42102  # Adjust this based on your available memory and training needs

# Custom function to randomly sample a subset
def get_subset_sampler(subset_size):
    return SubsetRandomSampler(torch.randperm(subset_size))

# Creating a custom data loader with the subset sampler
def get_data_loader(dataset, subset_size, batch_size):
    subset_sampler = get_subset_sampler(subset_size)

    # Creating PT data samplers and loaders
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=subset_sampler,
        collate_fn=my_collate1
    )

    return data_loader

# Example usage

train_loader = get_data_loader(train_dataset, subset_size, TRAIN_BATCH_SIZE)
valid_loader = get_data_loader(valid_dataset, subset_size, TRAIN_BATCH_SIZE)
test_loader = get_data_loader(test_dataset, subset_size, TRAIN_BATCH_SIZE)

# Checking Data Loading

# for batch_idx, batch in enumerate(test_loader):
#     ids = batch[batch_idx]['ids']
#     mask = batch[batch_idx]['mask']
#     token_type_ids = batch[batch_idx]['token_type_ids']
#     targets = batch[batch_idx]['targets']
#     length = batch[batch_idx]['len']

#     # Now you can print or process these items as needed
#     print(f"Batch {batch_idx + 1} IDs: {ids}")
#     print(f"Batch {batch_idx + 1} Mask: {mask}")
#     print(f"Batch {batch_idx + 1} Token Type IDs: {token_type_ids}")
#     print(f"Batch {batch_idx + 1} Targets: {targets}")
#     print(f"Batch {batch_idx + 1} Length: {length}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


num_training_steps = int(len(train_dataset) / TRAIN_BATCH_SIZE * EPOCH)


class Hi_Bert_Classification_Model_GCN(nn.Module):
    
    """ A Model for bert fine tuning, put an lstm on top of BERT encoding """

    def __init__(self, graph_type, num_class, device, adj_method, pooling_method='mean'):
        super(Hi_Bert_Classification_Model_GCN, self).__init__()
        self.graph_type = graph_type
        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        
        # self.roberta = RobertaTokenizer.from_pretrained("roberta-base")


        self.lstm_layer_number = 2
        'default 128 and 32'
        self.lstm_hidden_size = 128
        self.hidden_dim = 32
        
        # self.bert_lstm = nn.Linear(768, self.lstm_hidden_size)
        self.device = device
        self.pooling_method=pooling_method

        self.mapping = nn.Linear(768, self.lstm_hidden_size).to(device)

        'start GCN'
        if self.graph_type == 'gcn':
            self.gcn = GCN(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.graph_type == 'gat':
            self.gcn = GAT(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.graph_type == 'graphsage':
            self.gcn = GraphSAGE(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.graph_type == 'linear':
            self.gcn = LinearFirst(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.graph_type == 'rank':
            self.gcn = SimpleRank(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.graph_type == 'diffpool':
            self.gcn = DiffPool(self.device,max_nodes=10,input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.graph_type == 'hipool':
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

        if self.graph_type.endswith('pool'):
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


model = Hi_Bert_Classification_Model_GCN(graph_type = 'hipool',
                                       num_class=train_dataset.num_class,
                                       device=device,
                                       adj_method='bigbird').to(device)

def loss_fun(outputs, targets):
    loss = nn.CrossEntropyLoss()
    return loss(outputs, targets)

def graph_feature_stats(graph_feature_list):
    total_number = len(graph_feature_list)
    stats = {k:[] for k in graph_feature_list[0].keys()}
    for feature_dict in graph_feature_list:
        for key in stats.keys():
            stats[key].append(feature_dict[key])
    'get mean'
    stats_mean = {k:sum(v)/len(v) for (k,v) in stats.items()}
    return stats_mean

def get_graph_features(graph):
    'more https://networkx.org/documentation/stable/reference/algorithms/approximation.html'
    try:

        # import pdb;pdb.set_trace()
        node_number = nx.number_of_nodes(graph)  # int
        centrality = nx.degree_centrality(graph) # a dictionary
        centrality = sum(centrality.values())/node_number
        edge_number = nx.number_of_edges(graph) # int
        degrees = dict(graph.degree) # a dictionary
        degrees = sum(degrees.values()) /edge_number
        density = nx.density(graph) # a float
        clustring_coef = nx.average_clustering(graph) # a float Compute the average clustering coefficient for the graph G.
        closeness_centrality = nx.closeness_centrality(graph) # dict
        closeness_centrality = sum(closeness_centrality.values())/len(closeness_centrality)
        number_triangles = nx.triangles(graph) # dict
        number_triangles = sum(number_triangles.values())/len(number_triangles)
        number_clique = nx.graph_clique_number(graph) # a float Returns the number of maximal cliques in the graph.
        number_connected_components = nx.number_connected_components(graph) # int Returns the number of connected components.
        # avg_shortest_path_len = nx.average_shortest_path_length(graph) # float Return the average shortest path length; The average shortest path length is the sum of path lengths d(u,v) between all pairs of nodes (assuming the length is zero if v is not reachable from v) normalized by n*(n-1) where n is the number of nodes in G.
        # diameter = nx.distance_measures.diameter(graph) # int The diameter is the maximum eccentricity.
        return {'node_number': node_number, 'edge_number': edge_number, 'centrality': centrality, 'degrees': degrees,
                'density': density, 'clustring_coef': clustring_coef, 'closeness_centrality': closeness_centrality,
                'number_triangles': number_triangles, 'number_clique': number_clique,
                'number_connected_components': number_connected_components,
                'avg_shortest_path_len': 0, 'diameter': 0}
    except:
        return {'node_number': 1, 'edge_number': 1, 'centrality': 0, 'degrees': 0,
                'density': 0, 'clustring_coef': 0, 'closeness_centrality': 0,
                'number_triangles': 0, 'number_clique': 0,
                'number_connected_components': 0,
                'avg_shortest_path_len': 0, 'diameter': 0}


def train_loop_fun1(data_loader, model, optimizer, device, scheduler=None):
    '''optimized function for Hi-BERT'''

    model.train()
    t0 = time.time()
    losses = []
    #import pdb;pdb.set_trace()

    graph_features = []
    for batch_idx, batch in enumerate(data_loader):

        ids = [data["ids"] for data in batch] # size of 8
        mask = [data["mask"] for data in batch]
        token_type_ids = [data["token_type_ids"] for data in batch]
        targets = [data["targets"] for data in batch] # length: 8
        length = [data['len'] for data in batch] # [tensor([3]), tensor([7]), tensor([2]), tensor([4]), tensor([2]), tensor([4]), tensor([2]), tensor([3])]


        'cat is not working for hi-bert'
        # ids = torch.cat(ids)
        # mask = torch.cat(mask)
        # token_type_ids = torch.cat(token_type_ids)
        # targets = torch.cat(targets)
        # length = torch.cat(length)


        # ids = ids.to(device, dtype=torch.long)
        # mask = mask.to(device, dtype=torch.long)
        # token_type_ids = token_type_ids.to(device, dtype=torch.long)
        # targets = targets.to(device, dtype=torch.long)

        target_labels = torch.stack([x[0] for x in targets]).long().to(device)

        optimizer.zero_grad()

        # measure time
        start = timeit.timeit()
        outputs,adj_graph = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        end = timeit.timeit()
        model_time = end - start


        loss = loss_fun(outputs, target_labels)
        loss.backward()
        model.float()
        optimizer.step()
        if scheduler:
            scheduler.step()
        losses.append(loss.item())
        if batch_idx % 50 == 0:
            print(
                f"___ batch index = {batch_idx} / {len(data_loader)} ({100*batch_idx / len(data_loader):.2f}%), loss = {np.mean(losses[-10:]):.4f}, time = {time.time()-t0:.2f} secondes ___")
            t0 = time.time()

        graph_features.append(get_graph_features(adj_graph))


    stats_mean = graph_feature_stats(graph_features)
    pprint.pprint(stats_mean)
    
    return losses

def eval_loop_fun1(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    losses = []
    for batch_idx, batch in enumerate(data_loader):
        ids = [data["ids"] for data in batch]  # size of 8
        mask = [data["mask"] for data in batch]
        token_type_ids = [data["token_type_ids"] for data in batch]
        targets = [data["targets"] for data in batch]  # length: 8

        with torch.no_grad():
            target_labels = torch.stack([x[0] for x in targets]).long().to(device)
            outputs, _ = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = loss_fun(outputs, target_labels)
            losses.append(loss.item())

        fin_targets.append(target_labels.cpu().detach().numpy())
        fin_outputs.append(torch.softmax(outputs, dim=1).cpu().detach().numpy())
    return np.concatenate(fin_outputs), np.concatenate(fin_targets), losses

def evaluate(target, predicted):
    true_label_mask = [1 if (np.argmax(x)-target[i]) ==
                       0 else 0 for i, x in enumerate(predicted)]
    nb_prediction = len(true_label_mask)
    true_prediction = sum(true_label_mask)
    false_prediction = nb_prediction-true_prediction
    accuracy = true_prediction/nb_prediction
    return{
        "accuracy": accuracy,
        "nb exemple": len(target),
        "true_prediction": true_prediction,
        "false_prediction": false_prediction,
    }


optimizer=AdamW(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps = 0,
                                        num_training_steps = num_training_steps)
val_losses=[]
batches_losses=[]
val_acc=[]
avg_running_time = []

torch.cuda.empty_cache()

for epoch in range(EPOCH):

    t0 = time.time()
    print(f"\n=============== EPOCH {epoch+1} / {EPOCH} ===============\n")
    batches_losses_tmp=train_loop_fun1(train_loader, model, optimizer, device)
    epoch_loss=np.mean(batches_losses_tmp)
    print ("\n ******** Running time this step..",time.time()-t0)
    avg_running_time.append(time.time()-t0)
    print(f"\n*** avg_loss : {epoch_loss:.2f}, time : ~{(time.time()-t0)//60} min ({time.time()-t0:.2f} sec) ***\n")
    t1=time.time()
    output, target, val_losses_tmp=eval_loop_fun1(valid_loader, model, device)
    print(f"==> evaluation : avg_loss = {np.mean(val_losses_tmp):.2f}, time : {time.time()-t1:.2f} sec\n")
    tmp_evaluate=evaluate(target.reshape(-1), output)
    print(f"=====>\t{tmp_evaluate}")
    val_acc.append(tmp_evaluate['accuracy'])
    val_losses.append(val_losses_tmp)
    batches_losses.append(batches_losses_tmp)
    print("\t§§ model has been saved §§")

print("\n\n$$$$ average running time per epoch (sec)..", sum(avg_running_time)/len(avg_running_time))

torch.save(model, '/scratch/smanduru/NLP/project/saved_models/A512' + '/hipool_10eps.pth')