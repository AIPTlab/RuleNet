import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import *
import os
import logging
from torch.utils.data import DataLoader
from data_for_o1 import o1Dataset, TestDataset,O1TestDataset

class SelfAttention(nn.Module):
    """
    Canonical implementation of multi-head self attention.
    """

    def __init__(self, emb, heads=8, mask=False):
        """

        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads
        self.mask = mask

        s = emb // heads
        # - We will break the embedding into `heads` chunks and feed each to a different attention head

        self.tokeys    = nn.Linear(emb, emb, bias=False)
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues  = nn.Linear(emb, emb, bias=False)

        self.unifyheads = nn.Linear(emb, emb)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        s = e // h

        keys    = self.tokeys(x)
        queries = self.toqueries(x)
        values  = self.tovalues(x)

        keys    = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values  = values.view(b, t, h, s)

        # -- We first compute the k/q/v's on the whole embedding vectors, and then split into the different heads.
        #    See the following video for an explanation: https://youtu.be/KmAISyVvE1Y

        # Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        # if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
        #     mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)

class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0, pos_embedding=None):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        x = self.do(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)
        x = self.do(x)
        return x

 
class QTransformer(nn.Module):
    """
    Transformer for generating query embeddings q1 q2 q3 ... qT.
    """

    def __init__(self, emb, heads, depth, seq_length, device, dropout=0.0 ):
        """
        :param emb: Embedding dimension
        :param heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        """
        super().__init__()
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)
        self.length  = seq_length
        self.device  = device
        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=False, dropout=dropout))
        self.tblocks = nn.Sequential(*tblocks)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: (batch_size * length * dim)
        :return: generate vector (batch_size *length * dim)
        """
    
        b, t, e = x.size()
        positions = self.pos_embedding(torch.arange(t, device= self.device))[None, :, :].expand(b, t, e)
        x = x + positions
        x = self.do(x)
        x = self.tblocks(x)
        return x

class EmbeddingLayer(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma):
        super(EmbeddingLayer, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        ) #

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )




    def forward(self, sample):
        head_part, tail_part = sample
        batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

        head = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=head_part[:, 0]
        ).unsqueeze(1) #(b, 1, 500)


        relation = torch.index_select(
            self.relation_embedding,
            dim=0,
            index=head_part[:, 1]
        ).unsqueeze(1)  #(b, 500)

        nega_tail = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=tail_part.view(-1)
        ).view(batch_size, negative_sample_size, -1) #(b, 128, 500)

        posi_tail = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=head_part[:, 2]
        ).unsqueeze(1)

        # t = torch.einsum('bad, badp-> bap', head, relation)
        nega_score = head * (relation * nega_tail)
        nega_score = nega_score.sum(dim = 2)
        posi_score =  head * (relation * posi_tail)
        posi_score =  posi_score.sum(dim = 2)
        # nega_score = torch.einsum('bad, bkd-> bak', t, nega_tail).squeeze() #shape (b, negadim)
        # posi_score = torch.einsum('bad, bkd-> bak', t, posi_tail).squeeze() #shape(b, 1)
        nega_score1 = F.logsigmoid(-nega_score).mean(dim = 1)
        posi_score1 = F.logsigmoid(posi_score).squeeze()
        return posi_score1, nega_score1, nega_score


    def get_embedding(self):
        return self.entity_embedding, self.relation_embedding


class Rule_Net(nn.Module):
    def __init__(self, steps, nentity, nrelation, hidden_dim, gamma,  args ):
        super(Rule_Net, self).__init__()
        self.steps = steps
        self.hidden_dim = hidden_dim
        self.EmbeddingLayer = EmbeddingLayer('dis', nentity, nrelation, hidden_dim, gamma)
        self.relation_dim = hidden_dim
        self.nrelation  = nrelation
        self.nentity = nentity
        self.transformer = QTransformer(emb = hidden_dim, heads = 2, depth = 1, seq_length = self.steps + 1,
         device = args.device, dropout=0.0)
        self.epsilon = 2.0
        self.neighbor_size = args.neighbor_size
        self.device = args.device

        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )
        # device = torch.device(args.device)
        # self.H1 = torch.tensor(data.H1).permute(1,0)
        # self.G1 =  torch.tensor(data.G1).permute(1,0)  #(num_entities,  num_relations) (num_entities, d)
        # # self.H1 = F.softmax(self.H1, dim = 1)
        # # self.G1 = F.softmax(self.G1, dim = 1)
        # self.s_predicate =  torch.matmul(self.H1, self.EmbeddingLayer.entity_embedding) 
        # self.o_predicate =  torch.matmul(self.G1, self.EmbeddingLayer.entity_embedding) 

        self.s_predicate = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        # nn.init.uniform_(
        #     tensor=self.s_predicate, 
        #     a=-self.embedding_range.item(), 
        #     b=self.embedding_range.item()
        # )
        
        self.o_predicate = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        # nn.init.uniform_(
        #     tensor=self.o_predicate, 
        #     a=-self.embedding_range.item(), 
        #     b=self.embedding_range.item()
        # )

        self.mlp_1 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(self.steps + 1)])
        self.mlp_2 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(self.steps + 1)])
        self.mlp_3 = nn.Linear(hidden_dim, nrelation)
        # self.w1 = nn.Parameter(torch.zeros(self.relation_dim, self.relation_dim))
        # self.w2 = nn.Parameter(torch.zeros(self.relation_dim, self.relation_dim))
        # nn.init.uniform_(
        #     tensor=self.w1,
        #     a=-self.embedding_range.item(),
        #     b=self.embedding_range.item()
        # )
        # nn.init.uniform_(
        #     tensor=self.w2,
        #     a=-self.embedding_range.item(),
        #     b=self.embedding_range.item()
        # )

     

    
    def condition_similarity(self, p1, p2, q,l):
        '''
        input: p1, p2 represent predicate embedding, shape (|r| * d)
            q: represent query embedding, shape (batch * r)
        return:  batch * |r| * |r|
        '''
        batch = q.size(0)
        p1 = p1.unsqueeze(0).repeat(batch, 1, 1)
        p2 = p2.unsqueeze(0).repeat(batch, 1, 1)
        q = q.unsqueeze(1).repeat(1, self.nrelation, 1) #batch, r, r
        # q = q.uns
        p1 = self.mlp_1[l](p1) # p(r1,r2|q) = w1(p1+q)^T * w2(p2+q)
        p2 = self.mlp_2[l](p2)
        s = torch.bmm(p1, p2.transpose(1,2))
        s = torch.mul(s, q)
        return s
        # return F.softmax(s, dim  = 2)

    def condition_similarity_head(self, p1, p2, q,l):
        '''
        input: p1: shape (batch * d)
            p2: represent predicate embedding, shape (|r| * d)
            q: represent query embedding, shape (batch * r)
        return:  batch * 1 * |r|
        '''
        batch = q.size(0)
        p1 = p1.unsqueeze(1)# b 1 d
        p2 = p2.unsqueeze(0).repeat(batch, 1, 1) # b r d
        q = q.unsqueeze(1) #batch, 1, r
        # q1 = q.repeat(1, self.nrelation, 1) #batch, r, d
        
        p1 = self.mlp_1[l](p1 ) # p(r1,r2|q) = w1(p1+q)^T * w2(p2+q) b 1 d 
        p2 = self.mlp_2[l](p2 ) # b r d 
        s = torch.bmm(p1, p2.transpose(1,2)) #b * 1 * r
        s = torch.mul(s,q)

        return s 
        # return F.softmax(s, dim  = 2)

    def forward(self, sample, ss_adj, oo_adj, os_adj):
        head_part, tail_part = sample #rewrite the dataset, add the query. mask shape: r * r
        query = head_part[:,1]
        batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
        head = torch.index_select(
            self.EmbeddingLayer.entity_embedding,
            dim=0,
            index=head_part[:, 0]
        ) #(b, 500)

        relations  = self.EmbeddingLayer.relation_embedding 
        batch_queries = torch.index_select(
            self.EmbeddingLayer.relation_embedding,
            dim=0,
            index=query
        )# (batch, d)

        posi_tail = torch.index_select(
            self.EmbeddingLayer.entity_embedding,
            dim=0,
            index=head_part[:, 2]
        ).unsqueeze(1)

        nega_tail = torch.index_select(
            self.EmbeddingLayer.entity_embedding,
            dim=0,
            index=tail_part.view(-1)
        ).view(batch_size, negative_sample_size, -1) #(b, 128, d)

        # generate query sequences:
        batch_queries = batch_queries.unsqueeze(1).repeat(1, self.steps + 1, 1)
        query_sequence = self.transformer(batch_queries) #(batch, L, d)
        query_sequence = self.mlp_3(query_sequence) # batch L nrelation

        # a sigmoid function should be added here 

        temp = head
        feature = list()
        #relations b d 
        relations = relations.unsqueeze(0).repeat(batch_size,1,1).permute(0,2,1) #b d r

        #if l == 3 : v* M * M* M * t, 3 loops + 1
        

        for l in range(self.steps):
            if l == 0: 
                # first layer, calculate the similarity between query subjective and relation subjective
                p1 = torch.index_select(self.s_predicate, dim=0, index = query) # b * d
                p2 = self.s_predicate #r * d 
                S = self.condition_similarity_head(p1, p2, query_sequence[:,l,:],l) # b 1 r
                S = F.softmax(10*S, dim  = 2)
                
                #mask step

                S1 = torch.index_select(ss_adj, dim = 0, index = query).unsqueeze(1) # b *1* r
                S = torch.mul(S1,S)
                # S = S + S1
                S = S.repeat(1, self.hidden_dim, 1) # b d r
                
                temp = temp.unsqueeze(2).repeat(1, 1, self.nrelation) # b d r
                temp = torch.mul(temp, S) # b d r  对应位置相乘
                #relations: r d
                temp = torch.mul(temp, relations) # b d r  因为relations是对角矩阵，所以直接用mul
                temp  = F.normalize(temp, p=2, dim = 1)
                ski = temp
                # feature.append(temp)
            else:
                #temp: b d r
                #calculate the similarity between objective and subjective
                p1 = self.o_predicate
                p2 = self.s_predicate
                S = self.condition_similarity(p1, p2, query_sequence[:,l,:],l) #b r1 r2
                S = F.softmax(10*S, dim  = 2)
                #mask step
                S1  = os_adj.unsqueeze(0).repeat(batch_size, 1, 1) # b r1 r2
                S = torch.mul(S,S1)
                # S = S + S1
                #relations b d r2
                select_relations = torch.einsum('brs, bds->bdr', S, relations)  # b d r double but float
                temp = torch.mul(temp, select_relations) # b d r, b d r --> b d r 
                temp = F.normalize(temp, p=2, dim = 1)
                ski = ski + temp

        # last multiply 
        p1  = self.o_predicate # r * d
        p2 = torch.index_select(self.o_predicate, dim=0, index = query) #b * d
        S = self.condition_similarity_head(p2, p1, query_sequence[:,self.steps,:], self.steps) # b *1* r 
        S = F.softmax(10*S, dim  = 2)
        #mask step
        S1 = torch.index_select(oo_adj, dim = 0, index = query).unsqueeze(1) # b 1 r
        S = torch.mul(S, S1) # b 1  r
        S = S + S1 

        temp = ski
        S = S.repeat(1,  self.hidden_dim, 1) # b d r
        temp = torch.mul(temp, S) 
        temp = torch.sum(temp, 2) 
        temp = F.normalize(temp, p=2, dim= 1)
        temp = temp.unsqueeze(1) #b 1 d

        posi_score = torch.einsum('bld,bld->bl', (temp, posi_tail))#b * 1
        nega_score = torch.einsum('bld,bpd->bpl',(temp, nega_tail)).squeeze(2)

        posi_score1 = F.logsigmoid(posi_score).squeeze()
        nega_score1 = F.logsigmoid(-nega_score).mean(dim  = 1)
        return posi_score1, nega_score1, nega_score


 

 

    



