import torch
from torch import optim
import torch.nn as nn
#import torch.nn.functional as F
#from torchvision import datasets, transforms
import numpy as np
import random
import json
#import gc
from tqdm import tqdm
import math
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.autograd.set_detect_anomaly(True)
#import sys
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8

num_data_lines = 338000
num_batches = num_data_lines // batch_size
num_epochs = 50

hrtw_embedding_sizes = [100, 100, 100, 300]  # H_embedding, R_embedding, T_embedding, W_embedding size.

dynamic_batch_sequence_length = [True,
                                 60]  # If true it evaluates max of all data in the set, else it sets to the second value in this list.
dynamic_batch_subgraph_triples_size = [True, 25]

# No dropouts detected

Graph_encoder_internal_size = 100  # num_trans_units

gru_encoder_hidden_size = 512
gru_decoder_hidden_size = 512
gru_encoder_num_layers = 2
gru_decoder_num_layers = 2

Graph_Attention_Top_V_size = 512  # units -> num_units (model) -> attnV (attention_decoder)
Encoder_Attention_V_size = 512

#base_directory = "/home/chris/Documents/columbia/fall_22/dialog_project/CCM_Torch/Data/"
base_directory = "C:/Users/manue/Downloads/ConvAI Assignment1/Projects/CCM_Torch/Data/"
save_directory = base_directory + "Model_save_dir/"

train_file = base_directory + "Post_processed_data/validset.txt"
# graph_embedding_lookup_file = "C:/Users/manue/Downloads/ConvAI Assignment1/Projects/CCM_Torch/Data/Post_processed_data/embedding_lookup.txt"
word_embedding_lookup_file = base_directory + "Post_processed_data/word_embeddings.txt"
word_embedding_symbol_dictionary_file = base_directory + "Post_processed_data/word_dictionary.txt"
word_embedding_appender_file = base_directory + "add_words_vocabulary.txt"
appender_include_length = 1

f = open(word_embedding_appender_file)
word_vocabulary_appender_length = [len(f.readlines()), appender_include_length]
f.close()

if (word_vocabulary_appender_length[1] > word_vocabulary_appender_length[0]):
    word_vocabulary_appender_length[1] = word_vocabulary_appender_length[0]

f = open(word_embedding_symbol_dictionary_file, "r")
token_to_symbol_dictionary = json.loads(f.readline())
f.close()
token_to_symbol_default = token_to_symbol_dictionary['unk']


class sparse_to_dense_tensors(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        #print("my context fwd is")
        #print(ctx.saved_tensors)
        return input.to_dense()
    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        #print("my context rev is")
        #print(input)
        #print("my context grad is")
        #print(grad_output)
        output = torch.mul(grad_output,input)
        #print("my context output is")
        #print(output)
        return output

class batch_tensor_input_gen:
    def __init__(self, batch_token_target, batch_subgraph_triples_target, hrtw_embedding_sizes_as_tuples):
        self.num_tokens_in_sentence = batch_token_target
        self.num_triples_per_token = batch_subgraph_triples_target
        self.all_subgraph_embedding_keys = ['h_emb', 'r_emb', 't_emb']
        self.word_embedding_size = hrtw_embedding_sizes_as_tuples[
            3]  # Best to have it as a list instead of a single number to have a similar processing pipeline
        self.graph_comp_embedding_sizes_as_tuples = hrtw_embedding_sizes_as_tuples[0:3]

    def generate_lookup_tensor(self, lookup_data_list, embedding_type):
        if (embedding_type == 'words'):
            length_lookup_embedding = self.word_embedding_size
            g = np.array([[float(x) for x in y.split('\t')] for y in lookup_data_list])
        elif (embedding_type == 'graph_embeddings'):
            length_lookup_embedding = sum(self.graph_comp_embedding_sizes_as_tuples)
            g = [[json.loads(x)[embed_key].split('\t')[0:embed_size] for (embed_key, embed_size) in
                  zip(self.all_subgraph_embedding_keys, self.graph_comp_embedding_sizes_as_tuples)] for x in
                 lookup_data_list]

        Lookup = torch.empty(len(lookup_data_list), length_lookup_embedding)

        if (embedding_type == 'words'):
            for i in range(0, len(lookup_data_list)):
                Lookup[i, :] = torch.from_numpy(g[i])

        elif (embedding_type == 'graph_embeddings'):
            for i in range(0, len(lookup_data_list)):
                h = []
                for j in g[i]:
                    h = h + j
                    Lookup[i, :] = torch.from_numpy(np.array([float(x) for x in h])).float()

        return Lookup

    def pre_process_subgraphs(self, all_subgraph_embeddings_as_tuples, all_embedding_sizes_as_tuples,
                              all_subgraph_matchers):
        num_subgraphs = set([len(x) for x in all_subgraph_embeddings_as_tuples])
        if (len(num_subgraphs) != 1):
            print("incompatible inputs")
        num_subgraphs = num_subgraphs.pop()
        all_subgraph_processed_matrices_as_tuples = [
            np.zeros([(num_subgraphs + 1), self.num_triples_per_token, embedding_size]) for (x, embedding_size) in
            zip(all_subgraph_embeddings_as_tuples, all_embedding_sizes_as_tuples)]

        # Initialize the first one to be the matrix for not a fact set of triple

        for subgraph_index in range(0, num_subgraphs):
            # print("subgraph_index is ")
            # print(subgraph_index)
            current_subgraph_embeddings_as_tuples = [x[subgraph_index] for x in all_subgraph_embeddings_as_tuples]
            current_subgraph_matchers = all_subgraph_matchers[subgraph_index]

            current_subgraph_with_embeddings_length = set([len(x) for x in current_subgraph_embeddings_as_tuples])
            if (len(current_subgraph_with_embeddings_length) != 1):
                print("incompatible inputs")
            current_subgraph_with_embeddings_length = current_subgraph_with_embeddings_length.pop()

            if (current_subgraph_with_embeddings_length == 1):
                embedding_as_tuples = [
                    np.array([float(x) for x in current_subgraph_embeddings[0].split('\t')[0:embedding_size]]) for
                    (current_subgraph_embeddings, embedding_size) in
                    zip(current_subgraph_embeddings_as_tuples, all_embedding_sizes_as_tuples)]
                new_graph_embeddings_as_tuples = [np.tile(x, (self.num_triples_per_token, 1)) for x in
                                                  embedding_as_tuples]

            else:
                new_graph_embeddings_as_tuples = [np.empty([0, embedding_size]) for embedding_size in
                                                  all_embedding_sizes_as_tuples]
                graph_embeddings_padding_as_tuples = [np.empty([0, embedding_size]) for embedding_size in
                                                      all_embedding_sizes_as_tuples]

                for entity_index in range(0, current_subgraph_with_embeddings_length):
                    embedding_as_tuples = [np.array([float(x) for x in y[entity_index].split('\t')[0:embedding_size]])
                                           for (y, embedding_size) in
                                           zip(current_subgraph_embeddings_as_tuples, all_embedding_sizes_as_tuples)]
                    new_graph_embeddings_as_tuples = [
                        np.append(new_graph_embeddings, np.reshape(embedding, (1, embedding_size)), axis=0) for
                        (embedding, new_graph_embeddings, embedding_size) in
                        zip(embedding_as_tuples, new_graph_embeddings_as_tuples, all_embedding_sizes_as_tuples)]
                    if (current_subgraph_matchers[entity_index] == 0):
                        graph_embeddings_padding_as_tuples = [
                            np.append(graph_embeddings_padding, np.reshape(embedding, (1, embedding_size)), axis=0) for
                            (embedding, graph_embeddings_padding, embedding_size) in
                            zip(embedding_as_tuples, graph_embeddings_padding_as_tuples, all_embedding_sizes_as_tuples)]

                padder_length = self.num_triples_per_token - current_subgraph_with_embeddings_length

                if (all(current_subgraph_matchers)):
                    graph_embeddings_padding_as_tuples = new_graph_embeddings_as_tuples

                full_padder_length = set([graph_embeddings_padding.shape[0] for graph_embeddings_padding in
                                          graph_embeddings_padding_as_tuples])
                full_padder_length = full_padder_length.pop()

                full_padder_remaining = padder_length // full_padder_length
                elementwise_pads_remaining = padder_length - (full_padder_remaining * full_padder_length)
                element_idx_picker = np.arange(
                    full_padder_length)  # This is the main reason I am doing it in this set notation together. It should replace the correct pairs.

                # print(full_padder_remaining)
                # print(elementwise_pads_remaining)

                while (full_padder_remaining > 0):
                    new_graph_embeddings_as_tuples = [np.append(new_graph_embeddings, graph_embeddings_padding, axis=0)
                                                      for (new_graph_embeddings, graph_embeddings_padding) in
                                                      zip(new_graph_embeddings_as_tuples,
                                                          graph_embeddings_padding_as_tuples)]
                    # print("full_padder_remaining" + str(full_padder_remaining))
                    # print(new_graph_embeddings_as_tuples[0].shape)
                    full_padder_remaining = full_padder_remaining - 1
                    # print(full_padder_remaining)

                while (elementwise_pads_remaining > 0):
                    # print(new_graph_embeddings_as_tuples[0].shape())
                    idx_random_pick = random.choice(element_idx_picker)
                    # print("elementwise_pads_remaining" + str(elementwise_pads_remaining))
                    # print(new_graph_embeddings_as_tuples[0].shape)
                    # print(np.reshape(graph_embeddings_padding_as_tuples[0][idx_random_pick,:],(1 size(graph_embeddings_padding_as_tuples[0][0]))).shape)
                    new_graph_embeddings_as_tuples = [np.append(new_graph_embeddings,
                                                                np.reshape(graph_embeddings_padding[idx_random_pick, :],
                                                                           (1, embedding_size)), axis=0) for
                                                      (new_graph_embeddings, graph_embeddings_padding, embedding_size)
                                                      in zip(new_graph_embeddings_as_tuples,
                                                             graph_embeddings_padding_as_tuples,
                                                             all_embedding_sizes_as_tuples)]
                    elementwise_pads_remaining = elementwise_pads_remaining - 1

            for embedding_id in range(len(all_subgraph_embeddings_as_tuples)):
                all_subgraph_processed_matrices_as_tuples[embedding_id][(subgraph_index + 1), :, :] = \
                new_graph_embeddings_as_tuples[embedding_id]

                '''
                for dim1 in range(all_subgraph_processed_matrices_as_tuples[embedding_id].shape[1]):
                    for dim2 in range(all_subgraph_processed_matrices_as_tuples[embedding_id].shape[2]):
                        all_subgraph_processed_matrices_as_tuples[embedding_id][(subgraph_index+1),dim1,dim2] = new_graph_embeddings_as_tuples[embedding_id][dim1,dim2]
                '''

        return all_subgraph_processed_matrices_as_tuples

    def sentence_embed_gen(self, batch_sentence_dicts, batch_size):
        batch_token_embeddings = torch.zeros([batch_size, self.num_tokens_in_sentence,
                                              self.word_embedding_size])  # We changed above to tuple so we need to access just one element of the tuple.
        batch_respose_q_val = torch.zeros([batch_size, self.num_tokens_in_sentence])
        batch_respose_token_ids = torch.zeros([batch_size, self.num_tokens_in_sentence])
        # batch_sentence_token_graph_masks = np.zeros([batch_size, self.num_tokens_in_sentence])
        # print("generating np tensor for batch")
        
        batch_sparse_tensor_sentence_ids = [];
        batch_sparse_tensor_token_ids = [];
        batch_sparse_tensor_embed_vals = ([] for x in self.all_subgraph_embedding_keys)
        
        
        for sentence_idx in range(batch_size):
            # print(sentence_idx,end=",")
            sentence_dict = batch_sentence_dicts[sentence_idx]
            sentence_m_embed_subgraphs = sentence_dict['emb_matcher']
            all_subgraph_embeddings_as_tuples = [sentence_dict[x] for x in self.all_subgraph_embedding_keys]
            all_subgraph_processed_matrices_as_tuples = self.pre_process_subgraphs(all_subgraph_embeddings_as_tuples,
                                                                                   self.graph_comp_embedding_sizes_as_tuples,
                                                                                   sentence_m_embed_subgraphs)

            sentence_token_graph_ids = sentence_dict['post_triples']
            sentence_token_graph_ids.insert(0, 0)
            sentence_token_graph_ids.extend([0] * (self.num_tokens_in_sentence - len(sentence_token_graph_ids)))
            
            sentence_graph_sparse_token_ids = [idx for idx, val in enumerate(sentence_token_graph_ids) if val != 0]
            sentence_graph_sparse_sentence_ids = [sentence_idx]*len(sentence_graph_sparse_token_ids)
            batch_sparse_tensor_sentence_ids = batch_sparse_tensor_sentence_ids + sentence_graph_sparse_sentence_ids
            batch_sparse_tensor_token_ids = batch_sparse_tensor_token_ids + sentence_graph_sparse_token_ids
            
            sentence_sparse_token_subgraph_ids = [sentence_token_graph_ids[x] for x in sentence_graph_sparse_token_ids]
            
            batch_sparse_tensor_embed_vals = [x + [y[z,:,:] for z in sentence_sparse_token_subgraph_ids] for (x,y) in zip(batch_sparse_tensor_embed_vals,all_subgraph_processed_matrices_as_tuples)]

            # sentence_token_graph_masks = [float('inf') if x == 0 float(0) else for x in sentence_token_graph_ids]

            sentence_post_tokens = sentence_dict['post']
            sentence_post_tokens.insert(0, 'go')
            sentence_post_tokens.extend(['eos'])
            sentence_post_tokens.extend(['pad'] * (self.num_tokens_in_sentence - len(sentence_post_tokens)))

            batch_token_embeddings[sentence_idx, :] = word_embedding_lookup[(torch.tensor(
                [token_to_symbol_dictionary.get(x, token_to_symbol_default) for x in sentence_post_tokens])), :]

            sentence_response_tokens = sentence_dict['response']
            sentence_response_tokens.insert(0, 'go')
            sentence_response_tokens.extend(['eos'])
            sentence_response_tokens.extend(['pad'] * (self.num_tokens_in_sentence - len(sentence_response_tokens)))

            batch_respose_token_ids[sentence_idx, :] = torch.tensor(
                [token_to_symbol_dictionary.get(x, token_to_symbol_default) for x in sentence_response_tokens])

            response_q_val = [0 if x == -1 else x for x in sentence_dict['response_triples_local']]
            response_q_val.extend([0] * (self.num_tokens_in_sentence - len(response_q_val)))
            batch_respose_q_val[sentence_idx, :] = torch.tensor(response_q_val)
            # There is no cls in response, so there is no insert at the beginning. Just append the later values

            # for token_index in range(self.num_tokens_in_sentence):
            #    batch_respose_q_val[sentence_idx,token_index] = response_q_val[token_index]
            # batch_sentence_token_graph_masks[sentence_idx,token_index] = sentence_token_graph_masks[token_index]
            # get the token embedding
            # for dim1 in range(batch_token_embeddings.shape[2]):
            # write to batch_token_embeddings
        
        sparse_batch_tokenwise_subgraph_embeddings_as_tuples = [torch.sparse_coo_tensor([batch_sparse_tensor_sentence_ids,batch_sparse_tensor_token_ids],[x.tolist() for x in y],(batch_size,self.num_tokens_in_sentence,self.num_triples_per_token,embedding_size)).coalesce() for (y,embedding_size) in zip(batch_sparse_tensor_embed_vals,self.graph_comp_embedding_sizes_as_tuples)]
        
        return batch_token_embeddings, batch_respose_q_val, batch_respose_token_ids, sparse_batch_tokenwise_subgraph_embeddings_as_tuples

my_preprocessor_ini = batch_tensor_input_gen(dynamic_batch_sequence_length[1], dynamic_batch_subgraph_triples_size[1],
                                             hrtw_embedding_sizes)
f = open(word_embedding_lookup_file, "r")
h = f.readlines()
word_embedding_lookup = my_preprocessor_ini.generate_lookup_tensor(h, 'words')  # Best to give it as a list to process
f.close()

word_embedding_lookup_gpu = word_embedding_lookup
word_embedding_lookup_gpu = word_embedding_lookup_gpu.to(device)

class Graph_Encoder_Attention(nn.Module):
    def __init__(self, entity_embedding_size, rel_embedding_size, Graph_encoder_internal_size):
        super().__init__()
        self.head_mlp = nn.Linear(entity_embedding_size, Graph_encoder_internal_size, bias=False)
        self.tail_mlp = nn.Linear(entity_embedding_size, Graph_encoder_internal_size, bias=False)
        self.rel_mlp = nn.Linear(rel_embedding_size, Graph_encoder_internal_size, bias=False)

    def forward(self, sparse_head_subgraph_embed, sparse_tail_subgraph_embed, sparse_rel_subgraph_embed):
        rh = self.rel_mlp(sparse_rel_subgraph_embed)
        hh = self.head_mlp(sparse_head_subgraph_embed)
        th = self.tail_mlp(sparse_tail_subgraph_embed)
        beta = torch.sum(torch.mul(rh, torch.tanh(torch.add(hh, th))), dim=2)
        alpha = torch.softmax(beta, dim=1)
        g = torch.sum(torch.mul(alpha.unsqueeze(2), torch.cat((sparse_head_subgraph_embed, sparse_tail_subgraph_embed), dim=2)),
                      dim=1)
        return g

#    def forward(self, head_subgraph_embed, tail_subgraph_embed, rel_subgraph_embed):
#        rel_subgraph_embed_to_dense = rel_subgraph_embed.to_dense()
#        rh = self.rel_mlp(rel_subgraph_embed_to_dense)
#        head_subgraph_embed_to_dense = head_subgraph_embed.to_dense()
#        hh = self.head_mlp(head_subgraph_embed_to_dense)
#        tail_subgraph_embed_to_dense = tail_subgraph_embed.to_dense()
#        th = self.tail_mlp(tail_subgraph_embed_to_dense)
#        beta = torch.sum(torch.mul(rh, torch.tanh(torch.add(hh, th))), dim=3)
#        alpha = torch.softmax(beta, dim=2)
#        g = torch.sum(torch.mul(alpha.unsqueeze(3), torch.cat((head_subgraph_embed_to_dense, tail_subgraph_embed_to_dense), dim=3)),
#                      dim=2)
#        return g

class Bahadanau_Attention_Common(nn.Module):
    def __init__(self, Decoder_Input_size, V_size):
        super().__init__()
        self.Bahadnau_attention_output_mlp = nn.Linear(Decoder_Input_size, V_size, bias=False)

    def forward(self, Batch_Decoder_Input_Attender):
        output = self.Bahadnau_attention_output_mlp(Batch_Decoder_Input_Attender)
        return output


class Bahadanau_Attention(nn.Module):
    def __init__(self, Decoder_Hidden_Size, V_size):
        super().__init__()
        self.decoder_hidden_state_mlp = nn.Linear(Decoder_Hidden_Size, V_size, bias=False)
        self.decoder_mapper_mlp = nn.Linear(V_size, 1, bias=False)

    def forward(self, Decoder_Hidden_State, Batch_decoder_input_hh, Batch_Decoder_Input_Attender, sparse_indices, sparse_val):
        decoder_output_hh = self.decoder_hidden_state_mlp(Decoder_Hidden_State)
        
        #decoder_output_hh_repeat = decoder_output_hh.unsqueeze(1).expand(Batch_decoder_input_hh.shape)
        decoder_output_hh_sparse = torch.zeros(Batch_decoder_input_hh.shape).to(device)
        decoder_output_hh_sparse[sparse_indices[0],sparse_indices[1],:] = decoder_output_hh[sparse_indices[0],:]
        
        if(sparse_val == 1):
            in_hh = Batch_decoder_input_hh
        else:
            in_hh = torch.add(Batch_decoder_input_hh,decoder_output_hh_sparse)
            
        #print(sparse_val)
        
        sum_val_hh = torch.add(decoder_output_hh_sparse,in_hh)
        alpha = torch.sparse.softmax(self.decoder_mapper_mlp(sum_val_hh).to_sparse(sparse_dim=2),dim=1)
        c_bahadanau = torch.sparse.sum(Batch_Decoder_Input_Attender.mul(alpha),dim=1)
        #if(sparse_val == 1):
        #    c_bahadanau = torch.sparse.sum(Batch_Decoder_Input_Attender.mul(alpha),dim=1)
        #else:
        #    c_bahadanau = torch.sum(Batch_Decoder_Input_Attender.mul(alpha),dim=1)
        
        #
        #print(decoder_output_hh_sparse.shape)
        #  
        
        #print(sparse_val)
        #sum_val_hh = torch.add(decoder_output_hh_sparse,Batch_decoder_input_hh)
        #beta = self.decoder_mapper_mlp(sum_val_hh)
        
        #if(sparse_val == 1):
        #    alpha = torch.sparse.softmax(beta.to_sparse(sparse_dim=2),dim=1)
        #    g1 = Batch_Decoder_Input_Attender.mul(alpha)
        
        #else:
        #    alpha = torch.softmax(beta,dim=2)
        #    g1 = Batch_Decoder_Input_Attender.mul(alpha).to_sparse(sparse_dim=1)

        #print(g1)
        
        #c_bahadanau = torch.sparse.sum(g1, dim=1)
        #print(c_bahadanau)
        
        return c_bahadanau, alpha

class Graph_Attention_Hierarchy_Triples(nn.Module):
    def __init__(self, decoder_hidden_size, all_knowledge_triple_embedding_size):
        super().__init__()
        self.decoder_mapper_mlp = nn.Linear(decoder_hidden_size, all_knowledge_triple_embedding_size, bias=False)

#    def forward(self, decoder_hidden_state, alpha_graph_attention_top, all_embeddings):
#        intermediate = self.decoder_mapper_mlp(decoder_hidden_state)
#        beta = torch.sum(torch.mul(all_embeddings, intermediate.unsqueeze(1).unsqueeze(2)),
#                         dim=3)  # Should it be mul or add?. It should be Mul, we are multiplying with kj and transpose which we model as sum over mul.
#        alpha = torch.mul(alpha_graph_attention_top, torch.softmax(beta, dim=2))
#        c_hierarchical = torch.sum(torch.sum(torch.mul(alpha.unsqueeze(3), all_embeddings), dim=2), dim=1)
#        return c_hierarchical, alpha

    def forward(self, decoder_hidden_state, alpha_graph_attention_top, all_embeddings):
        intermediate = self.decoder_mapper_mlp(decoder_hidden_state)
        #print("Are all embeddings input sparse?")
        #print(all_embeddings.is_sparse)
        #print("intermediate shape is")
        #print(intermediate.shape)
        #Below is multiplication of sparse input (all_embeddings) with potentially non sparse input (intermediate) and result is still sparse. So we use sparse sum below
        int1 = torch.mul(all_embeddings, intermediate.unsqueeze(1).unsqueeze(2))
        beta = torch.sparse.sum(int1,[3])  # Should it be mul or add?. It should be add, we are multiplying with kj and transpose which we model as sum over mul.
        alpha = torch.mul(alpha_graph_attention_top, torch.sparse.softmax(beta, dim=2))
        int2 = torch.mul(alpha.unsqueeze(3), all_embeddings)
        #print("int2 sparse dim is")
        #print(int2.sparse_dim())
        #print("int2 dense dim is")
        c_hierarchical = torch.sparse.sum(int2,[1,2])
        #print("c_hierarchical sparse dim is")
        #print(c_hierarchical.sparse_dim())
        return c_hierarchical, alpha

class MultiLayer_GRU_Cell(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru_cell_array = []
        self.num_layers = num_layers
        layer_input_size = input_size
        for layer in range(num_layers):
            self.gru_cell_array.append(nn.GRUCell(layer_input_size, hidden_size).to(device))
            layer_input_size = hidden_size

    def forward(self, gru_input, gru_hidden_state_input):
        gru_hidden_state = torch.zeros(gru_hidden_state_input.size()).to(device)
        layer_input = gru_input
        for layer in range(self.num_layers):
            gru_hidden_state[layer] = self.gru_cell_array[layer](layer_input, gru_hidden_state_input[layer])
            layer_input = gru_hidden_state_input[layer]
        return gru_hidden_state[self.num_layers - 1], gru_hidden_state


"""
List of sizes for all models
1. inputs : word_embedding_size, entity_embedding_size, rel_embedding_size, sequence_length, largest subgraph length.
2. model_params : gru_encoder_hidden_size, gru_encoder_num_layers, gru_decoder_hidden_size, gru_decoder_num_layers
                  graph_attention_top_vb_size
                  Graph_encoder_internal_size
3. Graph Encoder Attention:
        1. W_r -> multiplies with rel_embedding. Can either be outputting to
                    a.dimension of entity_embedding (W_r = entity_embedding x rel_embedding)
                    b.dimension of rel_embedding (W_r = rel_embedding x rel_embedding)
                    # or they can both map to another variable size, that we can specify as model param "Graph_encoder_internal_size"
        2. W_h, W_t -> multiplies with ent_embed_post and ent_embed_response respectively. Can either be outputting to
                    a.dimension of entity_embedding (W_h = entity_embedding x entity_embedding)
                    b.dimension of r_embedding (W_h = rel_embedding x entity_embedding)
                    # We can also use "Graph_encoder_internal_size"
Internal3. B_g_enc -> result of multiplying (transpose(W_r*rel_embedding) with (W_h*ent_embed_post + W_t*ent_embed_response) for each subgraph triple.
                a. Should be of dimension (batch_size x (sequence_length x largest_subgraph_length) x 1) : Last dimension is 1 as it is a single number
Internal4. A_g_enc -> Scaling all values of B_g_enc in a given sequence ie (along largest_subgraph_length x 1 space) by using exponential function.
                ie (for batch in batch size for sequence in sequence length)
                    A = exponential function (batch,sequence,:,:)
                a. Should be of dimension (batch_size x (sequence_length x largest_subgraph_length) x 1) : Last dimension is 1 as it is a single number
Retain  5. g -> Multiplying values of A_g_enc with concatenation of ent_embed_post;ent_embed_response in a given sequence.
                ie (for batch in batch size for sequence in sequence length)
                is of dimension (batch,sequence,2*entity_embedding_size)
        Overall: takes in subgraph of one token and outputs a single vector gi per token.
                 We create a tensor that collects all gi of all sentence and token for a given batch. This tensor will be of size (batch_size x sequence_length x entity_embedding_size)
4. GRU encoder :
        1. hidden state and num_layers : gru_encoder_hidden_size, gru_encoder_num_layers
        2. input : concatenation of token_embedding and gi outputted by Graph Encoder Attention
                   input_size is (batch_size x sequence_length x (word_embedding_size+2*entity_embedding_size))
        3. Output : h_t output_size is (batch_size x sequence_length x gru_encoder_hidden_size)
5. Graph_attention_top :
        1. V_b -> A vector of size "graph_attention_top_vb_size"
        2. W_b -> Multiplies with GRU_decoder_hidden_state (s_t) for that token
                 must be of size (graph_attention_top_vb_size x gru_decoder_hidden_size)
        3. U_b -> Multiplies with Graph Encoder Output (gi) for that token.
                 must be of size (graph_attention_top_vb_size x 2*entity_embedding_size)
Internal4. B_g_top -> result of multiplying (transpose(V_b) with (W_b*s_t + U_b*gi) for each token.
                      s_t varies for each token and can be evaluated only when the GRU cell updates
                      gi is constant for each token and V_b, W_b, U_b are also constant for the iteration.
                      So we can compute U_b*gi (result is a vector) as a tensor for entire batch and keep first
                      UBGI_epoch is of size (batch_size x sequence_length x graph_attention_top_vb_size)
                      and then evaluate W_b*s_t for each token as we compute the value for GRU output
                      WBST_token is of size (batch_size x 1 x graph_attention_top_vb_size)
                      for each token, we compute WBST_token (which is a vector)
                      Add WBST_token for each sequence entry in UBGI_epoch

                      Final b_g_top is of size (batch_size x sequence_length x 1) for each token instant.

Retain  5. A_g_top -> Scaling all values of B_g_top in a given sequence by exponential function
                      Final A_g_top is of size (batch_size x sequence_length x 1) for each token instant.
Retain  6. C_g_top -> Scaled sum of entries in gi with A_g_top
                      Final C_g_top is of size (batch_size x 2*entity_embedding_size) for each token instant.
6. Graph_attention_bottom :
        1. W_c -> pre multiplies with concatenated (ent_embed_post; rel_embedding; ent_embed_response) and post multiplies with
                  GRU_decoder_hidden_state (s_t) for that token.
                  of size ((2*entity_embedding_size+rel_embedding_size) x gru_decoder_hidden_size)
                  here s_t is the variable.
Input   2. k_j -> set of knowledge graph embeddings is fixed for both within a set of subgraphs per token and also for all tokens in sentence.
Internal3. B_g_bottom -> result of multiplying transpose(kj)*W_c*s_t
                         s_t proceeds token by token but kj*W_c can be computed altogether
                         but it needs to store it as transpose of a vector. We can store as it's transpose and then compute later.
                         KJWC_epoch is of size (batch_size x sequence_length x largest_subgraph_length x gru_decoder_hidden_size)

                         from here, for given token
                         B_g_bottom is computed as taking each subgraph in KJWC_epoch
                            ie KJWC_epoch(batch,sequence,:,:) .* s_t ie taking a dot product.
                         But actually we will need to compute B_g for all triples in entire sentence as it attends over all graphs hierarchically.
                         B_g_bottom is of size (batch_size x sequence_length x largest_subgraph_length) for each token instant.
                            we just taken a dot product with KJWC_epoch(batch,:,:,:) with st
                        
??      4. A_g_bottom -> Scaling all values of B_g_top in a given sequence by exponential function per each token.
                         So we need to compute exponential scaling token by token, but we finally need it of the same size 
                         But we also need to multiply this with A_g_top for that sequence.

Retain  5. C_g_bottom -> 

7. Attentive Read:
        1. input : all encoder outputs h_t for entire sequence
Internal2. A_k   : for all inputs, and decoder state s_t for each token.
Retain  3. output : Scaling all values of h_t by A_k and summing them.
        Done token by token.

8. GRU_decoder :
        1. hidden state and num_layers : gru_decoder_hidden_size, gru_decoder_num_layers
        2. input : concatenation of [attentive read of all encoder_hidden_states h_t]
                                    [C_g_top]
                                    [C_g_bottom]
                                    [concatenation of [word_embedding_y_(t-1) ; knowledge_triple_kj used for word_embedding]]
                   input_data_size: (gru_encoder_hidden_size + 2*entity_embedding_size + (2*entity_embedding_size+rel_embedding_size) + word_embedding_size + (2*entity_embedding_size+rel_embedding_size))
                                  => (word_embedding_size + gru_encoder_hidden_size + 6*entity_embedding_size + 2*rel_embedding_size)
                   input_size = (batch_size x sequence_length x word_embedding_size + gru_encoder_hidden_size + 6*entity_embedding_size + 2*rel_embedding_size)
        3. Output : s_t output_size is (batch_size x sequence_length x gru_decoder_hidden_size)
8. Generator :
        1. A_t = concatenation of GRU_decoder_output[s_t]
                                  [attentive read of all encoder_hidden_states h_t]
                                  [C_g_top]
                                  [C_g_bottom]
                data_size of A_t (gru_decoder_hidden_size + gru_encoder_hidden_size + 2*entity_embedding_size + (2*entity_embedding_size+rel_embedding_size))
                              ie (gru_decoder_hidden_size + gru_encoder_hidden_size + 4*entity_embedding_size + rel_embedding_size)
        2. V_0 -> of same dimension as A_t tranposed. Multiplies with A_t to give a single number
        3. gamma_t = sigmoid(V_0 .* A_t)
        4. Pc = softmax(W_0 * A_t)
"""


class CCM_Model(nn.Module):
    def __init__(self, Graph_encoder_internal_size, gru_encoder_hidden_size, gru_encoder_num_layers,
                 Encoder_Attention_V_size, Graph_Attention_Top_V_size, gru_decoder_hidden_size, gru_decoder_num_layers,
                 all_embedding_sizes_as_tuples, word_vocabulary_appender_length):
        super(CCM_Model, self).__init__()

        entity_embedding_size = all_embedding_sizes_as_tuples[0]
        relation_embedding_size = all_embedding_sizes_as_tuples[2]
        word_embedding_size = all_embedding_sizes_as_tuples[3]

        encoder_input_size = word_embedding_size + 2 * entity_embedding_size
        #encoder_input_size = word_embedding_size
        self.gru_encoder_num_layers = gru_encoder_num_layers
        self.encoder = nn.GRU(encoder_input_size, gru_encoder_hidden_size, self.gru_encoder_num_layers,
                              batch_first=True)

        self.Graph_Encoder_Attention_output_size = 2 * entity_embedding_size
        self.my_graph_attention = Graph_Encoder_Attention(entity_embedding_size, relation_embedding_size,
                                                          Graph_encoder_internal_size)
        
        self.Graph_Attention_Top_V_size = Graph_Attention_Top_V_size

        self.my_graph_attention_top_common = Bahadanau_Attention_Common(self.Graph_Encoder_Attention_output_size,
                                                                        Graph_Attention_Top_V_size)
        self.my_graph_attention_top = Bahadanau_Attention(gru_decoder_hidden_size, Graph_Attention_Top_V_size)

        Encoder_attention_output_size = gru_encoder_hidden_size
        self.my_encoder_attention_common = Bahadanau_Attention_Common(gru_encoder_hidden_size, Encoder_Attention_V_size)
        self.my_encoder_attention = Bahadanau_Attention(gru_decoder_hidden_size, Encoder_Attention_V_size)

        self.all_knowledge_triple_embedding_size = relation_embedding_size + 2 * entity_embedding_size  # called in a few places so best to use self.values
        self.my_graph_attention_triples = Graph_Attention_Hierarchy_Triples(gru_decoder_hidden_size,
                                                                            self.all_knowledge_triple_embedding_size)

        decoder_input_size = self.Graph_Encoder_Attention_output_size + (
                    2 * self.all_knowledge_triple_embedding_size) + Encoder_attention_output_size + word_embedding_size
        #decoder_input_size = word_embedding_size + self.all_knowledge_triple_embedding_size
        self.decoder_multilayer_cell = MultiLayer_GRU_Cell(decoder_input_size, gru_decoder_hidden_size,
                                                           gru_decoder_num_layers)

        self.word_vocabulary_size = word_embedding_lookup_gpu.shape[0]
        word_generator_input_size = gru_decoder_hidden_size
        self.word_symbol_mlp = nn.Linear(word_generator_input_size, self.word_vocabulary_size, bias=False)
        self.probability_word_knowledge_mlp = nn.Linear(word_generator_input_size, 1, bias=False)

        self.word_vocab_appender_all_idx = self.word_vocabulary_size - word_vocabulary_appender_length[0] - 1
        self.word_vocab_appender_min_idx = self.word_vocabulary_size + word_vocabulary_appender_length[1] - \
                                           word_vocabulary_appender_length[0] - 1

        self.vocab_Softmax = nn.Softmax(dim=1)

    def gen_output_word_knowledge_probabilites(self, input, word_lookup):
        predicted_word_symbol_probabilities = self.vocab_Softmax(self.word_symbol_mlp(input))
        predicted_word_symbol_index = torch.argmax(predicted_word_symbol_probabilities,
                                                   dim=1)  # Should already be present as int
        # predicted_word_symbol_index_no_appender = torch.argmax(
        #     self.word_symbol_mlp(input[:, 0:self.word_vocab_appender_all_idx]), dim=1)
        predicted_word_embedding = word_lookup[predicted_word_symbol_index, :]
        probability_word_knowledge = self.probability_word_knowledge_mlp(input)

        # return predicted_word_symbol_probabilities, predicted_word_symbol_index, predicted_word_embedding, probability_word_knowledge, predicted_word_symbol_index_no_appender, predicted_word_symbol_index_min_appender
        return predicted_word_symbol_probabilities, predicted_word_embedding, probability_word_knowledge

    def forward(self, Batch_All_embeddings, word_lookup, Batch_All_embeddings_cpu):
        sentence_ids = torch.arange(0, Batch_All_embeddings[0].shape[0]).to('cpu')
        #Batch_Graph_Embeddings = Batch_All_embeddings[0:3]  # h,r,t
        #Batch_Graph_Embeddings = [current_embedding.to_dense() for current_embedding in Batch_All_embeddings[0:3]]
        Batch_Graph_Embeddings_sparse = [current_embedding.values() for current_embedding in Batch_All_embeddings[0:3]]
        Batch_Graph_Embeddings_sparse_indices = Batch_All_embeddings[0].indices()
        Batch_Graph_Embeddings_sparse_cpu = torch.cat((Batch_All_embeddings_cpu[0],Batch_All_embeddings_cpu[2],Batch_All_embeddings_cpu[1]), dim=3)
        #Batch_Graph_Embeddings_dense = [current_embedding.to_dense() for current_embedding in Batch_Graph_Embeddings]
        Batch_Word_Embeddings = Batch_All_embeddings[3]
        
        g_sparse_vals = self.my_graph_attention(Batch_Graph_Embeddings_sparse[0], Batch_Graph_Embeddings_sparse[2],
                                     Batch_Graph_Embeddings_sparse[1])

        g_sparse = torch.sparse_coo_tensor(Batch_Graph_Embeddings_sparse_indices,self.my_graph_attention(Batch_Graph_Embeddings_sparse[0], Batch_Graph_Embeddings_sparse[2],
                                     Batch_Graph_Embeddings_sparse[1]),[Batch_All_embeddings[0].shape[0],Batch_All_embeddings[0].shape[1],self.Graph_Encoder_Attention_output_size])  # h,t,r in the input
        
        encoder_inp = torch.cat((Batch_Word_Embeddings, g_sparse.to_dense()), dim=2)
        #encoder_inp = Batch_Word_Embeddings
        encoded_all, encoder_fin = self.encoder(encoder_inp)
        
        spdense = sparse_to_dense_tensors.apply

        g_top_attention = torch.sparse_coo_tensor(Batch_Graph_Embeddings_sparse_indices,self.my_graph_attention_top_common(g_sparse_vals),[Batch_All_embeddings[0].shape[0],Batch_All_embeddings[0].shape[1],self.Graph_Attention_Top_V_size])
        encoder_top_attention = self.my_encoder_attention_common(encoded_all)

        # decoder_hidden_state = torch.zeros(batch_size,gru_decoder_hidden_size) # initializing to zeros for entire batch initially
        decoder_layer_states = encoder_fin
        decoder_hidden_state = encoder_fin[self.gru_encoder_num_layers - 1]
        word_symbol_probabilties_tensor = torch.zeros(batch_size, 0, self.word_vocabulary_size).to(device)
        probability_word_knowledge_tensor = torch.zeros(batch_size, 0).to(device)

        predicted_word_symbol_probabilities, predicted_word_embedding, probability_word_knowledge = self.gen_output_word_knowledge_probabilites(
            decoder_hidden_state,word_lookup)
        predicted_knowledge_embedding_gpu = torch.zeros(batch_size, self.all_knowledge_triple_embedding_size).to(device)

        for seq_pos in range(0, Batch_Word_Embeddings.shape[1]):
            #print("current sequence position is" + str(seq_pos))
            
            c_g, alpha_graph_attention_top = self.my_graph_attention_top(decoder_hidden_state, g_top_attention, g_sparse, Batch_Graph_Embeddings_sparse_indices, True)
            c_hierarchical, alpha_hierarchical_attention = self.my_graph_attention_triples(decoder_hidden_state,alpha_graph_attention_top,torch.cat((Batch_All_embeddings[0],Batch_All_embeddings[2],Batch_All_embeddings[1]), dim=3))

            #alpha_hierarchical_attention = alpha_hierarchical_attention.to_dense()

            c_e, alpha_encoder_attention = self.my_encoder_attention(decoder_hidden_state, encoder_top_attention, encoded_all, Batch_Graph_Embeddings_sparse_indices, False)
            
            #c_hierarchical.sparse_resize_(c_hierarchical.shape,2,0)
            
            #c_hierarchical = c_g.to_sparse().sparse_resize_([Batch_Word_Embeddings.shape[0],self.Graph_Encoder_Attention_output_size],1,1)
            #print("c_hierarchical sparse_dim is")
            #print(c_hierarchical.sparse_dim())
            
            #c_hierarchical.sparse_resize_([Batch_Word_Embeddings.shape[0],self.all_knowledge_triple_embedding_size],,0)
            
            #h1 = torch.cat((c_g.to_sparse(sparse_dim = 1), c_hierarchical, c_e.to_sparse(sparse_dim = 1), predicted_word_embedding.to_sparse(sparse_dim = 1), predicted_knowledge_embedding.to_sparse(sparse_dim = 1)),dim=1)
            h1 = torch.cat((c_g, c_hierarchical, c_e, predicted_word_embedding.to_sparse(sparse_dim=1), predicted_knowledge_embedding_gpu.to_sparse(sparse_dim=1)),dim=1).coalesce()
            print(h1)
            #print(h1)
            #print(h1)
            h2 = spdense(h1)
            #print(h2)
            
            #h1 = torch.cat((predicted_word_embedding, predicted_knowledge_embedding_gpu),dim=1)
            
            #define a new function that maps the h1 and alpha_hierarchical_attention to dense without losing gradients. They are actually dense but just annotated as sparse.
            #just force some gradients to 0 when then they are sparse and make it propagate instead of going through the ususal neglection in the existing backprop.

            decoder_hidden_state, decoder_layer_states = self.decoder_multilayer_cell(h2,decoder_layer_states)

            # predicted_word_symbol_probabilities, predicted_word_symbol_index, predicted_word_embedding, probability_word_knowledge, predicted_word_symbol_index_no_appender, predicted_word_symbol_index_min_appender = self.gen_output_word_knowledge_probabilites(
            #     decoder_hidden_state)

            predicted_word_symbol_probabilities, predicted_word_embedding, probability_word_knowledge = self.gen_output_word_knowledge_probabilites(
                decoder_hidden_state,word_lookup)
            
            alpha_hierarchical_attention_cpu = alpha_hierarchical_attention.to('cpu')
            
            alpha_hierarchical_attention_dense = alpha_hierarchical_attention_cpu.to_dense()
            
            #maybe rewrite the module to be more efficient. We are using two instances of the dense array which is memory consuming.
            #We ideally just need to get argmax over only one dimension. And this is never going to be used in the gradients (no need to worry about to_dense propagation). Since it is only used as a lookup.
            predicted_subgraph_token_index = torch.argmax(torch.max(alpha_hierarchical_attention_dense, dim=2).values, dim=1)
            predicted_triple_index_in_subgraph = torch.argmax(alpha_hierarchical_attention_dense[sentence_ids, predicted_subgraph_token_index, :], dim=1)

            Batch_Graph_Embeddings_dense_cpu = Batch_Graph_Embeddings_sparse_cpu.to_dense()
            predicted_knowledge_embedding_cpu = Batch_Graph_Embeddings_dense_cpu[sentence_ids, predicted_subgraph_token_index.to('cpu'), predicted_triple_index_in_subgraph.to('cpu')]

            predicted_knowledge_embedding_gpu = predicted_knowledge_embedding_cpu.to(device)

            word_symbol_probabilties_tensor = torch.cat(
                (word_symbol_probabilties_tensor, predicted_word_symbol_probabilities.unsqueeze(1)), dim=1)
            probability_word_knowledge_tensor = torch.cat(
                (probability_word_knowledge_tensor, probability_word_knowledge),
                dim=1)  # No need for unsqueeze as MLP automatically adds last dimension as 1
            # print("Predicted predicted_word_symbol_index tensor dimension, predicted_word_embedding dimension, probability_word_knowledge tensor dimension")
            # print(word_symbol_index_tensor.shape)
            # print(predicted_word_embedding.shape)
            # print(probability_word_knowledge_tensor.shape)
            # print("predicted probability word knowledge shape is")
            # print(probability_word_knowledge.shape)

        #return decoder_hidden_tensor, word_symbol_probabilties_tensor, word_symbol_index_tensor, probability_word_knowledge_tensor, alpha_hierarchical_attention, alpha_encoder_attention, word_symbol_index_no_appender_tensor, word_symbol_index_min_appender_tensor
        return word_symbol_probabilties_tensor, probability_word_knowledge_tensor, alpha_hierarchical_attention


def generate_batches(train_file, batch_size=512):
    curr = []

    with open(train_file, 'r') as fin:
        for line in fin.readlines():
            curr.append(json.loads(line))

            if len(curr) == batch_size:
                yield curr
                curr = []

    return None

word_vocabulary_size = word_embedding_lookup.shape[0]

mymodel = CCM_Model(Graph_encoder_internal_size, gru_encoder_hidden_size, gru_encoder_num_layers,
                    Encoder_Attention_V_size, Graph_Attention_Top_V_size, gru_decoder_hidden_size,
                    gru_decoder_num_layers, hrtw_embedding_sizes,
                    word_vocabulary_appender_length).to(device)
loss = torch.nn.CrossEntropyLoss()
learning_rate = 4 * 1e-3
optimizer = optim.SGD(mymodel.parameters(), lr=learning_rate)

for epoch in range(0, 100):
    f = open(train_file, "r")
    # print("epoch is" + str(epoch))
    for batch in tqdm(range(0, 1)):
        h = []
        for sentence_id in range(batch_size):
            g = f.readline()
            h.append(json.loads(g))

        max_post_length = max([len(x['post']) for x in h])
        max_response_length = max([len(x['response']) for x in h])
        current_batch_length = max([max_post_length, max_response_length]) + 4  # POS

        my_preprocessor = batch_tensor_input_gen(current_batch_length,
                                                 dynamic_batch_subgraph_triples_size[1], hrtw_embedding_sizes)
        
        batchwise_all_token_embeddings, q_vals, batch_all_responses, sparse_sentence_processed_subgraphs  = my_preprocessor.sentence_embed_gen(h, batch_size)

        sparse_sentence_processed_subgraphs_gpu = [x.to(device) for x in sparse_sentence_processed_subgraphs]
        sparse_sentence_processed_subgraphs_gpu.append(batchwise_all_token_embeddings.to(device))

        gg = torch.zeros([batch_size, current_batch_length, word_vocabulary_size])
        for x in range(gg.shape[0]):
            for y in range(gg.shape[1]):
                gg[x, y, batch_all_responses.int()[x, y]] = 1

        optimizer.zero_grad()

        word_symbol_probabilties_tensor, probability_word_knowledge_tensor, triple_weights= mymodel(sparse_sentence_processed_subgraphs_gpu,word_embedding_lookup_gpu,sparse_sentence_processed_subgraphs)

        # for x in sentence_processed_subgraphs_torch:
        #     del x
        #
        # sentence_processed_subgraphs_torch.clear()
        # torch.cuda.empty_cache()

        # anything greater than 0 should be 1
        j = torch.where(q_vals > 0, 1.0, 0.0).to(device)
        computed_loss = loss(probability_word_knowledge_tensor, j) + loss(word_symbol_probabilties_tensor,
                                                                          gg.to(device))
        print(f"Loss: {computed_loss.item()} PPL: {torch.exp(computed_loss).item()} ")
        computed_loss.backward()
        #optimizer.step()
    file_saver = save_directory + "epoch_" + str(epoch) + ".pt"
    print(f"End epoch {epoch} saving " + file_saver)
    #torch.save({
    #    'epoch': epoch,
    #    'model_state_dict': mymodel.state_dict(),
    #    'optimizer_state_dict': optimizer.state_dict(),
    #    'loss': computed_loss.item(),
    #    'ppl': torch.exp(computed_loss).item(),
    #}, "epoch_" + str(epoch) + ".pt")
    f.close()

'''
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    

for epoch in range(num_epochs):
    train(model, device, train_loader, optimizer, epoch)
'''
