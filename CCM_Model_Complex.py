import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d
import numpy as np
import random
import json
import gc
from tqdm import tqdm
import sys
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
num_accumulate_batches = 4
num_data_lines = 33700
num_batches = (num_data_lines // (batch_size*num_accumulate_batches))*num_accumulate_batches
num_epochs = 50

hrtw_embedding_sizes = [100, 50, 100, 300]  # H_embedding, R_embedding, T_embedding, W_embedding size.

dynamic_batch_sequence_length = [True,
                                 60]  # If true it evaluates max of all data in the set, else it sets to the second value in this list.
dynamic_batch_subgraph_triples_size = [True, 25]

# No dropouts detected

Graph_encoder_internal_size = 100  # num_trans_units

gru_encoder_hidden_size = 256
gru_decoder_hidden_size = 256
gru_encoder_num_layers = 2
gru_decoder_num_layers = 2

Graph_Attention_Top_V_size = 64  # units -> num_units (model) -> attnV (attention_decoder)
Encoder_Attention_V_size = 256

#base_directory = "/home/chris/Documents/columbia/fall_22/dialog_project/CCM_Torch/Data/"
base_directory = "./Data/"
save_directory = "./Model_save_dir/"

train_file = base_directory + "Post_processed_data/trainset.txt"
# graph_embedding_lookup_file = "C:/Users/manue/Downloads/ConvAI Assignment1/Projects/CCM_Torch/Data/Post_processed_data/embedding_lookup.txt"
test_file = base_directory + "Post_processed_data/testset.txt"
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
token_to_symbol_weight = token_to_symbol_dictionary['pad']

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
        self.head_mlp = ComplexLinear(entity_embedding_size, Graph_encoder_internal_size)
        self.tail_mlp = ComplexLinear(entity_embedding_size, Graph_encoder_internal_size)
        self.rel_mlp = ComplexLinear(rel_embedding_size, Graph_encoder_internal_size)

    def forward(self, All_subgraph_embeds, slice_ids):
        #print(All_subgraph_embeds[:,:,:,slice_ids[1]:slice_ids[2]].dtype)
        rh = self.rel_mlp(All_subgraph_embeds[:,:,:,slice_ids[1]:slice_ids[2]])
        hh = self.head_mlp(All_subgraph_embeds[:,:,:,0:slice_ids[0]])
        th = self.tail_mlp(All_subgraph_embeds[:,:,:,slice_ids[0]:slice_ids[1]])
        beta = torch.sum(torch.mul(rh, torch.conj(complex_relu(torch.add(hh, th)))), dim=3)
        beta = beta.abs()
        alpha = torch.softmax(beta, dim=2)
        g = torch.sum(torch.mul(alpha.unsqueeze(3), All_subgraph_embeds[:,:,:,0:slice_ids[1]]),dim=2).abs()
        return g


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

    def forward(self, Decoder_Hidden_State, Batch_decoder_input_hh, Batch_Decoder_Input_Attender):
        decoder_output_hh = self.decoder_hidden_state_mlp(Decoder_Hidden_State)
        beta = self.decoder_mapper_mlp(torch.add(decoder_output_hh.unsqueeze(1), Batch_decoder_input_hh))
        alpha = torch.softmax(beta, dim=1)
        c_bahadanau = torch.sum(torch.mul(alpha, Batch_Decoder_Input_Attender), dim=1)
        return c_bahadanau, alpha


class Graph_Attention_Hierarchy_Triples(nn.Module):
    def __init__(self, decoder_hidden_size, all_knowledge_triple_embedding_size):
        super().__init__()
        self.decoder_mapper_mlp = ComplexLinear(decoder_hidden_size, all_knowledge_triple_embedding_size)

    def forward(self, decoder_hidden_state, alpha_graph_attention_top, all_embeddings, sentence_ids):
        #print(decoder_hidden_state.shape)
        decoder_hidden_state_complex = torch.zeros(decoder_hidden_state.shape[0],decoder_hidden_state.shape[1],dtype=torch.cfloat).to(device)
        decoder_hidden_state_complex.real = decoder_hidden_state
        intermediate = torch.conj(self.decoder_mapper_mlp(decoder_hidden_state_complex))
        beta = torch.sum(torch.mul(all_embeddings, intermediate.unsqueeze(1).unsqueeze(2)),dim=3).abs()  # Should it be mul or add?. It should be Mul, we are multiplying with kj and transpose which we model as sum over mul.
        alpha = torch.mul(alpha_graph_attention_top, torch.softmax(beta, dim=2))
        c_hierarchical = torch.sum(torch.sum(torch.mul(alpha.unsqueeze(3), all_embeddings), dim=2), dim=1).abs()
        predicted_subgraph_token_index = torch.argmax(torch.max(alpha, dim=2).values, dim=1)
        predicted_triple_index_in_subgraph = torch.argmax(alpha[sentence_ids, predicted_subgraph_token_index, :], dim=1)
        return c_hierarchical, predicted_subgraph_token_index, predicted_triple_index_in_subgraph

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

        self.entity_embedding_size = all_embedding_sizes_as_tuples[0]//2
        self.relation_embedding_size = all_embedding_sizes_as_tuples[1]
        self.embedding_size_slices = [self.entity_embedding_size,2*self.entity_embedding_size,self.relation_embedding_size + 2*self.entity_embedding_size]
        word_embedding_size = all_embedding_sizes_as_tuples[3]

        encoder_input_size = word_embedding_size + 2 * self.entity_embedding_size
        self.gru_encoder_num_layers = gru_encoder_num_layers
        self.encoder = nn.GRU(encoder_input_size, gru_encoder_hidden_size, self.gru_encoder_num_layers,
                              batch_first=True)

        Graph_Encoder_Attention_output_size = 2 * self.entity_embedding_size
        self.my_graph_attention = Graph_Encoder_Attention(self.entity_embedding_size, self.relation_embedding_size,
                                                          Graph_encoder_internal_size)

        self.my_graph_attention_top_common = Bahadanau_Attention_Common(Graph_Encoder_Attention_output_size,
                                                                        Graph_Attention_Top_V_size)
        self.my_graph_attention_top = Bahadanau_Attention(gru_decoder_hidden_size, Graph_Attention_Top_V_size)

        Encoder_attention_output_size = gru_encoder_hidden_size
        self.my_encoder_attention_common = Bahadanau_Attention_Common(gru_encoder_hidden_size, Encoder_Attention_V_size)
        self.my_encoder_attention = Bahadanau_Attention(gru_decoder_hidden_size, Encoder_Attention_V_size)

        self.all_knowledge_triple_embedding_size = self.relation_embedding_size + 2 * self.entity_embedding_size  # called in a few places so best to use self.values
        self.my_graph_attention_triples = Graph_Attention_Hierarchy_Triples(gru_decoder_hidden_size,
                                                                            self.all_knowledge_triple_embedding_size)

        decoder_input_size = Graph_Encoder_Attention_output_size + (
                    2 * self.all_knowledge_triple_embedding_size) + Encoder_attention_output_size + word_embedding_size
        self.decoder_multilayer_cell = MultiLayer_GRU_Cell(decoder_input_size, gru_decoder_hidden_size,
                                                           gru_decoder_num_layers)

        self.word_vocabulary_size = word_embedding_lookup_gpu.shape[0]
        word_generator_input_size = gru_decoder_hidden_size
        self.word_symbol_mlp = nn.Linear(word_generator_input_size, self.word_vocabulary_size, bias=False)
        self.probability_word_knowledge_mlp = nn.Linear(word_generator_input_size, 1, bias=False)

        self.word_vocab_appender_all_idx = self.word_vocabulary_size - word_vocabulary_appender_length[0] - 1
        self.word_vocab_appender_min_idx = self.word_vocabulary_size + word_vocabulary_appender_length[1] - \
                                           word_vocabulary_appender_length[0] - 1
                                           
        self.my_sigmoid_function = nn.Sigmoid()

        self.vocab_Softmax = nn.LogSoftmax(dim=1)

    def gen_output_word_knowledge_probabilites(self, input, word_lookup):
        #predicted_word_symbol_probabilities = self.vocab_Softmax(self.word_symbol_mlp(input))
        predicted_word_symbol_probabilities = self.vocab_Softmax(self.word_symbol_mlp(input))
        #predicted_word_symbol_probabilities_sum = torch.exp(torch.sum(torch.log(predicted_word_symbol_probabilities),dim=1))
        predicted_word_symbol_index = torch.argmax(predicted_word_symbol_probabilities,dim=1)  # Should already be present as int
        # predicted_word_symbol_index_no_appender = torch.argmax(
        #     self.word_symbol_mlp(input[:, 0:self.word_vocab_appender_all_idx]), dim=1)
        # predicted_word_symbol_index_min_appender = torch.argmax(
        #     self.word_symbol_mlp(input[:, 0:self.word_vocab_appender_min_idx]), dim=1)
        predicted_word_embedding = word_lookup[predicted_word_symbol_index, :]
        # print("predicted word embedding shape is")
        # print(predicted_word_embedding.shape)
        probability_word_knowledge = self.my_sigmoid_function(self.probability_word_knowledge_mlp(input))

        # return predicted_word_symbol_probabilities, predicted_word_symbol_index, predicted_word_embedding, probability_word_knowledge, predicted_word_symbol_index_no_appender, predicted_word_symbol_index_min_appender
        return predicted_word_symbol_probabilities, predicted_word_embedding, probability_word_knowledge, predicted_word_symbol_index#, predicted_word_symbol_probabilities_sum

    def forward(self, Batch_Graph_embeddings, Batch_Word_Embeddings, word_lookup, word_responses):
        sentence_ids = torch.arange(0, Batch_Graph_embeddings.shape[0]).to(device)
        #Batch_Graph_Embeddings = Batch_All_embeddings[0:3]  # h,r,t
        #Batch_Word_Embeddings = Batch_All_embeddings[3]

        g1 = self.my_graph_attention(Batch_Graph_embeddings,self.embedding_size_slices)  # h,t,r in the input

        encoder_inp = torch.cat((Batch_Word_Embeddings, g1), dim=2)
        encoded_all, encoder_fin = self.encoder(encoder_inp)

        g_top_attention = self.my_graph_attention_top_common(g1)
        encoder_top_attention = self.my_encoder_attention_common(encoded_all)

        # decoder_hidden_state = torch.zeros(batch_size,gru_decoder_hidden_size) # initializing to zeros for entire batch initially
        decoder_layer_states = encoder_fin
        decoder_hidden_state = encoder_fin[self.gru_encoder_num_layers - 1]
        # decoder_word_state_pred = torch.zeros(gru_decoder_hidden_size).repeat(batch_size,1) # Need to initialize to CLS prediction (change from zeros by doing an embedding lookup), and it's token.
        # decoder_graph_pred = torch.zeros(self.all_knowledge_triple_embedding_size).repeat(batch_size,1) # Assume we are getting this from not a fact triple which is of zero values.
        #decoder_hidden_tensor = torch.zeros(batch_size, 0, gru_decoder_hidden_size).to(device)  # Should remove it later
        word_symbol_probabilties_tensor = torch.zeros(batch_size, 0).to(device)
        #word_symbol_probabilties_sum_tensor = torch.zeros(batch_size, 0).to(device)
        #word_symbol_index_tensor = torch.zeros(batch_size, 0).to(device)
        #word_symbol_index_no_appender_tensor = torch.zeros(batch_size, 0).to(device)
        #word_symbol_index_min_appender_tensor = torch.zeros(batch_size, 0).to(device)
        probability_word_knowledge_tensor = torch.zeros(batch_size, 0).to(device)
        
        predictions_tensor = torch.zeros(batch_size,0,3).to(device)
        
        #graph_token_predict_tensor = torch.zeros(batch_size, 0)
        #subgraph_token_predict_tensor = torch.zeros(batch_size, 0)

        # predicted_word_symbol_probabilities, predicted_word_symbol_index, predicted_word_embedding, probability_word_knowledge, predicted_word_symbol_index_no_appender, predicted_word_symbol_index_min_appender = self.gen_output_word_knowledge_probabilites(
        #     decoder_hidden_state)
        _ , predicted_word_embedding, probability_word_knowledge, _ = self.gen_output_word_knowledge_probabilites(
            decoder_hidden_state,word_lookup)
        predicted_knowledge_embedding = torch.zeros(batch_size, self.all_knowledge_triple_embedding_size).to(device)

        for seq_pos in range(0, Batch_Word_Embeddings.shape[1]):
            # print("current sequence position is" + str(seq_pos))
            c_g, alpha_graph_attention_top = self.my_graph_attention_top(decoder_hidden_state, g_top_attention, g1)
            c_hierarchical, predicted_subgraph_token_index, predicted_triple_index_in_subgraph = self.my_graph_attention_triples(decoder_hidden_state,
                                                                                           alpha_graph_attention_top,
                                                                                           Batch_Graph_embeddings,sentence_ids)

            c_e, _ = self.my_encoder_attention(decoder_hidden_state, encoder_top_attention,
                                                                     encoded_all)

            decoder_hidden_state, decoder_layer_states = self.decoder_multilayer_cell(
                torch.cat((c_g, c_hierarchical, c_e, predicted_word_embedding, predicted_knowledge_embedding), dim=1),
                decoder_layer_states)

            # predicted_word_symbol_probabilities, predicted_word_symbol_index, predicted_word_embedding, probability_word_knowledge, predicted_word_symbol_index_no_appender, predicted_word_symbol_index_min_appender = self.gen_output_word_knowledge_probabilites(
            #     decoder_hidden_state)

            predicted_word_symbol_probabilities, predicted_word_embedding, probability_word_knowledge, predicted_word_symbol_index = self.gen_output_word_knowledge_probabilites(decoder_hidden_state,word_lookup)

            predicted_knowledge_embedding = Batch_Graph_embeddings[sentence_ids, predicted_subgraph_token_index, predicted_triple_index_in_subgraph].abs()
            # print("predicted knowledge embedding shape is")
            # print(predicted_knowledge_embedding.shape)
            #print(predicted_word_symbol_probabilities[sentence_ids,word_responses[:,seq_pos].long()])
            #print(predicted_word_symbol_probabilities[sentence_ids,word_responses[:,seq_pos].long()].shape)
            
            predictions_tensor = torch.cat((predictions_tensor,torch.cat((predicted_word_symbol_index.unsqueeze(1).unsqueeze(2),predicted_subgraph_token_index.unsqueeze(1).unsqueeze(2),predicted_triple_index_in_subgraph.unsqueeze(1).unsqueeze(2)),dim=2)),dim=1)

            #decoder_hidden_tensor = torch.cat((decoder_hidden_tensor, decoder_hidden_state.unsqueeze(1)), dim=1)
            word_symbol_probabilties_tensor = torch.cat(
                (word_symbol_probabilties_tensor, predicted_word_symbol_probabilities[sentence_ids,word_responses[:,seq_pos]].unsqueeze(1)), dim=1)
            #word_symbol_probabilties_sum_tensor = torch.cat(
            #    (word_symbol_probabilties_sum_tensor, predicted_word_symbol_probabilities_sum.unsqueeze(1)), dim=1)
            #word_symbol_index_tensor = torch.cat((word_symbol_index_tensor, predicted_word_symbol_index.unsqueeze(1)),dim=1)
            #word_symbol_index_no_appender_tensor = torch.cat(
            #    (word_symbol_index_no_appender_tensor, predicted_word_symbol_index_no_appender.unsqueeze(1)), dim=1)
            #word_symbol_index_min_appender_tensor = torch.cat(
            #    (word_symbol_index_min_appender_tensor, predicted_word_symbol_index_min_appender.unsqueeze(1)), dim=1)
            probability_word_knowledge_tensor = torch.cat((probability_word_knowledge_tensor, probability_word_knowledge),dim=1)  # No need for unsqueeze as MLP automatically adds last dimension as 1
            # print("Predicted predicted_word_symbol_index tensor dimension, predicted_word_embedding dimension, probability_word_knowledge tensor dimension")
            # print(word_symbol_index_tensor.shape)
            # print(predicted_word_embedding.shape)
            # print(probability_word_knowledge_tensor.shape)
            # print("predicted probability word knowledge shape is")
            # print(probability_word_knowledge.shape)
            #graph_token_predict_tensor = torch.cat((graph_token_predict_tensor, predicted_subgraph_token_index.unsqueeze(1).to('cpu')),dim=1)
            #subgraph_token_predict_tensor = torch.cat((subgraph_token_predict_tensor, predicted_triple_index_in_subgraph.unsqueeze(1).to('cpu')),dim=1)

        #return decoder_hidden_tensor, word_symbol_probabilties_tensor, word_symbol_index_tensor, probability_word_knowledge_tensor, alpha_hierarchical_attention, alpha_encoder_attention, word_symbol_index_no_appender_tensor, word_symbol_index_min_appender_tensor
        return word_symbol_probabilties_tensor, probability_word_knowledge_tensor, predictions_tensor # word_symbol_index_tensor, graph_token_predict_tensor, subgraph_token_predict_tensor#, word_symbol_probabilties_sum_tensor#, graph_token_predict_tensor, subgraph_token_predict_tensor


def generate_batches(train_file, batch_size=512):
    curr = []

    with open(train_file, 'r') as fin:
        for line in fin.readlines():
            curr.append(json.loads(line))

            if len(curr) == batch_size:
                yield curr
                curr = []

    return None


# f = open(graph_embedding_lookup_file,"r")
# h = f.readlines()
# graph_embedding_lookup = my_preprocessor_ini.generate_lookup_tensor(h,'graph_embeddings')
# gc.collect()

word_vocabulary_size = word_embedding_lookup.shape[0]

mymodel = CCM_Model(Graph_encoder_internal_size, gru_encoder_hidden_size, gru_encoder_num_layers,
                    Encoder_Attention_V_size, Graph_Attention_Top_V_size, gru_decoder_hidden_size,
                    gru_decoder_num_layers, hrtw_embedding_sizes,
                    word_vocabulary_appender_length).to(device)
loss_word_symbol_probabilities = torch.nn.NLLLoss()
loss_word_knowledge_probabilities = torch.nn.BCELoss()
learning_rate = 1e-3
optimizer = optim.Adam(mymodel.parameters(), lr=learning_rate)

#mymodel.load_state_dict(torch.load('./epoch_1.pt')['model_state_dict'])
#optimizer.load_state_dict(torch.load('./epoch_1.pt')['optimizer_state_dict'])

def train(epoch_begin,epoch_end):
    if(epoch_begin != 0):
        mymodel.load_state_dict(torch.load(save_directory + "epoch_" + str(epoch_begin-1) + ".pt")['model_state_dict'])
        optimizer.load_state_dict(torch.load(save_directory + "epoch_" + str(epoch_begin-1) + ".pt")['optimizer_state_dict'])
        computed_loss = torch.load(save_directory + "epoch_" + str(epoch_begin-1) + ".pt")['loss']
    
    mymodel.train()
    
    for epoch in range(epoch_begin, epoch_end):
        f = open(train_file, "r")
        # print("epoch is" + str(epoch))
        for batch in tqdm(range(0, num_batches)):
            h = []
            for sentence_id in range(batch_size):
                g = f.readline()
                h.append(json.loads(g))

            max_post_length = max([len(x['post']) for x in h])
            max_response_length = max([len(x['response']) for x in h])
            current_batch_length = max([max_post_length, max_response_length]) + 4  # POS

            my_preprocessor = batch_tensor_input_gen(current_batch_length,
                                                 dynamic_batch_subgraph_triples_size[1], hrtw_embedding_sizes)
            batchwise_all_token_embeddings, q_vals, batch_all_responses, sparse_sentence_processed_subgraphs = my_preprocessor.sentence_embed_gen(h, batch_size)
            
            Complex1 = torch.zeros(sparse_sentence_processed_subgraphs[0].shape[0],sparse_sentence_processed_subgraphs[0].shape[1],sparse_sentence_processed_subgraphs[0].shape[2],sparse_sentence_processed_subgraphs[0].shape[3]//2, dtype = torch.cfloat);
            Complex1.Real = sparse_sentence_processed_subgraphs[0].to_dense()[:,:,:,0:sparse_sentence_processed_subgraphs[0].shape[3]//2]
            Complex1.Imag = sparse_sentence_processed_subgraphs[0].to_dense()[:,:,:,sparse_sentence_processed_subgraphs[0].shape[3]//2:sparse_sentence_processed_subgraphs[0].shape[3]]
            Complex2 = torch.zeros(sparse_sentence_processed_subgraphs[0].shape[0],sparse_sentence_processed_subgraphs[0].shape[1],sparse_sentence_processed_subgraphs[0].shape[2],sparse_sentence_processed_subgraphs[0].shape[3]//2, dtype = torch.cfloat);
            Complex2.Real = sparse_sentence_processed_subgraphs[2].to_dense()[:,:,:,0:sparse_sentence_processed_subgraphs[0].shape[3]//2]
            Complex2.Imag = sparse_sentence_processed_subgraphs[2].to_dense()[:,:,:,sparse_sentence_processed_subgraphs[0].shape[3]//2:sparse_sentence_processed_subgraphs[0].shape[3]]
            Complex3 = torch.zeros(sparse_sentence_processed_subgraphs[0].shape[0],sparse_sentence_processed_subgraphs[0].shape[1],sparse_sentence_processed_subgraphs[0].shape[2],sparse_sentence_processed_subgraphs[0].shape[3]//2, dtype = torch.cfloat);
            Complex3.Real = sparse_sentence_processed_subgraphs[1].to_dense()[:,:,:,0:sparse_sentence_processed_subgraphs[0].shape[3]//2]
        
            batchwise_concatenated_data = torch.cat((Complex1,Complex2,Complex3),dim=3).to(device)
            #print(batchwise_concatenated_data.dtype)

            batchwise_all_token_embeddings = batchwise_all_token_embeddings.to(device)
            batch_all_responses_gpu = batch_all_responses.long().to(device)
            batch_all_responses_weights = torch.where(batch_all_responses_gpu == token_to_symbol_weight, 0, 1)
            batch_all_responses_weights_complement = torch.where(batch_all_responses_gpu == token_to_symbol_weight, 0.85, 0)
            #b = torch.sum(batch_all_responses_weights,dim=1)
            #vertical_ids = torch.zeros(0)
            #horizontal_ids = torch.zeros(0)
            #for vertical_id in range(0,b.shape[0]):
            #    vertical_ids = torch.cat((vertical_ids,vertical_id*torch.ones(b[vertical_id])),dim=0)
            #    horizontal_ids = torch.cat((horizontal_ids,torch.arange(0,b[vertical_id])),dim=0)

            word_symbol_probabilties_tensor, probability_word_knowledge_tensor, predictions = mymodel(batchwise_concatenated_data,batchwise_all_token_embeddings,word_embedding_lookup_gpu,batch_all_responses_gpu)
            #valid_word_symbol_probabilities = word_symbol_probabilties_tensor[vertical_ids.long(),horizontal_ids.long()]
        
            valid_word_symbol_probabilities = torch.add(torch.mul(word_symbol_probabilties_tensor,batch_all_responses_weights),batch_all_responses_weights_complement)

            # for x in sentence_processed_subgraphs_torch:
            #     del x
            #
            # sentence_processed_subgraphs_torch.clear()
            # torch.cuda.empty_cache()
            q_val_target = torch.where(q_vals > 0, 1.0, 0.0).to(device)
            word_predict_target = torch.zeros(batch_size,current_batch_length).long().to(device)
            #word_predict_target = torch.zeros(valid_word_symbol_probabilities).long().to(device)
        
            #word_predict_target = torch.cat((torch.ones(batch_size,current_batch_length).unsqueeze(2),torch.zeros(batch_size,current_batch_length).unsqueeze(2)),dim=2).to(device)
            word_regularizer = torch.div(torch.sum(torch.add(0.85*batch_all_responses_weights,batch_all_responses_weights_complement)),torch.sum(1.15*batch_all_responses_weights))
            #print("loss scaling is")
            #print(loss_scaling)
            
            #knowledge_regularizer = torch.sum(q_val_target)

            # anything greater than 0 should be 1
            if(batch%num_accumulate_batches == 0):
                optimizer.zero_grad()
            
            if(batch%100 == 0):
                print(predictions[:,:,0])
            
            #print(word_symbol_probabilties_tensor.view(-1,1))
            #print(word_symbol_probabilties_tensor.view(-1,1).shape)
            #print(loss_word_symbol_probabilities(valid_word_symbol_probabilities.view(-1,1),word_predict_target.view(-1)))
            computed_loss = torch.mul((0.15*current_batch_length*loss_word_knowledge_probabilities(probability_word_knowledge_tensor.view(-1), q_val_target.view(-1)) + loss_word_symbol_probabilities(valid_word_symbol_probabilities.view(-1,1),word_predict_target.view(-1))),word_regularizer)
            computed_loss.backward()
            if(batch%num_accumulate_batches == num_accumulate_batches-1):
                optimizer.step()
            
            print(f"Loss: {computed_loss.item()} PPL: {torch.exp(computed_loss).item()} ")
        
        
        #file_saver = save_directory + "epoch_" + str(epoch) + ".pt"
        print(f"End epoch {epoch} saving " + str(epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': mymodel.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': computed_loss,
            'ppl': torch.exp(computed_loss).item(),
        }, save_directory + "epoch_" + str(epoch) + ".pt")
        #torch.save(mymodel.state_dict(), save_directory + "epoch_model_only_" + str(epoch) + ".pt")
        f.close()
        
        f = open(test_file, "r")
        for batch1 in range(0,3):
            h = []
            for sentence_id in range(40):
                g = f.readline()
                h.append(json.loads(g))
            
            max_post_length = max([len(x['post']) for x in h])
            max_response_length = max([len(x['response']) for x in h])
            current_batch_length = max([max_post_length, max_response_length]) + 4  # POS
            
            my_preprocessor = batch_tensor_input_gen(current_batch_length,
                                                 dynamic_batch_subgraph_triples_size[1], hrtw_embedding_sizes)
            batchwise_all_token_embeddings, q_vals, batch_all_responses, sparse_sentence_processed_subgraphs = my_preprocessor.sentence_embed_gen(h, batch_size)
            
            Complex1 = torch.zeros(sparse_sentence_processed_subgraphs[0].shape[0],sparse_sentence_processed_subgraphs[0].shape[1],sparse_sentence_processed_subgraphs[0].shape[2],sparse_sentence_processed_subgraphs[0].shape[3]//2, dtype = torch.cfloat);
            Complex1.Real = sparse_sentence_processed_subgraphs[0].to_dense()[:,:,:,0:sparse_sentence_processed_subgraphs[0].shape[3]//2]
            Complex1.Imag = sparse_sentence_processed_subgraphs[0].to_dense()[:,:,:,sparse_sentence_processed_subgraphs[0].shape[3]//2:sparse_sentence_processed_subgraphs[0].shape[3]]
            Complex2 = torch.zeros(sparse_sentence_processed_subgraphs[0].shape[0],sparse_sentence_processed_subgraphs[0].shape[1],sparse_sentence_processed_subgraphs[0].shape[2],sparse_sentence_processed_subgraphs[0].shape[3]//2, dtype = torch.cfloat);
            Complex2.Real = sparse_sentence_processed_subgraphs[2].to_dense()[:,:,:,0:sparse_sentence_processed_subgraphs[0].shape[3]//2]
            Complex2.Imag = sparse_sentence_processed_subgraphs[2].to_dense()[:,:,:,sparse_sentence_processed_subgraphs[0].shape[3]//2:sparse_sentence_processed_subgraphs[0].shape[3]]
            Complex3 = torch.zeros(sparse_sentence_processed_subgraphs[0].shape[0],sparse_sentence_processed_subgraphs[0].shape[1],sparse_sentence_processed_subgraphs[0].shape[2],sparse_sentence_processed_subgraphs[0].shape[3]//2, dtype = torch.cfloat);
            Complex3.Real = sparse_sentence_processed_subgraphs[1].to_dense()[:,:,:,0:sparse_sentence_processed_subgraphs[0].shape[3]//2]
        
            batchwise_concatenated_data = torch.cat((Complex1,Complex2,Complex3),dim=3).to(device)
            #print(batchwise_concatenated_data.dtype)

            batchwise_all_token_embeddings = batchwise_all_token_embeddings.to(device)
            batch_all_responses_gpu = batch_all_responses.long().to(device)
            batch_all_responses_weights = torch.where(batch_all_responses_gpu == token_to_symbol_weight, 0, 1)
            batch_all_responses_weights_complement = torch.where(batch_all_responses_gpu == token_to_symbol_weight, 0.85, 0)
            #b = torch.sum(batch_all_responses_weights,dim=1)
            #vertical_ids = torch.zeros(0)
            #horizontal_ids = torch.zeros(0)
            #for vertical_id in range(0,b.shape[0]):
            #    vertical_ids = torch.cat((vertical_ids,vertical_id*torch.ones(b[vertical_id])),dim=0)
            #    horizontal_ids = torch.cat((horizontal_ids,torch.arange(0,b[vertical_id])),dim=0)

            word_symbol_probabilties_tensor, probability_word_knowledge_tensor, predictions = mymodel(batchwise_concatenated_data,batchwise_all_token_embeddings,word_embedding_lookup_gpu,batch_all_responses_gpu)
            
            print(word_symbol_probabilties_tensor)
            print(predictions[:,:,0])
            
            np.save(save_directory + "predictions_" + str(epoch) + "_batch_" + str(batch1) + ".npy", predictions.cpu().numpy())
            np.save(save_directory + "probabilities_" + str(epoch) + "_batch_" + str(batch1) + ".npy", word_symbol_probabilties_tensor.detach().cpu().numpy())
        
        print("end epoch results")

    # f = open(train_file,"r")
    # sentence_processed_subgraphs.append(np.random.rand(batch_size,dynamic_batch_sequence_length[1],hrtw_embedding_sizes[3]))

train(2,num_epochs)