using CSV
using DataFrames
using JSON3
using ArgParse
using DelimitedFiles

augmentation = "Augmented"
embedding_style = "RotatE"

base_folder = "./Data/"
file_resource = base_folder * "resource.txt";
output_file = base_folder * "Post_processed_data/trainset.txt";
output_embedding_lookup_file = base_folder * "Post_processed_data/embedding_lookup.txt";
file_vs = base_folder * "Pre_processed_data/trainset.txt";
relation_dictionary_file = base_folder * "Knowledge_Data/relation.dict";
entity_embeddings_file = base_folder * "Knowledge_Data/Embeddings/" * embedding_style * "/" * augmentation * "/entity_" * embedding_style * ".txt";
relation_embeddings_file = base_folder * "Knowledge_Data/Embeddings/" * embedding_style * "/" * augmentation * "/relation_" * embedding_style * ".txt";
all_entity_array = readlines(entity_embeddings_file);
all_relations_array = readlines(relation_embeddings_file);

delimiter_in = ", ";

resource_dicts = JSON3.read.(readlines(file_resource))[1];
entity_dict = resource_dicts["dict_csk_entities"];
relations_dict = JSON3.read(readline(relation_dictionary_file));
triples_dict = resource_dicts["dict_csk_triples"];

function generate_states(entity_relation_triples,entity_id_lut,entity_id_embedding,relation_id_lut,relation_id_embedding)
    delimiter_in = ", ";
    g = split(String(entity_relation_triples),delimiter_in);
    out = Array{String,1}(undef,3);
    if((max(entity_id_lut[g[1]],entity_id_lut[g[3]]) < length(entity_id_embedding)) & (relation_id_lut[g[2]] < length(relation_id_embedding)))
        out[1] = entity_id_embedding[entity_id_lut[g[1]] + 1];
        out[3] = entity_id_embedding[entity_id_lut[g[3]] + 1];
        out[2] = relation_id_embedding[relation_id_lut[g[2]] + 1];
    else
        out .= "out of bounds";
    end
    embeddings_as_dict = Dict("h_emb"=>out[1],"r_emb"=>out[2],"t_emb"=>out[3])
    return out, embeddings_as_dict
end

reversed_triples_dict_with_missing = Dict(value => generate_states(key,entity_dict,all_entity_array,relations_dict,all_relations_array)[1] for (key, value) in triples_dict);
reversed_triples_dict = filter(((k,v),) -> v[1] != "out of bounds", reversed_triples_dict_with_missing);

reversed_triples_dict_with_missing_writer = Dict(value => generate_states(key,entity_dict,all_entity_array,relations_dict,all_relations_array)[2] for (key, value) in triples_dict);

embedding_writer = open(output_embedding_lookup_file,"w");

for i in collect((0:(length(reversed_triples_dict_with_missing_writer)-1)))
    if(reversed_triples_dict_with_missing_writer[i]["t_emb"] == "out of bounds")
        println("skipped dict id " * string(i))
    else
        JSON3.write(embedding_writer,reversed_triples_dict_with_missing_writer[i]);
        write(embedding_writer,"\n");
    end
end

close(embedding_writer)

all_data_json = JSON3.read.(readlines(file_vs));

output_writer = open(output_file,"w");

#global data_line_index_in = 0;
#global data_line_index_out = 0;
for (data_line_index_in,current_data_json) in enumerate(all_data_json)
    h_emb = Vector{Any}();
    t_emb = Vector{Any}();
    r_emb = Vector{Any}();
    emb_matcher = Vector{Any}();
    token_level_triple_remover = Vector{Any}();
    post_triples_ini = current_data_json["post_triples"];
    if(all(x-> x ∈ keys(reversed_triples_dict),current_data_json["match_triples"]))
        #data_line_index_out = data_line_index_out + 1;
        num_removed_triple_sets = 0;
        matching_triple_embeds = map(x->reversed_triples_dict[x],current_data_json["match_triples"]);
        h_matching = map(x->x[1],matching_triple_embeds);
        t_matching = map(x->x[3],matching_triple_embeds);
        r_matching = map(x->x[2],matching_triple_embeds);
        for (all_triple_id,token_level_triple) in enumerate(current_data_json["all_triples"])
            triples_temp = map(x->reversed_triples_dict[x],filter(a -> a ∈ keys(reversed_triples_dict),token_level_triple));
            if(!isempty(triples_temp))
                h_temp = map(x->x[1],triples_temp);
                t_temp = map(x->x[3],triples_temp);
                r_temp = map(x->x[2],triples_temp);
                emb_matcher_temp = map(x->x∈current_data_json["match_triples"] ? 1 : 0,token_level_triple);
                push!(h_emb,h_temp);
                push!(t_emb,t_temp);
                push!(r_emb,r_temp);
                push!(emb_matcher,emb_matcher_temp);
            else
                push!(token_level_triple_remover,all_triple_id);
                post_triples_ini = map(x-> (((x+num_removed_triple_sets) > all_triple_id) ? (x-1) : ((x+num_removed_triple_sets) == all_triple_id) ? 0 : x), post_triples_ini);
                num_removed_triple_sets = num_removed_triple_sets + 1;
            end
        end
        if(num_removed_triple_sets > 0)
            #println("token level triple set removed at line " * string(data_line_index_out))
            println("token level triple set removed at line " * string(data_line_index_in))
        end
        response_triples_local_match_ids = map(x-> (x == -1) ? x : findfirst(y -> y == x,current_data_json["match_triples"]),current_data_json["response_triples"]);
        out_dict = Dict("h_emb"=> h_emb, "h_matching"=> h_matching, "t_emb"=> t_emb, "t_matching"=> t_matching, "r_emb"=> r_emb, "r_matching"=> r_matching,
                        "emb_matcher" => emb_matcher,
                        "token_level_triple_remover" => token_level_triple_remover,
                        "match_index"=> current_data_json["match_index"],
                        "post_triples"=> post_triples_ini,
                        "response_triples"=> current_data_json["response_triples"],
                        "response_triples_local"=> response_triples_local_match_ids,
                        "post"=> current_data_json["post"],
                        "response"=> current_data_json["response"])
        JSON3.write(output_writer,out_dict);
        write(output_writer,"\n");
        if(data_line_index_in % 100 == 0)
            println("finished data line" * string(data_line_index_in))
        end
    else
        println("data_skipped" * string(data_line_index_in))
    end
end

close(output_writer)

#output_reader = open(output_file);
post_token_sizes = [];
response_token_sizes = [];
graph_lengths = [];

for line in eachline(output_file)
    current_line_data = JSON3.read(line);
    push!(graph_lengths,map(x -> length(x),current_line_data["h_emb"]));
    push!(post_token_sizes,length(current_line_data["post"]));
    push!(response_token_sizes,length(current_line_data["response"]));
end

largest_graph_per_token_all = map(x->findmax(x)[1],graph_lengths);
graph_size_for_sentence = map(x->sum(x),graph_lengths);

findmax(post_token_sizes)[1]
findmax(response_token_sizes)[1]
findmax(largest_graph_per_token_all)[1]
