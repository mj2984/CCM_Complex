using CSV
using DataFrames
using JSON3
using ArgParse
using DelimitedFiles

base_folder = "C:/Users/manue/Downloads/ConvAI Assignment1/Projects/CCM_Torch/Data/"
word_embedding_input_file = base_folder * "Pre_processed_data/glove.840B.300d.txt"
word_embedding_processed_list = base_folder * "Post_processed_data/word_processing/All_glove_words.txt"
word_vocabulary_to_append = base_folder * "add_words_vocabulary.txt";
found_vocab_dict_output_file = base_folder * "Post_processed_data/word_processing/word_vocab_glove_index.txt";
found_vocab_glove_subset = base_folder * "Post_processed_data/word_processing/word_vocab_glove_embeds.txt";
found_vocab_to_append_glove_subset = base_folder * "Post_processed_data/word_processing/word_vocab_to_append_glove_embeds.txt";
vocab_mapper_dict = base_folder * "Post_processed_data/word_processing/word_vocab_mapper.txt";

word_vocab_dict_output_file = base_folder * "Post_processed_data/word_dictionary.txt";
word_embedding_output_file = base_folder * "Post_processed_data/word_embeddings.txt";

file_resource = base_folder * "resource.txt";

resource_dicts = JSON3.read.(readlines(file_resource))[1];
word_vocab_dict_symbols = resource_dicts["vocab_dict"];
word_vocab_dict = Dict(string(key) => string(value) for (key,value) in word_vocab_dict_symbols);
if(length(word_vocab_dict_symbols) != length(word_vocab_dict))
    println("missed word vocabs during parsing")
end

word_vocab_to_append = readlines(word_vocabulary_to_append)
for word in word_vocab_to_append
	if(word ∈ keys(word_vocab_dict))
		print("removed " * word * " from dict. It will be appended later")
		delete!(word_vocab_dict,word)
	end
end

word_vocab_dict_reversed = Dict(value => key for (key,value) in word_vocab_dict);

#all_word_vocab = []
#for (line_id,sentence) in enumerate(eachline(word_embedding_input_file))
#    q = split(sentence," ")
#    push!(all_word_vocab,q[1])
#    if(line_id % 1000 == 0)
#       println(line_id)
#    end
#end

#writedlm(word_embedding_processed_list,all_word_vocab)

all_word_vocab = readlines(word_embedding_processed_list);

found_vocab = []
found_vocab_line_dict = Dict()
not_found_vocab = []

found_word_vocab_to_append = []
found_word_vocab_to_append_line_dict = Dict()
not_found_word_vocab_to_append = []

#for (word_id,word) in enumerate(collect(keys(word_vocab_dict)))
#	if(word ∈ all_word_vocab)
#		push!(found_vocab,word)
#		found_vocab_line_dict[word] = findall(x->x == word,all_word_vocab) #Need to enable this when getting the output dict file
#	else
#		push!(not_found_vocab,word)
#	end
#	if(word_id % 1000 == 0)
#		println(word_id)
#	end
#end

#found_vocab_dict_output_file_writer = open(found_vocab_dict_output_file,"w");
#JSON3.write(found_vocab_dict_output_file,found_vocab_line_dict)
#close(found_vocab_dict_output_file_writer)

for (word_id,word) in enumerate(word_vocab_to_append)
	if(word ∈ all_word_vocab)
		push!(found_word_vocab_to_append,word)
		found_word_vocab_to_append_line_dict[word] = findall(x->x == word,all_word_vocab)
	else
		push!(not_found_word_vocab_to_append,word)
	end
	if(word_id % 1000 == 0)
		println(word_id)
	end
end

if(length(not_found_word_vocab_to_append) != 0)
	println("missed " * string(length(not_found_word_vocab_to_append)) * " vocabulary words in the append file")
end

found_vocab_line_dict = JSON3.read(readline(found_vocab_dict_output_file))
found_vocab = String.(keys(found_vocab_line_dict))
not_found_vocab = collect(setdiff(Set(String.(keys(word_vocab_dict))),Set(found_vocab)))

#found_word_embeds = [];
#found_word_to_append_embeds = [];

#reader_lines_word = sort(collect(values(Dict(key=>value[1] for (key,value) in found_vocab_line_dict))));
#reader_lines_word_to_append = sort(collect(values(Dict(key=>value[1] for (key,value) in found_word_vocab_to_append_line_dict))));

#for (line_id,sentence) in enumerate(eachline(word_embedding_input_file))
#	if(line_id ∈ reader_lines_word)
#		push!(found_word_embeds,sentence)
#	end
#	if(line_id ∈ reader_lines_word_to_append)
#		push!(found_word_to_append_embeds,sentence)
#	end
#	if(line_id % 1000 == 0)
#		println(line_id)
#	end
#end

#writedlm(found_vocab_glove_subset,found_word_embeds)
#writedlm(found_vocab_to_append_glove_subset,found_word_to_append_embeds)

found_word_embeds = readlines(found_vocab_glove_subset)
found_word_to_append_embeds = readlines(found_vocab_to_append_glove_subset)
found_word_vocab_to_append = word_vocab_to_append

all_vocab_original_ids = unique(map(x->parse(Int64,x),(collect(values(word_vocab_dict)))))
found_original_ids = unique(map(x->parse(Int64,x),(collect(values(Dict(key=>word_vocab_dict[key] for key ∈ found_vocab))))))
not_found_original_ids = unique(map(x->parse(Int64,x),(collect(values(Dict(key=>word_vocab_dict[key] for key ∈ not_found_vocab))))))
# 327 missing token ids and 1127 missing tokens

#found_original_ids_shared = []
#found_original_ids_unshared = []

#for elem in found_original_ids
#	if elem ∈ not_found_original_ids
#		push!(found_original_ids_shared,elem)
#	else
#		push!(found_original_ids_unshared,elem)
#	end
#end

not_found_original_ids_shared = []
not_found_original_ids_unshared = []

for elem in not_found_original_ids
	if elem ∈ found_original_ids
		push!(not_found_original_ids_shared,elem)
	else
		push!(not_found_original_ids_unshared,elem)
	end
end

found_vocab_dict = Dict(key => word_vocab_dict[key] for key ∈ found_vocab)

embedding_mapper_dict = Dict(key => key for key in found_vocab)

for not_found_word in not_found_vocab
	t = word_vocab_dict[not_found_word]
	if(parse(Int64,t) ∉ not_found_original_ids_unshared)
		temp = []
		for word in found_vocab
			if(found_vocab_dict[word] == t)
				push!(temp,word)
			end
		end
		idx = rand(1:length(temp))
		#println(not_found_word)
		#println(temp)
		embedding_mapper_dict[not_found_word] = temp[idx]
	end
end
	
embedding_mapper_dict_file_writer = open(vocab_mapper_dict,"w");
JSON3.write(embedding_mapper_dict_file_writer,embedding_mapper_dict);
close(embedding_mapper_dict_file_writer);

found_word_array = map(x -> split(x)[1],found_word_embeds);
found_word_embedding_array = map(x -> join(split(x)[2:301],"\t"),found_word_embeds);
found_word_to_append_array = map(x -> split(x)[1],found_word_to_append_embeds);
found_word_embedding_to_append_array = map(x -> join(split(x)[2:301],"\t"),found_word_to_append_embeds);

final_vocab_array = unique(keys(embedding_mapper_dict))
append!(final_vocab_array,found_word_vocab_to_append)
final_vocab_dict = Dict()

vocab_word_idx = 0
for word in final_vocab_array
	final_vocab_dict[word] = vocab_word_idx
	vocab_word_idx = vocab_word_idx + 1
end

final_vocab_dict_writer = open(word_vocab_dict_output_file,"w");
JSON3.write(final_vocab_dict_writer,final_vocab_dict);
close(final_vocab_dict_writer);

final_vocab_embedding_writer = open(word_embedding_output_file,"w");
for (token_id,token) in enumerate(final_vocab_array)
	if(token ∈ found_word_vocab_to_append)
		#println(findall(y->y==token,found_word_to_append_array))
		embedding = found_word_embedding_to_append_array[findall(y->y==token,found_word_to_append_array)[1]]
	else
		#println(findall(y->y==token,found_word_array))
		embedding = found_word_embedding_array[findall(y->y==embedding_mapper_dict[token],found_word_array)[1]] # For non found tokens we need to refer to found tokens using dictionary
	end
	println(final_vocab_embedding_writer,embedding)
	if(token_id % 1000 == 0)
		println(token_id)
	end
end
close(final_vocab_embedding_writer)

# remove elements with t_unshared and make them unk. It is neither in vocab dict,
# nor has an eqivalent in glove vocab
# check for case sensitivity in unk