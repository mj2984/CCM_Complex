using DelimitedFiles
using CSV
using DataFrames
using SparseArrays

base_folder = "C:/Users/manue/Downloads/ConvAI Assignment1/Projects/CCM_Torch/Data/"

file_entity = base_folder * "Knowledge_Data/entity.txt"
file_relation = base_folder * "Knowledge_Data/relation.txt"
entities = DelimitedFiles.readdlm(file_entity, String)[:,1];
entities = hcat(entities,collect(1:size(entities,1)));
entity_set = Set(entities[:,1]);

embedding_size = 300;

Relation_Properties_file = base_folder * "Knowledge_Data/RelationFiles/RelationAnnotations.csv";
AllRelationCategories = DataFrame(CSV.File(Relation_Properties_file));
AllRelations = AllRelationCategories."Relation" ∪ filter(x -> coalesce(x,"<self>") != "<self>", AllRelationCategories."inversion");

AntiSymmetricRelations = filter("Antisymmetric" => x -> coalesce(x,false) == true, AllRelationCategories)."Relation";

input_file = base_folder * "Knowledge_Data/RelationFiles/non_Augmented/train1.tsv";
All_triples = DelimitedFiles.readdlm(input_file);

valid_indexes_i = [];
valid_indexes_j = [];
valid_indexes_k = [];
anti_symmetric_indexes_i = [];
anti_symmetric_indexes_j = [];
anti_symmetric_indexes_k = [];

nodenames = ["index i", "index j", "index k", "value"]
SparseRelationTensorDataframe = DataFrame([[] for _ = nodenames] , nodenames)

for (head,relation,tail) in eachrow(All_triples)
	index_i_val = findfirst(x -> x == head,entities[:,1]);
	index_k_val = findfirst(x -> x == relation,AllRelations);
	index_j_val = findfirst(x -> x == tail,entities[:,1]);
	if((isnothing(index_i_val) || isnothing(index_k_val) || isnothing(index_j_val)) == false)
		push!(SparseRelationTensorDataframe,[index_i_val, index_j_val, index_k_val, 1]);
		if(relation ∈ AntiSymmetricRelations)
			push!(SparseRelationTensorDataframe,[index_j_val, index_i_val, index_k_val, -1]);
		end
	end
	if(size(SparseRelationTensorDataframe)[1] % 1000 == 0)
		print("done 1000")
	end
end

SparseTensor = Vector{SparseMatrixCSC}(undef,length(AllRelations));
RelationPredictionTensor = Vector{Vector{ComplexF64}}(undef,length(AllRelations));
#We are looking at only diagonal elements

for z_index in 1:length(AllRelations)
	current_tensor_data = filter("index k" => x -> x == z_index, SparseRelationTensorDataframe)
	SparseTensor[z_index] = sparse(current_tensor_data."index i",current_tensor_data."index j",current_tensor_data."value")
	RelationPredictionTensor[z_index] = rand(ComplexF64,embedding_size);
end

A = zeros(ComplexF64,(size(entities)[1],embedding_size));

SymmetricRelations = filter("Symmetric" => x -> coalesce(x,false) == true, AllRelationCategories)."Relation";
SymmetricRelationIndexes = map(y -> findfirst(x -> x == y,AllRelations),SymmetricRelations);
filter("index k" => x -> x ∈ SymmetricRelationIndexes,SparseRelationTensorDataframe)