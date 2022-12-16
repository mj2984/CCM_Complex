using CSV
using DataFrames
using JSON3
using ArgParse
using DelimitedFiles

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--rel_properties"
            help = "Relation Properties File"
            arg_type = String
            default = "C:/Users/manue/Downloads/ConvAI Assignment1/Projects/CCM_Torch/Data/Knowledge_Data/RelationFiles/RelationAnnotations.csv"
        "--rel_entries"
            help = "Relation Entries File"
            arg_type = String
            default = "C:/Users/manue/Downloads/ConvAI Assignment1/Projects/CCM_Torch/Data/resource.txt"
        "--output_file"
            help = "Output Location"
            arg_type = String
            default = "C:/Users/manue/Downloads/ConvAI Assignment1/Projects/CCM_Torch/Data/Knowledge_Data/RelationFiles/train1.tsv"
		"--print_progress"
			help = "Printing_Progress_Enable_Disable"
			arg_type = Bool
			default = false
    end
    return parse_args(s)
end

file_entity = "C:/Users/manue/Downloads/ConvAI Assignment1/Projects/CCM_Torch/Data/Knowledge_Data/entity.txt"
file_relation = "C:/Users/manue/Downloads/ConvAI Assignment1/Projects/CCM_Torch/Data/Knowledge_Data/relation.txt"

parsed_args = parse_commandline()

Relation_Properties_file = parsed_args["rel_properties"];
Relation_Entries_file = parsed_args["rel_entries"];
output_file = parsed_args["output_file"];
output_write_file = open(output_file,"w");

entities = DelimitedFiles.readdlm(file_entity, String)[:,1];
entity_set = Set(entities);

AllRelationCategories = DataFrame(CSV.File(Relation_Properties_file));
AllRelations = AllRelationCategories."Relation" ∪ filter(x -> coalesce(x,"<self>") != "<self>", AllRelationCategories."inversion");
AllRelations_set = Set(AllRelations);

IsCurrentEntrySymmetric = ones(size(AllRelations,1));
IsCurrentEntryPurelyUnsymmetric = ones(size(AllRelations,1));

if(parsed_args["print_progress"] == true)
	print(AllRelations);
	print(IsCurrentEntrySymmetric);
end

SymmetricRelationCategories = filter("Symmetric" => x -> coalesce(x,false) == true, AllRelationCategories);
SymmetricRelations = SymmetricRelationCategories."Relation";

InvertibleAsymmetricRelationCategories = filter(["has_inverse", "Symmetric"] => (x,y) -> x == true && coalesce(y,false) == false, dropmissing(AllRelationCategories, "has_inverse"));
InvertibleAsymmetricRelations = InvertibleAsymmetricRelationCategories."Relation";

AntiSymmetricRelations = filter("Antisymmetric" => x -> coalesce(x,false) == true, AllRelationCategories)."Relation";

json_file_read = read(Relation_Entries_file,String);
json_contents = JSON3.read(json_file_read);
json_relations_set = Set(json_contents["csk_triples"]);
json_relations_set_augmented = Set(json_contents["csk_triples"]);

delimiter_in = ", ";
delimiter_out = "\t";

nodenames = ["index i", "index j", "index k", "value"];
SparseRelationTensorDataframe = DataFrame([[] for _ = nodenames] , nodenames);

augmented_entities = Set();
augmented_relations = Set();

find_entity_position(y) = findfirst(x -> x == y,entities);
function augment_and_find_entity_position(y)
	push!(augmented_entities,y)
	push!(entity_set,y)
	push!(entities,y)
	return(length(entities))
end

find_relation_position(y) = findfirst(x -> x == y,AllRelations);
function augment_and_find_relation_position(y)
	push!(augmented_relations,y)
	push!(AllRelations_set,y)
	push!(AllRelations,y)
	append!(IsCurrentEntrySymmetric, 0);
	append!(IsCurrentEntryPurelyUnsymmetric, 1);
	return(length(AllRelations))
end

for hrt_string in json_relations_set
      hrt_array = (head,relation,tail) = String.(split(hrt_string,delimiter_in));
	trh_string = join(reverse(hrt_array),delimiter_in);
	if(head != tail)
		(index_i_val,index_j_val) = map(x -> x ∈ entity_set ? find_entity_position(x) : augment_and_find_entity_position(x),[head,tail]);
	else
		index_i_val = index_j_val = head ∈ entity_set ? find_entity_position(head) : augment_and_find_entity_position(head);
	end
	index_k_val = relation ∈ AllRelations_set ? find_relation_position(relation) : augment_and_find_relation_position(relation);
	push!(SparseRelationTensorDataframe,[index_i_val, index_j_val, index_k_val, 1]);
	if (trh_string ∉ json_relations_set_augmented)
		IsCurrentEntrySymmetric[findall(x -> x == relation, AllRelations)] .= 0
		if(relation ∈ SymmetricRelations)
			push!(json_relations_set_augmented, trh_string);
			push!(SparseRelationTensorDataframe,[index_j_val, index_i_val, index_k_val, 1]);
		elseif(relation ∈ AntiSymmetricRelations)
			push!(SparseRelationTensorDataframe,[index_j_val, index_i_val, index_k_val, -1]);
		end
	else
		IsCurrentEntryPurelyUnsymmetric[findall(x -> x == relation, AllRelations)] .= 0
	end
	if(relation ∈ InvertibleAsymmetricRelations)
		inverse_relations = filter("Relation" => x -> x == relation, InvertibleAsymmetricRelationCategories)."inversion";
		for inverse_relation in inverse_relations
			inverse_trh_array = reverse(hrt_array);
			inverse_trh_array[2] = inverse_relation;
			inverse_trh_string = join(inverse_trh_array,delimiter_in);
			if (inverse_trh_string ∉ json_relations_set_augmented)
				index_k_val_inverse = inverse_relation ∈ AllRelations_set ? find_relation_position(inverse_relation) : augment_and_find_relation_position(inverse_relation);
				push!(json_relations_set_augmented, inverse_trh_string);
				push!(SparseRelationTensorDataframe,[index_j_val, index_i_val, index_k_val_inverse, 1]);
			end
		end
	end
	if(size(SparseRelationTensorDataframe)[1] % 1000 == 1)
		if(parsed_args["print_progress"] == true)
			println(size(SparseRelationTensorDataframe))
		end
	end
end

SymmetricRelationIndexes = map(y -> findfirst(x -> x == y,AllRelations),SymmetricRelations);
AllSymmetricRelationTriples = filter("index k" => x -> x ∈ SymmetricRelationIndexes,SparseRelationTensorDataframe);

CSV.write("C:/Users/manue/Downloads/ConvAI Assignment1/Projects/ccm/data/RelationTensor.csv", df)

if(parsed_args["print_progress"] == true)
	print(IsCurrentEntrySymmetric);
	print(AllRelations);
end

IsCurrentEntrySymmetric = ones(size(AllRelations,1));

space_out = "";
i_val = 0;

for hrt_string in json_relations_set_augmented
	hrt_array = (head,relation,tail) = String.(split(hrt_string,delimiter_in));
	trh_string = join(reverse(hrt_array),delimiter_in);
	if (trh_string ∉ json_relations_set)
		IsCurrentEntrySymmetric[findall(x -> x == relation, AllRelations)] .= 0
	else
		IsCurrentEntryPurelyUnsymmetric[findall(x -> x == relation, AllRelations)] .= 0
	end
	tab_separated_string = space_out * join(hrt_array,delimiter_out);
	if(i_val == 0)
		global i_val = 1;
		global space_out = "\n";
	end
	write(output_write_file, tab_separated_string);
end

if(parsed_args["print_progress"] == true)
	print(IsCurrentEntrySymmetric);
end

close(output_write_file);