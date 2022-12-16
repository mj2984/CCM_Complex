# CCM Torch
A PyTorch derivative of CCM with support for Complex valued Neural Networks.

The model implemented here closely follows the CCM architecture with mild changes to the preprocessing pipeline and loss functions. The preprocessor is implemented in Julia.

# Required infrastructure:

1. The preprocessor is implemented in Julia Programming Language.
Install using: sudo apt-install julia
2. Install the required packages by running "./add_dependencies_julia.jl" in Julia.
3. The model is implemented using PyTorch in Python. Recommended to use any modern version of Python.

# Running Instructions:

Step 0 has already been executed and required files are generated in this repository. For running the model, skip to step 1.

Step 0: TSV_Generate.jl and CCM_PreProcessor_Word_Vocabulary.jl do initial pre-processing to generate the word vocabulary and relation triples as tsv format. The generate tsv file can be used with the [DGL-KE library](https://github.com/awslabs/dgl-ke) to generate the embeddings. Convert_checkpoint.py can be used to convert the embeddings to the format compatible with our model, and stored in ./Data/Knowledge_Data/Embeddings/ folder.

Step 1: Run "CCM_PreProcessor_Data.jl" in Julia to perform pre-processing on the training data stored in "Data/Pre_processed_data/trainset.txt". This will output a post-processed data at "Data/Post_processed_data/" with the same name.

Step 2: Run "CCM_Model_Optimized.py" in Python to train the model.
