# CCM Torch
A PyTorch derivative of CCM.

The model implemented here closely follows the [CCM](https://github.com/thu-coai/ccm) architecture with mild changes to the preprocessing pipeline and loss functions. The preprocessor is implemented in Julia. The architecture is also slightly tweaked to support complex valued neural networks.

# Required infrastructure:

1. The preprocessor is implemented in Julia Programming Language.
Install using: sudo apt-install julia
2. Install the required packages by running "add_dependencies_julia.jl" in Julia.
3. The model is implemented using PyTorch in Python using base PyTorch. Recommended to use any modern version of Python. For the Complex Valued network version, [complexPyTorch](https://github.com/wavefrontshaping/complexPyTorch) package is needed.

# Running Instructions:

All Data is stored in "/Data" folder.

Step 0 has already been executed and required files are generated in this repository. For running the model, skip to step 1.

Step 0: TSV_Generate.jl and CCM_PreProcessor_Word_Vocabulary.jl do initial pre-processing to generate the word vocabulary and relation triples as tsv format. The generate tsv file can be used with the [DGL-KE library](https://github.com/awslabs/dgl-ke) to generate the embeddings. Convert_checkpoint.py can be used to convert the embeddings to the format compatible with our model, and stored in "Data/Knowledge_Data/Embeddings" folder.

Step 1: Run "CCM_PreProcessor_Data.jl" in Julia to perform pre-processing on the training data stored in "Pre_processed_data/trainset.txt". This will output a post-processed data at "Post_processed_data" with the same name.

Step 2: Run "CCM_Model_Complex.py" in Python to train the model with Complex Valued Neural Networks (or "CCM_Model_Optimized.py" for the one without Complex Value support).

## Paper (Cited)

Hao Zhou, Tom Yang, Minlie Huang, Haizhou Zhao, Jingfang Xu, Xiaoyan Zhu.  
[Commonsense Knowledge Aware Conversation Generation with Graph Attention.](http://coai.cs.tsinghua.edu.cn/hml/media/files/2018_commonsense_ZhouHao_3_TYVQ7Iq.pdf)  
IJCAI-ECAI 2018, Stockholm, Sweden.

**Please kindly cite the original paper if the paper and the original code are helpful.**

## Acknowlegments

We would like to thank the authors of the paper for sharing their work. We also thank our Professor Zhou Yu, John Wright and Richard Zemel and Teaching Assistants Max Chen and Weiyan Shi for their support with helping us understand the model and in handling complex valued neural networks.

## License

Apache License 2.0
