from utils import *
from dataset_processed import get_adj_matrix,get_dist_matrix,molgraph_rep
import pandas as pd 
import numpy as np
import tqdm 
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

## load data. drug_A is SMILES list of the victim drug , drug_B is the SMILES list of perpetrator drug and DDI is list labels
toy_dataset = pd.read_csv('tr_dataset.csv')
toy_dataset

## obtain all SMILES in the toy dataset
unique_SMILES_in_toy_dataset = []
unique_SMILES_in_toy_dataset.extend(toy_dataset.drug_A)
unique_SMILES_in_toy_dataset.extend(toy_dataset.drug_B)

## obtain all unique SMILES in the toy dataset
unique_SMILES_in_toy_dataset = set(unique_SMILES_in_toy_dataset)

unique_SMILES_in_toy_dataset.__len__()

## create a dict to store all information of drugs
all_drugs_dict = {i:{'adj_matrix':[],'dist_matrix':[],'nums_list':[],'cliques':[],'edges':[],'single_dict':[]} for i in unique_SMILES_in_toy_dataset} 

## load tokenizer
tokenizer = Mol_Tokenizer('token_id.json') 

for drug in tqdm.tqdm(unique_SMILES_in_toy_dataset):
    # try:
    nums_list1, edges1,cliques1= tokenizer.tokenize(drug)
    # dist_matrix1 = get_dist_matrix(mol,cliques1,edges1,single_clique_idx)
    dist_matrix1 = get_dist_matrix(nums_list1,edges1)
    adjoin_matrix1 = get_adj_matrix(nums_list1,edges1)
    all_drugs_dict[drug]['adj_matrix']= adjoin_matrix1 
    all_drugs_dict[drug]['dist_matrix']= dist_matrix1 
    all_drugs_dict[drug]['nums_list']= nums_list1
    all_drugs_dict[drug]['edges']= edges1
    all_drugs_dict[drug]['cliques'] = cliques1
    all_drugs_dict[drug]['single_dict'] = molgraph_rep(drug,cliques1)

np.save('1preprocessed_molecular_info.npy',all_drugs_dict) 