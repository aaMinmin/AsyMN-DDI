# Overview
An Asymmetric Modulation Network for Drug-Drug Interaction Prediction Targeting Cross-Modal Redundancy

## 📂 Repository Structure
```text
├── data/                           # Directory for datasets (prepare manually as per instructions below)
├── data_processing/                # Mutual information statistics
├── AsyMNfusion.py                  # Relevance-Redundancy Asymmetric Modulation Module
├── dataset_processed.py            # dataset processing.
├── evidential_tf.py                # EDL.
├── features.py                     # Feature generation script
├── kgnn.py                         # Knowledge graph extractor
├── model_motif_level.py            # motif extractor
├── mol_fingerprint_extractor.py    # FP extractor
├── mol_graph.py                    # Script for extracting molecular motifs
├── processing.py                   # to make preprocessed_drug_info.npy
├── train_nounseendrug.py           # Main script for training the model (nounseendrug)
├── train_unseendrug.py             # Main script for training the model (unseendrug:one;two)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## ⚙️ Environment Setup

# Clone the repository
git clone https://github.com/aaMinmin/AsyMN-DDI.git
cd AsyMN-DDI

# Install dependencies
pip install -r requirements.txt

## 📊 Knowledge graph Dataset Preparation

### 📁 Dataset 1: Molecular Motif Knowledge Graph

**1. Construction Pipeline**
*   **Step 1:** Parse the input SMILES strings of drug molecules using the provided scripts `mol_graph.py` and `features.py` to extract their molecular scaffolds and local functional motifs (substructures).
*   **Step 2:** Associate the drug entities with their corresponding motif entities to formulate the triples.

**2. Triple Format**
*   **Head:** SMILES string of the drug molecule.
*   **Relation:** `included` (indicates the drug contains the specific motif).
*   **Tail:** The extracted motif/substructure fragment.

**3. Data Examples**
```text
O=C1N(C2CCC(=O)NC2=O)C(=O)C2=CC=CC=C12 included CN
O=C1N(C2CCC(=O)NC2=O)C(=O)C2=CC=CC=C12 included C=O
O=C1N(C2CCC(=O)NC2=O)C(=O)C2=CC=CC=C12 included C=O
O=C1N(C2CCC(=O)NC2=O)C(=O)C2=CC=CC=C12 included C=O
O=C1N(C2CCC(=O)NC2=O)C(=O)C2=CC=CC=C12 included C1CCCCN1
O=C1N(C2CCC(=O)NC2=O)C(=O)C2=CC=CC=C12 included C1ccCN1
O=C1N(C2CCC(=O)NC2=O)C(=O)C2=CC=CC=C12 included c1ccccc1
```
*(Note: If a drug contains multiple identical motifs, such as the carbonyl group `C=O`, the frequency information is retained.)*

---

### 📁 Dataset 2: Biomedical Pharmacological Knowledge Graph

**1. Data Source**
*   **Source:** The complete JSON database downloaded from [DrugBank](https://go.drugbank.com).
*   **Version Used:** DrugBank v5.1.8 (File size: approximately 1.86 GB).

**2. Construction Pipeline**
*   **Step 1:** Load the 1.86 GB raw JSON file.
*   **Step 2:** Recursively traverse the JSON structure to locate and extract information from the following five core fields: `targets`, `enzymes`, `carriers`, `transporters`, `pathways`.
*   **Step 3:** For each sub-entry within these fields, extract the `name`, `id`, `general-function`, and `action` (mechanism of action, e.g., inhibitor, inducer).
*   **Step 4:** Link the drug entities with the extracted biological entities (e.g., diseases, targets, enzymes) to construct the triples.

**3. Triple Format**
*   **Head:** SMILES string of the drug molecule.
*   **Relation:** The specific action/mechanism (e.g., `inhibitor`, `disease`) or the associated biological field.
*   **Tail:** The ID of the target biological entity (e.g., SMPDB ID, BioEntity ID).

**4. Data Examples**
```text
CC(O)=O disease SMP0000559
CCCC(CCC)C(O)=O inhibitor BE0000215
N[C@@H](CC1=CNC=N1)C(O)=O inhibitor BE0000086
```

### 🛠️ Reproduction Guide for Datasets

1.  **Prepare Raw Data:** Download the DrugBank v5.1.8 JSON source file and prepare the required drug SMILES lists.
2.  **Generate Dataset 1:** Run `python mol_graph.py` and `python features.py` on the SMILES data to obtain `dataset1_triples.txt`.
3.  **Generate Dataset 2:** Write a simple Python script to traverse the DrugBank JSON according to the pipeline described above, extracting the required fields and formatting them into `dataset2_triples.txt`.
4.  **Data Placement:** Place the generated `.txt` triple files in the `./data/` directory. The Data Loader will automatically read these files and construct the adjacency matrices.

## 🚀 Usage

### 1. Training
To train the AsyMN-DDI model on the prepared datasets:

run `python train_unseendrug.py` for unseendrugs (default: S3, using `--data_setting unseendrug-onedrug` to switch to S2), and run `python train_nounseendrug.py` for nounseendrug (S1).
run


