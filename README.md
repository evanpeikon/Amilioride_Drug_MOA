# 🧬 Introduction
## **Background and Overview**
Multiple myeloma (MM) is a hematologic malignancy characterized by the uncontrolled proliferation of plasma cells in the bone marrow. This disease is associated with a variety of complications, including bone destruction, anemia, kidney dysfunction, and immune suppression. Despite advancements in treatment options, such as proteasome inhibitors, immunomodulatory drugs, and monoclonal antibodies, multiple myeloma remains largely incurable, and resistance to therapy often develops over time. As a result, there is a continued need for novel therapeutic approaches to improve patient outcomes.

Amiloride is a potassium-sparing diuretic commonly used to treat conditions like hypertension and heart failure. It works by inhibiting the epithelial sodium channel (ENaC) in the kidneys, preventing sodium reabsorption and promoting potassium retention. Interestingly, multiple studies [LINK] have suggested that amiloride may have potential as a cancer therapeutic, including in multiple myeloma. It is believed that amiloride exerts its antitumor effects by targeting multiple cellular pathways involved in tumor progression, including ion channel regulation, apoptosis, and cell migration. Additionally, amiloride has shown synergistic effects when combined with other chemotherapeutic agents, such as dexamethasone and melphalan, in preclinical models.

Given amiloride’s promising effects in preclinical studies and its established safety profile, there is growing interest in investigating its potential as a therapeutic agent for multiple myeloma. Thus, the purpose of this project is to explore Amiliorides mechanisms of action at the molecular level to determine how it can be effectively integrated into existing treatment regimens.

By leveraging various bioinformatics approaches, including weighted gene co-expression network analysis (WGCNA), differential expression analysis, protein-protein interaction (PPI) network analysis, and functional enrichment analysis, this project will provide a comprehensive view of how amiloride impacts cellular processes at different levels of biological organization. The results of this project will enhance our understanding of amiloride’s therapeutic potential and guide future research on its use in the treatment of multiple myeloma.

This repository is structured into four main sections:
- Introduction: Provides an overview of the project, scientific background, the data sources used in this project, and the analyses employed. 
- Project Staging: Details project dependencies, data loading and preperation, exploratory data analysis (EDA), quality control, filtering, and normalization steps.
- Analyses: Includes weighted gene co-expression network analysis, differential expression analysis, protein-protein interaction network analysis, and functional enrichment analysis. Summarizations of the results for each analysis are also provided here.
- Results and Discussion: Synthesizes the results from all analyses and discusses their broader biological context and implications.

## **Data Sources and Analyses**
The data for this project is derived from the study titled "Amiloride, An Old Diuretic Drug, Is a Potential Therapeutic Agent for Multiple Myeloma," which is publicly available through the Gene Expression Omnibus (GEO) under accession number [GSE95077](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE95077). This study explores the effects of amiloride on multiple myeloma cell lines and a xenograft mouse model, providing valuable insights into the drug's potential as a therapeutic agent.

The bulk RNA-sequencing data for the JJNE cell lines (established from the bone marrow of a 57-year-old woman with plasma cell leukemia) used in this project are accessible through the following sample accession numbers: [GSM2495770](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2495770) , [GSM2495771](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2495771) , [GSM2495772](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2495772) , [GSM2495767](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2495767) , [GSM2495768](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2495768) , [GSM2495769](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2495769).

In this study, amiloride’s impact was assessed in multiple myeloma cell lines, alongside a xenograft mouse model, where human tumor cells were implanted into immunocompromised mice. The research aimed to evaluate the drug’s toxicity, specifically its ability to induce cell death (apoptosis) in tumor cells. Additionally, the study sought to understand the underlying mechanisms through a combination of RNA sequencing, quantitative PCR (qRT-PCR), immunoblotting, and immunofluorescence assays.

The investigators observed that amiloride treatment led to apoptosis in a broad panel of multiple myeloma cell lines and in the xenograft mouse model. Furthermore, they discovered that amiloride exhibited a synergistic effect when combined with other chemotherapeutic agents, including dexamethasone and melphalan. These findings suggest that amiloride could enhance the efficacy of existing treatments, potentially improving therapeutic outcomes for multiple myeloma patients.

For this project, I will analyze the RNA-sequencing data from this study to gain deeper insights into amiloride’s mechanisms of action. Specifically, the following analyses will be performed:

- [Weighted Gene Co-Expression Network Analysis](https://github.com/evanpeikon/co_expression_network): WGCNA will be used to identify coordinated patterns of gene expression across both the treatment and control groups. This analysis will help uncover underlying biological processes and regulatory networks that control gene expression. WGCNA is particularly useful for identifying upstream regulatory factors that may influence the expression of multiple genes, and it provides a global view of transcriptional organization, revealing how genes with similar expression profiles are grouped together, often reflecting shared functions or regulatory mechanisms.

- Differential Expression Analysis: This analysis will focus on identifying genes that show significant changes in expression between the treatment (amiloride) and control groups. Differential expression analysis will highlight genes that are directly affected by the drug, identifying those that are upregulated or downregulated as a result of amiloride treatment. By isolating these changes, we can pinpoint the most biologically relevant genes that may play key roles in amiloride’s effects on multiple myeloma cells.

- [Protein-Protein Interaction (PPI) Network Analysis](https://github.com/evanpeikon/PPI_Network_Analysis): Building on the results of the differential expression analysis, PPI network analysis will shift the focus to the interactions between the proteins encoded by the differentially expressed genes. This analysis will help us understand how amiloride’s effects may be mediated at the protein level, revealing how these proteins interact with each other to form complexes that drive cellular processes. By mapping out these interactions, we can connect the gene expression changes to functional outcomes and identify critical protein networks that are altered by amiloride.

- [Functional Enrichment Analysis](https://github.com/evanpeikon/functional_enrichment_analysis): To provide a broader biological context for the differentially expressed genes and proteins, functional enrichment analysis will categorize them into known biological functions and pathways. Gene Ontology (GO) analysis will assign genes to specific functional terms, such as biological processes, molecular functions, and cellular components. Additionally, pathway analysis (e.g., KEGG or Reactome) will map genes to known signaling and metabolic pathways. This step will abstract the results from the gene and protein levels to a higher-level understanding of the biological processes and molecular pathways that are impacted by amiloride treatment, linking gene expression changes and protein interactions to functional consequences within the cell.

By performing these analyses, we will examine amiloride’s effects at multiple levels of biological organization—from gene expression changes to protein interactions and functional pathway alterations—allowing us to gain a comprehensive understanding of its mechanisms of action in multiple myeloma.

# **🧬 Project Staging**

This section is broken into three sub-sections:

- **Project Dependencies:** Import the necessary libraries for downstream analyses.
- **Load, Inspect, and Prepare Data:** Steps to load raw datasets, inspect data structure, and prepare it for downstream processing.
- **Quality Control, Filtering, and Normalization:** Applying quality control measures and data transformations to generate high-quality, reliable data for down stream analysis.

## **Project Dependencies**
To start, we'll import the following Python libraries, which will be used in various steps of the analysis. Ensure that these libraries are installed in your Python environment before proceeding.

```python
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import mygene
import gseapy as gp
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from scipy.integrate import solve_ivp
import os 
import subprocess
import gzip
import requests 
import networkx as nx 
from io import StringIO
```

## **Load, Inspect, and Prepare Data**
Next, we'll retrieve our RNA-sequencing data for the control and treatment JJN3 cells lines:

```python
def download_and_load_data(url, output_filename, sep="\t", column_filter=None):
    # Step 1: Download the file using wget
    print(f"Downloading {output_filename} from {url}...")
    subprocess.run(["wget", "-O", output_filename + ".gz", url], check=True)

    # Step 2: Gunzip the file
    print(f"Unzipping {output_filename}.gz...")
    with gzip.open(output_filename + ".gz", "rb") as gz_file:
        with open(output_filename, "wb") as out_file:
            out_file.write(gz_file.read())

    # Step 3: Load the data into a pandas DataFrame
    print(f"Loading {output_filename} into a pandas DataFrame...")
    df = pd.read_csv(output_filename, sep=sep, index_col=0)

    # Optional: Filter columns based on the keyword
    if column_filter:
        print(f"Filtering columns with keyword '{column_filter}'...")
        filtered_columns = [col for col in df.columns if column_filter in col]
        df = df[filtered_columns]

    return df

# Load data 
url = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE95077&format=file&file=GSE95077%5FNormalized%5FCount%5FMatrix%5FJJN3%5FAmiloride%5Fand%5FCTRL%2Etxt%2Egz'
output_filename = "GSE95077_Normalized_Count_Matrix_JJN3_Amiloride_and_CTRL.txt"
column_keyword = "" #no keyword needed
countlist = download_and_load_data(url, output_filename, column_filter=column_keyword)

# View first 5 rows of data
countlist.head()
```
Which produces the following output:

<img width="1350" alt="Screenshot 2025-01-31 at 8 10 40 AM" src="https://github.com/user-attachments/assets/b799eb3d-f2e1-41c7-ae24-543d6e28cd1a" />

The data frame above is indexed by Ensemble gene ID (ENSG), and then there are six columns of RNA-sequencing expression data (i.e., counts). The first three columns contain expression counts for the experimental group (i.e., the amiloride treatment group), and the last three columns contain expression counts for the control group. 

Before performing downstream analyses we'll want to convert the Ensemble gene IDs to gene names, as demonstrated in next code block.

```python
# create MyGeneInfo object
mg = mygene.MyGeneInfo()

# get the ensembl id from index
ensembl_ids = countlist.index.tolist()

# query the gene symbols for the ensemble ids and onvert result to dataframe
gene_info = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='human')
gene_df = pd.DataFrame(gene_info)

# remove duplicate ensemble ids and rows where symbol is missing or duplicated
gene_df = gene_df.dropna(subset=['symbol']).drop_duplicates(subset='query')

# map gene symbols back to original dataframe and move gene_name column to front column
countlist['Gene_Name'] = countlist.index.map(gene_df.set_index('query')['symbol'])
cols = ['Gene_Name'] + [col for col in countlist.columns if col != 'Gene_Name']
countlist = countlist[cols]

countlist.head()
```
Which produces the following output:

<img width="1414" alt="Screenshot 2025-01-31 at 8 11 27 AM" src="https://github.com/user-attachments/assets/fa972904-d756-4138-b08d-786f6ab5e784" />

Next, we'll check for missing data and perform basic data exploration to understand the distribution and variability of RNA sequencing counts across the samples before performing any downstream analysis. First, let's check for missing value:

```python
# check for missing values
print(countlist.isnull().sum())
```
<img width="261" alt="Screenshot 2025-01-31 at 8 12 15 AM" src="https://github.com/user-attachments/assets/2184a678-d01f-4f66-8a8a-147619291833" />

Notably, the dataset has no null (missing) values. Next, we'll explore the distribution and variability in our dataset, as demonstrated in the code block below:

```python
# Drop the Gene Name column from countlist for counting
countlist_no_name = countlist.iloc[:, 1:]

# Calculate total counts per sample and log transform counts
total_counts = countlist_no_name.sum(axis=0)
log_counts = countlist_no_name.apply(lambda x: np.log2(x + 1))

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Plot total counts per sample
axes[0].bar(countlist_no_name.columns, total_counts, color='skyblue')  # Use countlist_no_name columns here
axes[0].set_ylabel('Total Counts')
axes[0].set_title('Total Counts per Sample')
axes[0].tick_params(axis='x', rotation=85)

# Plot log transformed counts per sample
log_counts.boxplot(ax=axes[1])
axes[1].set_ylabel('Log2(Counts + 1)')
axes[1].set_title('Log Transformed Counts per Sample')
axes[1].tick_params(axis='x', rotation=85)

plt.tight_layout()
plt.show()
```
Which produces the following output:

<img width="1352" alt="Screenshot 2025-01-31 at 8 12 58 AM" src="https://github.com/user-attachments/assets/40609334-93e2-45e3-82a4-06a9d997f6b5" />

The chart on the left, titled "Total Counts Per Sample," helps visualize the overall sequencing depth across the samples. Ideally, the bars, representing the total counts, should be of similar height, indicating that sequencing depth is consistent across samples, which is the case with this data set.

Conversely, the rightmost chart is used to assess the variability and distribution of gene expression counts across samples. In this case, we can see that the boxes, representing the interquartile range, and the whiskers, representing variability outside the upper and lower quartiles, are of similar sizes across the samples, indicating a consistent data distribution.

Now, before moving on to quality control and filtering, we'll use one last visualization to explore the similarity, or dissimilarity, between our six samples:

```python
# perform hierarchical clustering and create dendrogram
h_clustering = linkage(log_counts.T, 'ward')
plt.figure(figsize=(8, 6))
dendrogram(h_clustering, labels=countlist_no_name.columns)
plt.xticks(rotation=90)
plt.ylabel('Distance')
plt.show()
```

<img width="527" alt="Screenshot 2025-01-31 at 8 13 31 AM" src="https://github.com/user-attachments/assets/7e89400d-f823-4e47-a216-a00c2c001a48" />

The image above shows the results of hierarchical clustering, which can be visualized via a dendrogram. When viewing a dendrogram, special attention should be paid to the cluster groupings and branches. Samples clustered together are more similar to each other, and the length of the branches (vertical lines) connecting clusters represents the distance or dissimilarity between clusters.

The chart above shows that our three control samples are clustered on the right, whereas our three experimental (i.e.,m amiloride-exposed) samples are clustered together on the left. This is a good sign, suggesting that the control and experimental groups are distinct and that there is biological variation between the two groups of samples. Thus, we can feel confident that our downstream differential expression analyses will provide meaningful results.
-----




<img src="images/study_fig2a.png" alt="Description" width="700" height="300">
