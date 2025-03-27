<div align="center">
    <p> 
    	<b>
        Spatio-temporal Urban Knowledge Graph for heterogeneous urban data
        </b>
    </p>

------

<p align="center">
  <a href="## Overview">Overview</a> •
  <a href="## Installation">Installation</a> •
  <a href="## Dataset">Dataset</a> •
  <a href="## How to Run">How to Run </a> •
  <a href="## Directory Structure">Directory Structure</a> •
  <a href="## Citation">Citation</a> 
</p>
</div>

Our model basically forked this paper: ["UUKG: The Unified Urban Knowledge Graph Dataset for Urban Spatiotemporal Prediction"](https://arxiv.org/pdf/2306.11443.pdf).
Based on this model, we have the below achievement:
1. 
2.


## 1. Overview

[//]: # (<div style="display: flex; justify-content: center;">)

[//]: # (  <img src="https://github.com/usail-hkust/UUKG/blob/main/workflow.png" width="400">)

[//]: # (</div>)

**First proposal:**

In recent years, Evidence-Based Policy Making (EBPM) has achieved significant success in urban planning, decision making, and mobility management, benefiting from cutting-edge AI techniques and large-scale datasets. Prior research has extensively focused on improving performance by extracting knowledge from diverse urban data sources. However, much of this work relies on manual feature extraction, which constrains the broad usage of data and underrepresents the latent knowledge available from multiple data sources.

The recent success of Knowledge Graphs (KG) in natural language processing tasks has paved the way for Urban Knowledge Graphs (UrbanKGs) to support various urban tasks. However, existing UrbanKGs are often task-specific and primarily capture static urban information, such as Points of Interest (POI), administrative divisions, and population data.

Our lab has access to large-scale collective human mobility data, including Call Detail Records (CDR) and GPS data, enabling us to investigate the spatiotemporal relationships within urban areas and enrich UrbanKGs with higher-dimensional insights. This study aims to construct a Spatiotemporal Urban Knowledge Graph (ST-UrbanKG) using deep learning techniques and multiple spatiotemporal datasets to solve a variety of urban mobility tasks. Several ongoing tasks in our lab, such as Smart Mobility, behavior change under urban policy, and mobility prediction/generation, will benefit from this study.


## 2. Installation
Step 1: Create a python 3.10 environment and install dependencies:

[//]: # (```)

[//]: # (conda create -n python3.7 UUKG)

[//]: # (source activate UUKG)

[//]: # (```)

Step 2: Install library

[//]: # (```bash)

[//]: # (pip install -r ./UrbanKG_Embedding_Model/requirements.txt)

[//]: # (pip install -r ./USTP_Model/requirements.txt)

[//]: # (```)



## 3. Dataset
We use the following dataset to construct UrbanKG:

- Administrative data(open source)
- POI data(non-open source)
- Softbank data(non-open source)

##### Guidance on data usage and processing

We store the original unprocessed files in the **'./Meta_data'** directory. To preprocess, align, and filter these files, we utilize either the **`preprocess_meta_data_nyc.py`** or **`preprocess_meta_data_chi.py`** script. The processed data is then saved in the **'./Processed_data'** directory. Finally, we execute the **`construct_UrbanKG_NYC.py`** or **`construct_UrbanKG_CHI.py`** script to obtain the constructed urban knowledge graphs, which are stored in the **'./UrbanKG'** directory.

We divide the training set, verification set and test set by **`train_val_test_ent2id_rel2id.py`**

The file information in each directory is as follows:

```
./Meta_data    Raw data set: administrative division data, POI and road network data
./Processed_data   Aligned datasets: administrative region entity, POI entity, road network entity
./UrbanKG    City fact triples obtained from entities and relations
```

The following types of atomic files (Statics) are defined:

| filename            | content                                 | example              |
| ------------------- | --------------------------------------- | -------------------- |
| entity2id_XXX.txt   | entity_name, entity_id                  | Road/106710 11       |
| relation2id_XXX.txt | relation_name, relation_id              | BNB 2                |
| train               | entity_id, relation_id, entity_id       | 196632  12 85987     |
| valid               | entity_id, relation_id, entity_id       | 43982   10 233474    |
| test                | entity_id, relation_id, entity_id       | 167134  6  75149     |
| triplet.txt         | entity_id, relation_id, entity_id       | 48034   12 168303    |
| UrbanKG_XXX.txt     | entity_name, relation_name, entity_name | POI/663 PLA Area/230 |


## 4. How to Run
#### UrbanKG_Data
`data_precocessing.py` ->
`construct_ST-UrbanKG.py` ->
`data_split_en2id_rel2id.py` ->

#### UrbanKG_Embedding
`process.py` ->
`run.py`

## 5 Directory Structure

```
UrbanKG
 |-- UrbanKG_Data
 |    |-- Processed_data
 |    |    |-- Shizuoka
 |    |    |    |-- Administrative_data    
 |    |    |    |-- Mobile_OD_data     
 |    |    |    |-- POI_data  
 |    |-- Raw_data 
 |    |    |-- Shizuoka
 |    |    |    |-- Administrative_data    
 |    |    |    |-- Mobile_OD_data     
 |    |    |    |-- POI_data  
 |    |-- UrbanKG
 |    |    |-- Static_UrbanKG
 |    |    |    |-- entity2id.txt   
 |    |    |    |-- relation2id.txt     
 |    |    |    |-- UrbanKG_Shizuoka.txt
 |    |    |    |-- triplets.txt   
 |    |    |    |-- train.txt  
 |    |    |    |-- valid.txt
 |    |    |    |-- test.txt
 |    |    |-- Temporal_UrbanKG
 |    |    |    |-- T-entity2id.txt   
 |    |    |    |-- T-relation2id.txt     
 |    |    |    |-- T-UrbanKG_Shizuoka.txt
 |    |    |    |-- quadruplets.txt   
 |    |    |    |-- T-train.txt  
 |    |    |    |-- T-valid.txt
 |    |    |    |-- T-test.txt
 |    |-- utils
 |    |    |-- data_visualization.py
 |    |    |-- processing_functions.py
 |    |-- construct_ST-UrbanKG_Shizuoka.py
 |    |-- data_preprocessing_Shizuoka.py
 |    |-- data_split_en2id_rel2id.py

 |-- UrbanKG_Embedding
 |    |-- Data
 |    |    |-- Static
 |    |    |-- Temporal 
 |    |-- logs
 |    |-- models
 |    |    |-- base.py
 |    |    |-- static.py
 |    |    |-- complex.py
 |    |    |-- temporal.py
 |    |-- optimizers
 |    |-- utils
 |    |-- kg_dataset_class.py
 |    |-- process.py
 |    |-- run.py
```


## 6 Citation
