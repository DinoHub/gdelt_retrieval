import streamlit as st
import torch
import os
import pandas as pd
from clearml import Task, StorageManager, Dataset
import json, pickle

st.set_page_config(layout="wide")
st.title("Graph Retrieval Demo")

if "src" not in st.session_state.keys():
    st.session_state.src = []
if "rel" not in st.session_state.keys():
    st.session_state.rel = []
if "tgt" not in st.session_state.keys():
    st.session_state.tgt = []
if "document" not in st.session_state.keys():
    st.session_state.document = []


def set_source():
    #st.session_state.src.append(st.session_state.source)
    st.session_state.src = st.session_state.source

def set_target():
    #st.session_state.tgt.append(st.session_state.target)    
    st.session_state.tgt = st.session_state.target

def set_relation():
    #st.session_state.rel.append(st.session_state.relation)    
    st.session_state.rel = st.session_state.relation

#to do
def get_transe(path:str,local=True):
    if local:
        transe_path = os.path.join("data",path)
    else:
        transe_path = os.path.join(Dataset.get(dataset_project="shasha/IE-Demo",dataset_name=path).get_local_copy(),path)
    er_emb = torch.load(transe_path)
    return er_emb

#to do
def get_cluster(path:str,local=True):
    if local:
        cluster_path = os.path.join("data",path)
    else:
        cluster_path = os.path.join(Dataset.get(dataset_project="shasha/IE-Demo",dataset_name=path).get_local_copy(),path)
    with open(cluster_path,'r') as f:
        cluster_dict = json.load(f)
    return cluster_dict

#to do
def get_doc(path:str,local=True):
    if local:
        doc_path = os.path.join("data",path)
    else:
        doc_path = os.path.join(Dataset.get(dataset_project="shasha/IE-Demo",dataset_name=path).get_local_copy(),path)
    with open(doc_path,'rb') as f:
        doc_emb = pickle.load(f)
    output = {k: v for d in doc_emb for k, v in d.items()}
    return output

#to do
def get_entity_id(entity_text_list,path,local=True):
    if local:
        ent2id_path = os.path.join("data",path)
    else:
        ent2id_path = os.path.join(Dataset.get(dataset_project="shasha/IE-Demo",dataset_name=path).get_local_copy(),path)
    ent2id = pd.read_csv(ent2id_path)
    ent_id = ent2id[ent2id['value'].isin(entity_text_list)]['id'].values
    return ent_id

#to do
def get_relation_id(relation_text_list,path,local=True):
    if local:
        evt2id_path = os.path.join("data",path)
    else:
        evt2id_path = os.path.join(Dataset.get(dataset_project="shasha/IE-Demo",dataset_name=path).get_local_copy(),path)
    evt2id = pd.read_csv(evt2id_path)
    evt_id = evt2id[evt2id['value'].isin(relation_text_list)]['id'].values
    return evt_id

#to do
def get_entity_embedding(id_list, embedding_matrix):
    output_list = []
    for i in id_list:
        output_list.append(embedding_matrix[i])
    if output_list==[]:
        return torch.zeros(1,200).cuda()
    else:
        output_tensor = torch.stack(output_list)
        return output_tensor

#to do
def get_relation_embedding(id_list, embedding_matrix):
    output_list = []
    for i in id_list:
        output_list.append(embedding_matrix[i])
    if output_list==[]:
        return torch.zeros(1,200).cuda()
    else:
        output_tensor = torch.stack(output_list)
        return output_tensor

#to do
def get_doc_embedding(local):
    document_query=st.session_state.document
    print(document_query)
    transe=get_transe("transe.ckpt",local=local)
    src=[triple[0] for triple in document_query if triple[0]!=[]]
    rel=[triple[1] for triple in document_query if triple[1]!=[]]
    tgt=[triple[2] for triple in document_query if triple[2]!=[]]

    src_id = get_entity_id(src,"ent2id.txt",local=local)
    rel_id = get_relation_id(rel,"evt2id.txt",local=local)
    tgt_id = get_entity_id(tgt,"ent2id.txt",local=local)

    src_embedding = get_entity_embedding(src_id, transe["ent_embeddings.weight"]).mean(dim=0)
    rel_embedding = get_relation_embedding(rel_id, transe["rel_embeddings.weight"]).mean(dim=0)
    tgt_embedding = get_entity_embedding(tgt_id, transe["ent_embeddings.weight"]).mean(dim=0)

    doc_embedding = torch.cat((src_embedding.unsqueeze(0), rel_embedding.unsqueeze(0), tgt_embedding.unsqueeze(0)), dim=1)
    return doc_embedding

#to do
def graph_doc_matching(query_embedding,local):
    
    cluster_dict = get_cluster("cluster_data.json",local)
    full_doc_emb = get_doc("temporal_list_by_idx.pkl",local)
    max_cluster_score = 0
    cluster_number = ""
    
    score_list = []
    id_list = []
    
    use_cluster=True
    if use_cluster:
        for cluster in cluster_dict.keys():
            if cluster!=str(-1):
                cluster_centroid = torch.Tensor(cluster_dict[cluster]['centroid'])
                sim_score = torch.cosine_similarity(query_embedding.cuda().view(1,-1),cluster_centroid.cuda().view(1,-1)).item()
                if sim_score>=max_cluster_score:
                    cluster_number = str(cluster)
                    max_cluster_score = sim_score
            else:
                continue        
    else:
        for key in full_doc_emb.keys():
            sim_score = torch.cosine_similarity(query_embedding.cuda().view(1,-1),full_doc_emb[key].cuda().view(1,-1)).item()
            score_list.append(sim_score)
            id_list.append(key)

    for doc_id in cluster_dict[cluster_number]['id_list']:
        sim_score = torch.cosine_similarity(query_embedding.cuda().view(1,-1),full_doc_emb[doc_id].cuda().view(1,-1)).item()
        score_list.append(sim_score)
        id_list.append(doc_id)
    
    cluster_df = pd.DataFrame()
    cluster_df['id'] = id_list
    cluster_df['score'] = score_list
    cluster_df = cluster_df.sort_values(['score'],ascending=False).head(5)
    return cluster_df

def set_query():
    st.subheader("Add Query")
    src = st.text_input("insert source entity here", key="source", value="", on_change=set_source)
    rel = st.text_input("insert relation here", key="relation", value="", on_change=set_relation)
    tgt = st.text_input("insert target entity here", key="target", value="", on_change=set_target)
    if st.button("submit"):
        st.session_state.document.append([st.session_state.src, st.session_state.rel, st.session_state.tgt])
    # if st.button("calculate"):
    #     query_embedding = get_doc_embedding()
    #     output_table = graph_doc_matching(query_embedding)
    #     st.table(output_table)

#st.session_state.document=[]
set_query()
st.subheader("Registered Query")
#st.write([st.session_state.src, st.session_state.rel, st.session_state.tgt])
st.write(st.session_state.document)
if st.button("query"):
    query_embedding = get_doc_embedding(local=True)
    output_table = graph_doc_matching(query_embedding,local=True)
    url_ids = output_table["id"]
    dataset_obj = Dataset.get(dataset_project = "datasets/gdelt", dataset_name="raw_gdelt_2021")
    url2id = pd.read_csv(os.path.join(dataset_obj.get_local_copy(), "url2id.txt")).loc[1:,:]
    output_table["id"] = output_table["id"].astype("int32")
    urls = output_table.merge(url2id, how="left", on="id").rename(columns={"value": "top 5 URLs"})#.tolist()
    st.table(urls)

