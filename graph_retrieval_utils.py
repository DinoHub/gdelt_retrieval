import torch
import os
import pandas as pd
from clearml import Task, StorageManager, Dataset
import json, pickle


def get_transe(path:str,use_local=True):
    print("Loading TransE Matrix")
    if use_local:
        transe_path = os.path.join("data",path)
    else:
        #transe_path = os.path.join(Dataset.get(dataset_project="shasha/IE-Demo",dataset_name=path).get_use_local_copy(),path)
        transe_path = StorageManager.get_local_copy("s3://experiment-logging/storage/gdelt-embeddings/openke-graph-training.3ed8b6262cd34d52b092039d0ee1d374/artifacts/transe.ckpt/transe.ckpt")
    transe_matrix = torch.load(transe_path)
    return transe_matrix

def get_cluster(path:str,use_local=True):
    if use_local:
        cluster_path = os.path.join("data",path)
    else:
        #cluster_path = os.path.join(Dataset.get(dataset_project="datasets/gdelt",dataset_name=path).get_local_copy(),path)
        cluster_path = os.path.join("data",path)
    with open(cluster_path,'r') as f:
        cluster_dict = json.load(f)
    return cluster_dict

def get_doc(path:str,use_local=True):
    if use_local:
        doc_path = os.path.join("data",path)
    else:
        doc_path = StorageManager.get_local_copy("s3://experiment-logging/storage/gdelt-embeddings/graph-clustering-2020.43890e3a8a484997bd0e26bfd9568595/artifacts/temporal_list_by_idx/temporal_list_by_idx.pkl")    
    with open(doc_path,'rb') as f:
        doc_emb = pickle.load(f)
    output = {k: v for d in doc_emb for k, v in d.items()}
    return output

def get_entity_id(entity_text_list:list, path:str, use_local=True)->list:
    if use_local:
        ent2id_path = os.path.join("data",path)
    else:
        ent2id_path = os.path.join(Dataset.get(dataset_project="datasets/gdelt",dataset_name="gdelt_openke_format_w_extras").get_local_copy(),path)
    ent2id = pd.read_csv(ent2id_path, names=["id", "value"])[1:]
    ent_id = ent2id[ent2id['value'].isin(entity_text_list)]['id'].values
    return ent_id

def get_relation_id(relation_text_list:list, path:str, use_local=True):
    if use_local:
        evt2id_path = os.path.join("data",path)
    else:
        evt2id_path = os.path.join(Dataset.get(dataset_project="datasets/gdelt",dataset_name="gdelt_openke_format_w_extras").get_local_copy(),path)
    evt2id = pd.read_csv(evt2id_path, names=["id", "value"])[1:]
    evt_id = evt2id[evt2id['value'].isin(relation_text_list)]['id'].values
    return evt_id

def get_entity_embedding(id_list:list, embedding_matrix):
    output_list = []
    print(embedding_matrix.shape)
    for i in id_list:
        print(i)
        output_list.append(embedding_matrix[int(i)])
    if output_list==[]:
        # return torch.zeros(1,200).cuda()
        return None
    else:
        output_tensor = torch.stack(output_list)
        return output_tensor

def get_relation_embedding(id_list:list, embedding_matrix):
    output_list = []
    for i in id_list:
        output_list.append(embedding_matrix[int(i)])
    if output_list==[]:
        # return torch.zeros(1,200).cuda()
        return None
    else:
        output_tensor = torch.stack(output_list)
        return output_tensor

def get_doc_embedding(use_local, document:list):
    document_query=document
    print(document_query)
    transe=get_transe("transe.ckpt",use_local)
    src=[triple[0] for triple in document_query if triple[0]!=[]]
    rel=[triple[1] for triple in document_query if triple[1]!=[]]
    tgt=[triple[2] for triple in document_query if triple[2]!=[]]

    src_id = get_entity_id(src,"entity2id.txt",use_local)
    rel_id = get_relation_id(rel,"relation2id.txt",use_local)
    tgt_id = get_entity_id(tgt,"entity2id.txt",use_local)

    src_embedding = get_entity_embedding(src_id, transe["ent_embeddings.weight"])
    rel_embedding = get_relation_embedding(rel_id, transe["rel_embeddings.weight"])
    tgt_embedding = get_entity_embedding(tgt_id, transe["ent_embeddings.weight"])

    emb_list = []
    type_list = []
    for i,emb in enumerate([src_embedding,rel_embedding,tgt_embedding]):
        if emb==None:
            continue
        else:
            emb_list.append(emb.mean(dim=0).unsqueeze(0))
            type_list.append(i)

    # doc_embedding = torch.cat((src_embedding.unsqueeze(0), rel_embedding.unsqueeze(0), tgt_embedding.unsqueeze(0)), dim=1)
    doc_embedding = torch.cat(emb_list, dim=1)
    return doc_embedding, type_list

### Function that matches a query embedding to a document vector database ###
def graph_doc_matching(query_embedding:torch.tensor,type_list:list, use_local=True, use_cluster=True
):    
    print("Matching Embedding to DB...")
    full_doc_emb = get_doc("temporal_list_by_idx.pkl",use_local)
    cluster_dict = get_cluster("cluster_data.json",use_local)
    max_cluster_score = 0
    cluster_number = ""
    
    score_list = []
    id_list = []
    
    if use_cluster:
        for cluster in cluster_dict.keys():
            if cluster!=str(-1):
                cluster_centroid = torch.Tensor(cluster_dict[cluster]['centroid'])
                centroid_embedding = torch.split(cluster_centroid,200,0)
                emb_list = []
                for index in type_list:
                    emb_list.append(centroid_embedding[index].unsqueeze(0))
                centroid_embedding = torch.cat(emb_list,dim=1)
                sim_score = torch.cosine_similarity(query_embedding.cuda().view(1,-1),centroid_embedding.cuda().view(1,-1)).item()
                if sim_score>=max_cluster_score:
                    cluster_number = str(cluster)
                    max_cluster_score = sim_score
            else:
                continue    

        for doc_id in cluster_dict[cluster_number]['id_list']:
            doc_embedding = full_doc_emb[doc_id]
            doc_embedding = torch.split(doc_embedding,200,0)
            emb_list = []
            for index in type_list:
                emb_list.append(doc_embedding[index].unsqueeze(0))
            doc_embedding = torch.cat(emb_list,dim=1)
            print(doc_embedding.shape)
            print(query_embedding.cuda().view(1,-1).shape)
            sim_score = torch.cosine_similarity(query_embedding.cuda().view(1,-1),doc_embedding.cuda().view(1,-1)).item()
            score_list.append(sim_score)
            id_list.append(doc_id)    
    else:
        for key in full_doc_emb.keys():
            doc_embedding = full_doc_emb[key]
            doc_embedding = torch.split(doc_embedding,200,0)
            emb_list = []
            for index in type_list:
                emb_list.append(doc_embedding[index].unsqueeze(0))
            doc_embedding = torch.cat(emb_list,dim=1)
            sim_score = torch.cosine_similarity(query_embedding.cuda().view(1,-1),doc_embedding.cuda().view(1,-1)).item()
            score_list.append(sim_score)
            id_list.append(key)
    
    print(score_list)
    cluster_df = pd.DataFrame()
    cluster_df['id'] = id_list
    cluster_df['score'] = score_list
    cluster_df = cluster_df.sort_values(['score'],ascending=False).head(5)
    return cluster_df

### Function that converts text questions to triples ###
def process_text(input_text:str, question_templates)->tuple:
    if input_text in question_templates.keys():
        return question_templates[input_text]
    else:
        return [], [], []

