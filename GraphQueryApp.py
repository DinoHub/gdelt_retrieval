import streamlit as st
import torch
import os
import pandas as pd
from clearml import Task, StorageManager, Dataset
import json, pickle
from graph_retrieval_utils import *
from streamlit_agraph import agraph, TripleStore, Node, Edge, Config


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
if "graph" not in st.session_state.keys():
    st.session_state.graph = TripleStore()


def set_source():
    st.session_state.src = st.session_state.source

def set_target():
    st.session_state.tgt = st.session_state.target

def set_relation():
    st.session_state.rel = st.session_state.relation

def set_relation():
    st.session_state.rel = st.session_state.relation


def set_query():
    st.subheader("Add Query")
    src = st.text_input("insert source entity here", key="source", value="", on_change=set_source)
    rel = st.text_input("insert relation here", key="relation", value="", on_change=set_relation)
    tgt = st.text_input("insert target entity here", key="target", value="", on_change=set_target)
    if st.button("submit"):
        if st.session_state.src==[]:
            if st.session_state.tgt==[]:
                if st.session_state.rel==[]:
                    st.session_state.graph.add_triple("Unknown", "Unknown", "Unknown")
                else:
                    st.session_state.graph.add_triple("Unknown", st.session_state.rel, "Unknown")
            else:
                if st.session_state.rel==[]:
                    st.session_state.graph.add_triple("Unknown", "Unknown", st.session_state.tgt)
                else:
                    st.session_state.graph.add_triple("Unknown", st.session_state.rel, st.session_state.tgt)
        else:
            if st.session_state.tgt==[]:
                if st.session_state.rel==[]:
                    st.session_state.graph.add_triple(st.session_state.src, "Unknown", "Unknown")
                else:
                    st.session_state.graph.add_triple(st.session_state.src, st.session_state.rel, "Unknown")
            else:
                if st.session_state.rel==[]:
                    st.session_state.graph.add_triple(st.session_state.src, "Unknown", st.session_state.tgt)
                else:
                    st.session_state.graph.add_triple(st.session_state.src, st.session_state.rel, st.session_state.tgt)
        st.session_state.document.append([st.session_state.src, st.session_state.rel, st.session_state.tgt])
    
    # if st.button("calculate"):
    #     query_embedding = get_doc_embedding()
    #     output_table = graph_doc_matching(query_embedding)
    #     st.table(output_table)


def set_natural_query(question_templates:dict)->None:
    event_code_dict = {value:key for key, value in json.load(open("gdelt_event_dict.json")).items() if "010" in key or "043" in key or "051" in key or "042" in key or "190" in key}
    # event_code_dict = {value:key for key, value in json.load(open("gdelt_event_dict.json")).items()}
    # event_code_dict = {value:key for key, value in json.load(open("gdelt_event_dict.json")).items() if key in ["036","18"]}
    input_text = st.selectbox("Select a question", question_templates.keys(), index=0)

    selected_relation = st.selectbox("Select a theme", list(event_code_dict.keys()), index=0, key="relation", on_change=set_relation)

    #st.write(list(event_code_dict.keys()).index("Useunconventionalviolence,notspecifiedbelow"))

    if st.button("submit"):
        src, relation, target = process_text(input_text, question_templates)
        relation = [relation]

        # st.session_state.document.append([st.session_state.src, st.session_state.rel, st.session_state.tgt])
        if [src, event_code_dict[selected_relation], target] not in st.session_state.document:
            st.session_state.document.append([src, event_code_dict[selected_relation], target])

        ### Displays Query ###
        if src==[]:
            if target==[]:
                if relation==[]:
                    st.session_state.graph.add_triple("Unknown", "Unknown", "Unknown")
                else:
                    st.session_state.graph.add_triple("Unknown", relation, "Unknown")
            else:
                if relation==[]:
                    st.session_state.graph.add_triple("Unknown", "Unknown", target)
                else:
                    st.session_state.graph.add_triple("Unknown", relation, target)
        else:
            if target==[]:
                if relation==[]:
                    st.session_state.graph.add_triple(src, "Unknown", "Unknown")
                else:
                    st.session_state.graph.add_triple(src, relation, "Unknown")
            else:
                if relation==[]:
                    st.session_state.graph.add_triple(src, "Unknown", target)
                else:
                    st.session_state.graph.add_triple(src, relation, target)

    st.subheader("Formulated Graph Query")
    config = Config(height=200,width=800, 
                    nodeHighlightBehavior=True,
                    node_size=300,
                    highlightColor="#F7A7A6", 
                    directed=True, 
                    collapsible=False,
                    maxZoom=2,
                    minZoom=0.2,
                    initialZoom=1,
                    node={'labelProperty':'label'},
                    link={'labelProperty': 'label', 'renderLabel': True})
    agraph(list(st.session_state.graph.getNodes()), list(st.session_state.graph.getEdges()), config)

question_templates={
    "Any China and Taiwan dispute?":("CHINA", st.session_state.rel, "TAIWAN"),
    # "Any China and Philippines dispute?": ("CHINA", st.session_state.rel, "PHILIPPINE"),
    "Any China and Philippines dispute?": ("PHILIPPINE", st.session_state.rel, "CHINA"),
    "Any China and Indonesia dispute?": ("CHINA", st.session_state.rel, "INDONESIA"),
    "Any China and Vietnam dispute": ("CHINA", st.session_state.rel, "VIETNAM"),
}

### query by triple ###
#set_query()

### query by sentence ###
set_natural_query(question_templates)

#st.write(st.session_state.document)

if st.button("query"):
    query_embedding, type_list = get_doc_embedding(use_local=True, document=st.session_state.document)
    output_table = graph_doc_matching(query_embedding,type_list, use_local=True, use_cluster=True)
    # st.write(output_table)
    url_ids = output_table["id"]
    dataset_obj = Dataset.get(dataset_project = "datasets/gdelt", dataset_name="raw_gdelt_2021")
    url2id = pd.read_csv(os.path.join(dataset_obj.get_local_copy(), "url2id.txt")).loc[1:,:]
    output_table["id"] = output_table["id"].astype("int32")
    urls = output_table.merge(url2id, how="left", on="id").rename(columns={"value": "top 20 URLs"})#.tolist()
    st.table(urls)
# st.subheader("Registered Query")
# config = Config(height=500,width=700, 
#                 nodeHighlightBehavior=True,
#                 highlightColor="#F7A7A6", 
#                 directed=True, 
#                 collapsible=False,
#                 node={'labelProperty':'label'},
#                 link={'labelProperty': 'label', 'renderLabel': True})
# agraph(list(st.session_state.graph.getNodes()), list(st.session_state.graph.getEdges()), config)


