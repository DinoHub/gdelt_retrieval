#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 21:19:31 2021

@author: aaron
"""

import json
import jsonlines
import graph_functions
from itertools import chain

''' reads a json file '''
def read_json(jsonfile):
    with open(jsonfile, 'r') as file:
        file_object = json.loads(file.read())
    return file_object

''' Writes to a jsonl file'''
def to_jsonl(filename:str, file_obj):
    resultfile = open(filename, 'wb')
    writer = jsonlines.Writer(resultfile)
    writer.write_all(file_obj)


data = read_json('/mnt/d/storage/takeout/preds_gtt.json')

def ingest_gtt(data):
    graph_functions.clear_graph()
    for dockey in data.keys():
        print(dockey)
        doc = data[dockey]        
        pred_templates = doc["pred_templates"]
        
        for event in pred_templates:
            incident = event["incident_type"]
            roles = [(event[key], key) for key in event.keys() if key != "incident_type"]

            # Create event node            
            event_id = "{}_{}".format(incident, dockey)
            graph_functions.merge_node(incident, {'keyword': event_id})

            # Create role nodes
            for role, key in roles:
                role = ' '.join(list(chain.from_iterable(role)))
                if role!='':                    
                    graph_functions.merge_node(key, {'keyword': role})
                    graph_functions.merge_edge(key, role, incident, event_id, "related", {'keyword': dockey})


ingest_gtt(data)
            
            
        
        

