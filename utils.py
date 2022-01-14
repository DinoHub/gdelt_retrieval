from s3utils import *
import json, os
import pandas as pd
from annotated_text import annotated_text

def read_jsonl(jsonfile):
    with open(jsonfile, 'r') as file:
        file_object = [json.loads(sample) for sample in file]
    return file_object

def read_ops(bucket,endpoint_uri,aws_key,aws_secret,file_path):
    s3_client = S3Utils(bucket,endpoint_uri,aws_key,aws_secret, upload_multi_part=True).s3
    s3_object = s3_client.Object(bucket,file_path).get()
    file_content = s3_object['Body'].read().decode('utf-8')
    json_content = json.loads(file_content)
    return json_content

def remove_led_prefix_from_tokens(tokens):
    return [token[1:] if "Ġ" in token else token for token in tokens]

def combine_predictions(dygiepp_preds:list, gtt_preds:list=None):        
    dygiepp_preds = pd.json_normalize(dygiepp_preds, meta=["predicted_entities", "predicted_events", "sentences"])
    dygiepp_preds.set_index("doc_key", inplace=True)
    dygiepp_preds.reset_index(drop=True, inplace=True)

    gtt_preds = pd.DataFrame.from_dict(gtt_preds, orient="index")    
    gtt_preds.reset_index(drop=True, inplace=True)
    #combined_df = gtt_preds.merge(dygiepp_preds, how="left", left_index=True, right_index=True)    
    combined_df = pd.concat([gtt_preds, dygiepp_preds], axis=1)
    return combined_df 

def get_ops_gold(json_content):
    dict_list = []
    count = 0
    for log in json_content:
        if log['articles']==[]:
            continue
        for article in log['articles']:
            doc_dict = {}
            cob = log['incidentCob'].replace("[","")
            cob = cob.replace("]","")
            doc_dict['docid'] = "TEST3-OPS-" + str(count).zfill(4)
            doc_dict['orig_docid'] = str(article['id'])
            doc_dict['seq_attack'] = cob
            doc_dict['sitename'] = [item['name'] for item in log['sites']]
            doc_dict['date'] = [item['date'] for item in log['sites']]
            doc_dict['day'] = [item['day'] for item in log['sites']]
            doc_dict['country'] = [item['location']['country'] for item in log['sites']]
            doc_dict['city'] = [item['location']['city'] for item in log['sites']]
            doc_dict['site_conops'] = log['incidentConops']
            for entry in log['sites']:
                doc_dict['affiliation'] = [item['affiliation'] for item in entry['perps']]
                doc_dict['age_group'] = [item['ageGroup'] for item in entry['perps']]
                doc_dict['gender'] = [item['gender'] for item in entry['perps']]
                doc_dict['attire'] = [item['attire'] for item in entry['perps']]
                doc_dict['country_of_residence'] = [item['countryOfResidence'] for item in entry['perps']]
                tp_list = []
                for entry_3 in entry['targetProximity']:
                    tp_list.append(entry_3['targetDetails'])
                doc_dict['target_type_in_proximity_details'] = tp_list
                means_list = []
                for entry_2 in entry['conops']['actionTerms']:
                    means_list.append(entry_2['means']['means'])
                doc_dict['means'] = means_list
            doc_dict['target_type'] = [item['target']['targetDetails'] for item in log['sites']]
            dict_list.append(doc_dict)
            count += 1
    return dict_list

def get_info(doc, gtt_classes, dygiepp_classes):
    gtt_templates = doc["pred_templates"]
    dygiepp_entities=doc["predicted_entities"]
    dygiepp_events=doc["predicted_events"]
    dygiepp_relations=doc["predicted_relations"]

    possible_fields = []    
    if "incident_type" in gtt_classes.keys() and "triggers" in dygiepp_classes.keys():
        gtt_incidents = [template["incident_type"] for template in gtt_templates]
        dygiepp_triggers = [event[-2] for event in dygiepp_events]
        possible_fields = [gtt_incidents, dygiepp_triggers]
    elif "arguments" in dygiepp_classes.keys() and "entities" in dygiepp_classes.keys() and "relations" in dygiepp_classes.keys():
        gtt_fields = [template[role] for role in gtt_classes["roles"] for template in gtt_templates if role in template.keys()]       
        dygiepp_arguments = [arguments[0] for event in dygiepp_events for arguments in event[-1] if arguments[1] in dygiepp_classes["arguments"]]
        dygiepp_entities = [entity[0][0] for entity in dygiepp_entities if entity[1] in dygiepp_classes["entities"]]        
        dygiepp_relations = [(relation[0], relation[1]) for relation in dygiepp_relations if relation[2] in dygiepp_classes["relations"]]
        dygie_fields = list(set(dygiepp_arguments+dygiepp_entities))+dygiepp_relations
        possible_fields = [gtt_fields, dygie_fields]
    elif "arguments" in dygiepp_classes.keys():
        gtt_fields = [template[role] for role in gtt_classes["roles"] for template in gtt_templates if role in template.keys()]        
        dygiepp_arguments = list(set([arguments[0] for event in dygiepp_events for arguments in event[-1] if arguments[1] in dygiepp_classes["arguments"]]))
        possible_fields = [gtt_fields, dygiepp_arguments]
    elif "relations" in dygiepp_classes.keys():
        gtt_fields = [template[role] for role in gtt_classes["roles"] for template in gtt_templates if role in template.keys()]       
        dygiepp_relations = [(relation[0], relation[1]) for relation in dygiepp_relations if relation[2] in dygiepp_classes["relations"]]
        possible_fields = [gtt_fields, dygiepp_relations]
    else:
        gtt_fields = [template[role] for role in gtt_classes["roles"] for template in gtt_templates if role in template.keys()]        
        dygiepp_entities = list(set([entity[0][0] for entity in dygiepp_entities if entity[1] in dygiepp_classes["entities"]]))        
        possible_fields = [gtt_fields, dygiepp_entities]        
    return possible_fields

#  attempt to map MUC4 and dygiepp classes to ops classes
def fill_postulated_facts(doc):    
    postulated_facts = {
     'doctext': doc["doctext"],
     'seq_attack': get_info(doc, gtt_classes={"incident_type": True}, dygiepp_classes={"triggers": True}),
     'sitename': get_info(doc, gtt_classes={"roles": ["Target"]}, dygiepp_classes={"arguments": ["Place"], "entities": ['FAC'], "relations": ["PART-WHOLE.Geographical"]}),
     'date': get_info(doc, gtt_classes={"roles": []}, dygiepp_classes={"arguments": ["Time"]}),
     'day': get_info(doc, gtt_classes={"roles": []}, dygiepp_classes={"arguments": ["Time"]}),
    #  'time': None,
     'country': get_info(doc, gtt_classes={"roles": []}, dygiepp_classes={"arguments": ["Place", "Destination"], "entities": ['GPE'], "relations": ["PART-WHOLE.Geographical"]}),
    #  'region': get_info(doc, gtt_classes={"roles": []}, dygiepp_classes={"entities": ["LOC", "GPE"]}),
    #  'province': get_info(doc, gtt_classes={"roles": []}, dygiepp_classes={"entities": ["LOC", "GPE"]}),
    #  'sub-region': get_info(doc, gtt_classes={"roles": []}, dygiepp_classes={"entities": ["LOC", "GPE"]}),
     'city': get_info(doc, gtt_classes={"roles": []}, dygiepp_classes={"arguments": ["Place", "Destination"], "entities": ['GPE'], "relations": []}),
    #  'longitude': None,
    #  'latitude': None,
     'affiliation': get_info(doc, gtt_classes={"roles": ["PerpOrg"]}, dygiepp_classes={"arguments": ["Org"], "entities": ['ORG'], "relations": ["ORG-AFF.Membership", "GEN-AFF.Citizen-Resident-Religion-Ethnicity", "ORG-AFF.Employment"]}),
     'age_group': None,
     'gender': None,
     'attire': None,
     'country_of_residence': None,
     'site_conops': None,
    #  'suicide': None,
     'target_type': get_info(doc, gtt_classes={"roles":["Victim", "Target"]}, dygiepp_classes={"arguments": ["Victim", "Target"]}), #get_info(doc, gtt_classes={"roles":["Victim", "Target"]}, dygiepp_classes={"entities": ["PER", "ORG"]}),
    #  'target_type_details': None,
     'target_type_in_proximity_details': None,
     "means": get_info(doc, gtt_classes={"roles": ["Weapon"]}, dygiepp_classes={"arguments": ["Instrument", "Vehicle"], "entities": ['WEA', 'VEH']})
     }
    return pd.DataFrame.from_dict(postulated_facts, orient="index")    

def convert_streamlit_viz(doc):
    sentences = doc['sentences']
    events_doc = doc['predicted_events']
    result = []
    offset=0
    new_sent_list=[]
    for sent, events_sent in zip(sentences, events_doc):
        sent = [' '+token+' ' for token in sent]
        for event in events_sent:
            for element in event:
                if len(element)>4: #argument
                    start = int(element[0])-offset
                    try:
                        end = int(element[1])-offset+1 # add 1 to the end
                        num_blank_tokens = end-start-1
                        # When we join the words together we need to preserve the token indices for the spans
                        blank_tokens = [['<blank>'] for _ in range(num_blank_tokens)]
                        argument = (' '+' '.join([token[0] if isinstance(token, tuple()) else token for token in sent[start:end]])+' ', ' '+element[2])                        
                        sent = sent[:start]+[argument]+blank_tokens+sent[end:]

                    except:
                        end = int(element[1])-offset+1 # add 1 to the end
                        num_blank_tokens = end-start-1
                        # When we join the words together we need to preserve the token indices for the spans
                        blank_tokens = [['<blank>'] for _ in range(num_blank_tokens)]
                        argument = (' '+' '.join([token[0] if isinstance(token, tuple()) else token[0] for token in sent[start:end]])+' ', ' '+element[2])
                        sent = sent[:start]+[argument]+blank_tokens+sent[end:]
                    
                else: #trigger
                    start = int(element[0])-offset
                    end = int(element[0])-offset+1 # add 1 to the end
                    num_blank_tokens = end-start-1
                    # When we join the words together we need to preserve the token indices for the spans
                    blank_tokens = [['<blank>'] for _ in range(num_blank_tokens)]    
                    trigger = (' '+' '.join(sent[start:end][0])+' ', ' '+element[1], "#afa")
                    sent = sent[:start]+[trigger]+blank_tokens+sent[end:]

        new_sent_list.append([token for token in sent if token != ['<blank>']])
        offset=offset+len(sent)
    return new_sent_list


def display_dygiepp_annotations(doc):
    input_list = []
    for sample in convert_streamlit_viz(doc):
        if isinstance(sample, tuple):
            input_list=input_list+(sample)
        else:   
            input_list=input_list+sample
        
        if len(input_list)>80:
            annotated_text(*input_list)
            input_list=[]

def process_qa_output(text):
    text=str(text)
    return text.strip().replace("</s>", "")

