# compose_flask/app.py
from datetime import date, timedelta
import datetime as dt
import json
import time
#import pandas as pd

### REQUIRED FOR NEO4J TO FINISH SETTING UP
time.sleep(10)

# ### Community Package
# from py2neo import Graph, Node, Relationship
# graph = Graph("http://neo4j:7474/db/data/")

# # begin a transaction
# tx = graph.begin()

# alice = Node("Person", name="Alice")
# tx.create(alice)
# tx.commit()

### Official Package
from neo4j import GraphDatabase

uri = "bolt://127.0.0.1:7687/db/data"
#uri = "bolt://neo4j:7687/db/data"
driver = GraphDatabase.driver(uri, auth=("neo4j", "test"), encrypted=False)

# with driver.session() as session:
#       #DROP CONSTRAINT ON (<label_name>) ASSERT <property_name> IS UNIQUE
#       session.run("CREATE CONSTRAINT ON (d:datetime) ASSERT d.name IS UNIQUE")
#       session.run("CREATE CONSTRAINT ON (d:datetime) ASSERT d.time IS UNIQUE")

### ================================= GRAPH QUERY FUNCTIONS ===================================== ###

### Show total node and degree distribution
def query_by_day(date):     
      query = "MATCH (s:case)-[r]->(t:datetime) WHERE t.name='{}' RETURN DISTINCT s.name, r.contact_location, t.name".format(date) #DATE
      with driver.session() as session:
            return session.run(query)


# MERGE (attack:conflict {keyword: 'fight'})

# ### ================================= GRAPH MANAGEMENT FUNCTIONS ===================================== ###
'''
CREATE NODE
parameters: 
- node_label(str) = Node type (Location, Person, Organisation), 
- node_attributes(dict) = Attributes of Node (Name, Address, Location ,Coordinates) 
'''
# create_node creates duplicates and causes memory problems, use merge_node instead
def merge_node(node_label, node_attributes):
      with driver.session() as session:
            node_attributes = "{"+", ".join([k+" : '"+node_attributes[k]+"'" for k in node_attributes.keys()])+"}"
            #print("MERGE (p:{} {}) RETURN p".format(node_label, node_attributes))
            return session.run("MERGE (p:{} {}) RETURN p".format(node_label, node_attributes)).single().value()


'''
CREATE EDGE
parameters: 
- source_node (redis node object) = starting node of edge
- target_node (redis node object) = ending node of edge
- relationship_label (str) = relationship between the nodes (contact location, role in organisation)
- edge_attributes(dict) = Attributes of relationship (Name, Address, Location ,Coordinates) 
'''
# create_edge creates duplicates and causes memory problems, use merge_edge instead

def merge_edge(source_node_label, source_node_name, target_node_label, target_node_name, relation_type, edge_attributes):
      #DIRECTED
      #"MATCH (s:{} {{name: '{}'}}), (t:{} {{name:'{}'}}) CREATE (s)-[e:{} {}]->(t) RETURN e".format(source_node_label, source_node_name, target_node_label, target_node_name, relation_type ,edge_attributes)
      #UNDIRECTED
      with driver.session() as session:
            edge_attributes = "{"+", ".join([k+" : '"+edge_attributes[k]+"'" for k in edge_attributes.keys()])+"}"
            print("MATCH (s:{} {{keyword: '{}'}}), (t:{} {{keyword:'{}'}}) MERGE (s)-[e:{} {}]-(t) RETURN e".format(source_node_label, source_node_name, target_node_label, target_node_name, relation_type ,edge_attributes))
            return session.run("MATCH (s:{} {{keyword: '{}'}}), (t:{} {{keyword:'{}'}}) MERGE (s)-[e:{} {}]-(t) RETURN e".format(source_node_label, source_node_name, target_node_label, target_node_name, relation_type ,edge_attributes))#.single().value()

'''
USING GENERAL COMMANDS -   
DELETE is used to remove both nodes and relationships. Note that deleting a node 
also deletes all of its incoming and outgoing relationships.
To delete a node and all of its relationships:

DELETE NODE
parameters: 
- node_label(str) = Node type to delete 
- attribute_del_type(str) = Attribute key/type used as filter condition to delete(match by name/location/id)
- attribute_value(str) = filter condition value 

EXAMPLE:
GRAPH.QUERY DEMO_GRAPH "MATCH (p:person {name:'Jim'}) DELETE p"
'''
def delete_node(node_label, attribute_del_type, attribute_value):
      ### ADD PARAMETERS ASSERTION HERE ###
      with driver.session() as session:
            return session.run("MATCH (:{} {'{}':'{}'})-[r:{}]->() DELETE r".format(node_label, attribute_del_type, attribute_value, relationship_label)).single().value()

'''
DELETE EDGE
This query will delete all 'relationship_label' outgoing relationships from the node with the attribute_del_type = attribute_value.
parameters: 
- node_label(str) = Node type to delete 
- attribute_del_type(str) = Attribute key/type used as filter condition to delete(match by name/location/id)
- attribute_value(str) = filter condition value

EXAMPLE:
GRAPH.QUERY DEMO_GRAPH "MATCH (:person {name:'Jim'})-[r:friends]->() DELETE r"
'''
def delete_edge(node_label, attribute_del_type, attribute_value, relationship_label):
      ### ADD PARAMETERS ASSERTION HERE ###
      with driver.session() as session:
            return session.run("MATCH (:{} {'{}':'{}'})-[r:{}]->() DELETE r".format(node_label, attribute_del_type, attribute_value, relationship_label)).single().value()

'''
EDIT NODE
USING GENERAL REDIS COMMANDS - 
SET is used to create or update properties on nodes and relationships. Must use a unique node identifier as match condition

EXAMPLE:
GRAPH.QUERY DEMO_GRAPH "MATCH (n { name: 'Jim' }) SET n.name = 'Bob'"
'''
def edit_node_attribute(attribute_edit_type, attribute_value, new_attribute_value):
      ### ADD PARAMETERS ASSERTION HERE ###
      with driver.session() as session:
            return session.run("MATCH (n: { '{}': '{}' }) SET n.{} = '{}'".format(attribute_edit_type, attribute_value, attribute_edit_type, new_attribute_value)).single().value()


def clear_graph():
      with driver.session() as session:
            session.run("MATCH p=()-->() DELETE p")
            return None

