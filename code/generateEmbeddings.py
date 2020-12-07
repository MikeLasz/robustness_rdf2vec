import pyrdf2vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import UniformSampler
from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import rdflib
from adjustText import adjust_text
import numpy as np
import random

data = "mutag"
if data == "aifb":
    label_predicates = ["http://swrc.ontoware.org/ontology-07"]
    kg = KG(location="../data/aifbfixed_complete.n3", file_type="n3", label_predicates=label_predicates)
elif data == "bgs":
    label_predicates = ["http://data.bgs.ac.uk/ref/Lexicon"]
    kg = KG(location="../data/bgs.nt", file_type="nt", label_predicates=label_predicates)
elif data == "mutag":
    label_predicates = ["http://dl-learner.org/carcinogenesis#isMutagenic"]
    kg = KG(location="../data/mutag.xml", file_type="xml", label_predicates=label_predicates)

walkers = {RandomWalker(4, 5, UniformSampler())}
transformer = RDF2VecTransformer(Word2Vec(), walkers=walkers)

def get_entities(data):
    g = rdflib.Graph()
    if data == "aifb":
        g.parse("../data/aifbfixed_complete.n3", format="n3")

        q = g.query("""SELECT ?entity
            WHERE {
              ?entity rdf:type ?type.
            }""")
        entities = []
        for row in q:
            entities.append(row[0])
    if data == "bgs":
        g.parse("../data/bgs.nt", format="nt")

        q_rock = g.query("""SELECT ?entity
            WHERE {
              ?entity rdf:type ?type.
            }""")
        entities = []
        for row in q_rock:
            entities.append(row[0])
    if data == "mutag":
        g.parse("../data/mutag.xml", format="xml")

        q = g.query("""SELECT ?entity
            WHERE {
              ?entity rdf:type ?type.
            }""")
        entities = []
        for row in q:
            entities.append(row[0])
    return(entities)

entities = get_entities(data)

# copied from https://towardsdatascience.com/how-to-create-representations-of-entities-in-a-knowledge-graph-using-pyrdf2vec-82e44dad1a0
# Make sure that every entity can be found in our KG
filtered_entities = [e for e in entities if e in kg._entities]
not_found = set(entities) - set(filtered_entities)
print(f'{not_found} could not be found in the KG! Removing them...')
entities = filtered_entities

random_states = np.arange(10)
np.random.shuffle(random_states)

for iter in range(len(random_states)):
    random.seed(random_states[iter])
    embeddings = transformer.fit_transform(kg, entities)
    np.save("embeddings/" + data + "/embedding" + str(iter), embeddings)