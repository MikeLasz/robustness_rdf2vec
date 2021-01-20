import pyrdf2vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import UniformSampler
from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec

import progressbar
import rdflib
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="aifb", help="Select the data: aifb, mutag, bgs")
parser.add_argument("--nlp_model", type=str, default="skip-gram", help="Select the NLP-model: skip-gram or cbow")
parser.add_argument("--num_embeddings", type=int, default=10, help="Number of embeddings that should be generated")
args = parser.parse_args()

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

if __name__ == "__main__":
    data = args.data
    nlp_model = args.nlp_model
    num_embeddings = args.num_embeddings

    # construgt the KG
    if data == "aifb":
        kg = KG(location="../data/aifbfixed_complete.n3", file_type="n3")
    elif data == "bgs":
        #label_predicates = ["http://data.bgs.ac.uk/ref/Lexicon"]
        kg = KG(location="../data/bgs.nt", file_type="nt") #, label_predicates=label_predicates)
    elif data == "mutag":
        #label_predicates = ["http://dl-learner.org/carcinogenesis#isMutagenic"]
        kg = KG(location="../data/mutag.xml", file_type="xml")#, label_predicates=label_predicates)

    # define the random walker
    walkers = {RandomWalker(8, 10, UniformSampler())} # walks of length 8, 10 walks per entity

    # sg: skipgram vs CBOW. 1 for skip-gram
    if nlp_model=="skip-gram":
        sg = 1
    else:
        sg = 0

    transformer = RDF2VecTransformer(Word2Vec(window=5, sg=sg, negative=25), walkers=walkers)
    entities = get_entities(data)

    # copied from https://towardsdatascience.com/how-to-create-representations-of-entities-in-a-knowledge-graph-using-pyrdf2vec-82e44dad1a0
    # Make sure that every entity can be found in our KG
    filtered_entities = [e for e in entities if e in kg._entities]
    not_found = set(entities) - set(filtered_entities)
    print(f'{not_found} could not be found in the KG! Removing them...')
    entities = filtered_entities

    random_states = np.arange(num_embeddings)
    np.random.shuffle(random_states)

    # compute the emebddings
    with progressbar.ProgressBar(max_value=num_embeddings) as bar:
        bar.update(0)
        for iter in range(num_embeddings):
            random.seed(random_states[iter])
            embeddings = transformer.fit_transform(kg, entities)
            np.save("embeddings/" + data + "/" + nlp_model + "/embedding" + str(iter), embeddings)
            bar.update(iter + 1)