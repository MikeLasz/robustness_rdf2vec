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

label_predicates = ["http://swrc.ontoware.org/ontology-07"]
kg = KG(location="../data/aifbfixed_complete.n3", file_type="n3", label_predicates=label_predicates)

#kg.visualise()

walkers = {RandomWalker(4, 5, UniformSampler())}



transformer = RDF2VecTransformer(Word2Vec(sg=1), walkers=walkers)

def get_entities():
    g = rdflib.Graph()
    g.parse("../data/aifbfixed_complete.n3", format="n3")

    q_emp = g.query("""SELECT ?entity
WHERE {
  ?entity rdf:type ?type.
  ?type rdfs:subClassOf* :AcademicStaff.
}""")
    entities = []
    for row in q_emp:
        entities.append(row[0])
    q_proj = g.query("""SELECT ?entity
    WHERE {
      ?entity rdf:type ?type.
      ?type rdfs:subClassOf* :Project.
    }""")
    for row in q_proj:
        entities.append(row[0])
    return(entities)
# Entities should be a list of URIs that can be found in the Knowledge Graph

#entities = ["http://www.aifb.uni-karlsruhe.de/Forschungsgebiete/viewForschungsgebietOWL/id100instance",
 #           "http://www.aifb.uni-karlsruhe.de/Forschungsgebiete/viewForschungsgebietOWL/id102instance",
  #          "http://www.aifb.uni-karlsruhe.de/Personen/viewPersonOWL/id15instance"]

entities = get_entities()

embeddings = transformer.fit_transform(kg, entities)

walk_tsne = TSNE(random_state=42)
X_tsne = walk_tsne.fit_transform(embeddings)

plt.figure(figsize=(15,15))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

texts = []
for x, y, lab in zip(X_tsne[:, 0], X_tsne[:, 1], entities):
    lab = lab.split('/')[-1]
    text = plt.text(x, y, lab)
    texts.append(text)

adjust_text(texts, lim=5, arrowprops=dict(arrowstyle="->", color="r", lw=0.5))
plt.show()