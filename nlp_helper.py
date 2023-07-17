# For NLP pipelines
# import spacy
import numpy as np
import json
# from spacy.tokens import Span
from stanza.server import CoreNLPClient

'''
NLP HELPER FUNCTIONS
'''

NER_GROUP_LABEL='GROUP'

'''
A list of groups to also tag during Named Entity Recognition parsing, under the
NER label "GROUP"
'''
NER_GROUPS = [ 'medical officials', 'doctors', 'democrats', 'republicans', 'immigrants', 'government officials', 'legislators', 'health officials', 'communists', 'socialists', 'libertarians', 'children', 'parents', 'elders', 'the elderly', 'Americans', 'patriots', 'governors', 'legislators', 'migrants', 'terrorists', 'the public', 'citizens', 'epidemiologists', 'representatives', 'congress', 'hospital workers', 'scientists', 'mayors', 'celebrities', 'the media' ]

# Maybe somehow these should get put into BERT vectors so we could find similar enough synonyms and not have to hand code all of them...? As long as they are some epsilon threshold away in BERT space and contain similar words?

def label_ner_groups(doc, groups):
  '''
  DEPRECATED

  Given a spacy doc object, search it for instances of any of the groups passed
  in parameters, and tag them as a GROUP.

  :param doc: A spacy doc object.
  :param groups: A list of group strings to label with the NER GROUP label.

  Mutates the doc object so there's no need to return
  '''
  return -1
  # doc_text = [ token.text.lower() for token in doc ]
  # entities = []
  # for group in groups:
  #   group_tokens = group.split(' ')
  #   n_gram_tokens = []
  #   for i in range(len(doc_text)-len(group_tokens)):
  #     n_gram = []
  #     for j in range(len(group_tokens)):
  #       n_gram.append(doc_text[i+j])
  #     n_gram_tokens.append(n_gram)
  #   matches = list(map(lambda el: sum(el) == len(el), np.array(n_gram_tokens) == group_tokens))
  #   indices = np.where(matches)[0]
  #   for i in indices:
  #     ent = Span(doc, i, i+len(group_tokens), label=NER_GROUP_LABEL)
  #     entities.append(ent)
  # doc.set_ents(entities, default="unmodified")

def find_entity_set_text(nlp, text):
  '''
  Get a set of all of the entities mentioned in a piece of text.

  :param nlp: The spacy parser to use.
  :param text: The text to search for entities.
  '''
  VALID_ENTITY_CATEGORIES = ['ORG','PERSON','NORP','WORK_OF_ART','GPE','GROUP','FAC','LAW']
  doc = nlp(text)
  # label_ner_groups(doc, NER_GROUPS)
  entities = doc.ents
  valid_category_entities = filter(lambda entity: entity.label_ in VALID_ENTITY_CATEGORIES, entities)
  # return entities
  str_entities = map(lambda el: str(el), valid_category_entities)
  return set(list(str_entities))

def link_entity_set(nlp, text):
  nlp.add_pipe('entityLinker', last=True)
  text = ' '.join(entity_set)
  doc = nlp(text)
  nlp.remove_pipe('entityLinker')
  return doc

def find_entity_set_df(nlp, df):
  '''
  Get a set of all the entities mentioned in an entire dataframe
  of articles.

  :param nlp: The spacy parser to use
  :param df: The data frame to get article text from and discover entities in.
  '''
  entity_set = set([])
  for row in df.iterrows():
    row_data = row[1]
    entities = find_entity_set_text(nlp, row_data.article_data_raw)
    entity_set = entity_set.union(entities)
  return entity_set

def write_entity_set_to_json(entity_set, out_path):
  '''
  Write a set of named entities to a JSON file at a specified path.

  :param entity_set: The set of entities (strings) to write
  :param out_path: The path to write to
  '''
  f = open(out_path, 'w')
  json.dump(list(entity_set), f)
  f.close()

def split_into_sentences(text):
  '''
  Split a piece of text into sentence with the StanfordNLP library.

  :param text: The text to split into sentences.
  '''
  with CoreNLPClient(
    annotators=['tokenize','ssplit'],
    timeout=30000,
    memory='5G'
  ) as client:
    ann = client.annotate(text)
    return [ text[sent.characterOffsetBegin:sent.characterOffsetEnd] for sent in ann.sentence ]

