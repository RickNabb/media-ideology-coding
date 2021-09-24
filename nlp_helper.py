# For NLP pipelines
import spacy
from spacy.tokens import Span

'''
NLP HELPER FUNCTIONS
'''

NER_GROUP_LABEL='GROUP'

'''
A list of groups to also tag during Named Entity Recognition parsing, under the
NER label "GROUP"
'''
NER_GROUPS = [ 'medical officials', 'doctors', 'democrats', 'republicans', 'immigrants', 'government officials', 'legislators', 'health officials', 'communists', 'socialists', 'libertarians', 'children', 'parents', 'elders', 'the elderly', 'Americans', 'patriots', 'governors', 'legislators', 'migrants', 'terrorists', 'the public', 'citizens', 'epidemiologists', 'representatives', 'congress', 'hospital workers', 'scientists', 'mayors', 'celebrities', 'the media' ]

def label_ner_groups(doc, groups):
  '''
  Given a spacy doc object, search it for instances of any of the groups passed
  in parameters, and tag them as a GROUP.

  :param doc: A spacy doc object.
  :param groups: A list of group strings to label with the NER GROUP label.
  '''
  doc_text = [ token.text.lower() for token in doc ]
  entities = []
  for group in groups:
    group_tokens = group.split(' ')
    n_gram_tokens = []
    for i in range(len(doc_text)-len(group_tokens)):
      n_gram = []
      for j in range(len(group_tokens)):
        n_gram.append(doc_text[i+j])
      n_gram_tokens.append(n_gram)
    matches = list(map(lambda el: sum(el) == len(el), np.array(n_gram_tokens) == group_tokens))
    indices = np.where(matches)[0]
    for i in indices:
      ent = Span(doc, i, i+len(group_tokens), label=NER_GROUP_LABEL)
      entities.append(ent)
  doc.set_ents(entities, default="unmodified")