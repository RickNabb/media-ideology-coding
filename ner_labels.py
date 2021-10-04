from data_collector import *

path = str(input("path: "))
index = int(input("index: "))
word = 'mask'
num = 1

df = pd.read_csv(path)

nlp = spacy.load("en_core_web_sm")

def raw_text(index):
	return df.iloc[index].article_data_raw

def create_paragraphs(word, num):
	text = raw_text(index)
	return get_keyword_paragraph(text, word, num)

def display_ent(art):
	if art.ents:
		for ent in art.ents:
			print(ent.text+"; label = "+ent.label_+": "+str(spacy.explain(ent.label_)))

pars = create_paragraphs(word,num)
pars_str = ' '.join(map(str,pars))
res = nlp(pars_str)
display_ent(res)	
