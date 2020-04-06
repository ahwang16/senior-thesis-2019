# prelim_eval_cn.py

import sys, os, requests, time, json
from IPython import embed
from nltk.corpus import brown

# CORPORA_PATH = '/Users/alyssahwang/Documents/workspace2/seniorthesis/data'
CORPORA_PATH = "../data/"

def check_and_sleep(start_time, curr_call_ct):
	if curr_call_ct % 120 == 0:  # every 120 sleep for a minute
		print('short: {}'.format(curr_call_ct))
		time.sleep(60)

	if curr_call_ct % 3600 == 0:
		print("long sleep: {}".format(curr_call_ct))
		curr_time = time.time()
		d = max(60 * 60 - (curr_time - start_time) + 2, 2)
		print("  d={}".format(d))
		time.sleep(d)
		start_time = time.time()
	return start_time


def query_conceptnet(w, rel, curr_call_ct, start_time):
	rel_nodes = set()
	next_pg = run_query(w, rel, rel_nodes, '/c/en/{}?offset=0&limit=1000'.format(w))
	curr_call_ct += 1

	while next_pg is not None:
		start_time = check_and_sleep(start_time, curr_call_ct)
		next_pg = run_query(w, rel, rel_nodes, next_pg)
		curr_call_ct += 1

	return rel_nodes, curr_call_ct, start_time


def run_query(w, rel, rel_nodes, page_info):
	obj = requests.get('http://api.conceptnet.io{}'.format(page_info))
	try:
		obj = obj.json()
	except:
		print(obj)
	edges = obj['edges']
	for e in edges:
		if e['rel']['label'] != rel: continue
		if e['start']['label'] != w and e['start']['@id'].startswith('/c/en/'):
			if " " not in e['start']['label'] :
				rel_nodes.add(e['start']['label'])
		elif e['end']['label'] != w and e['end']['@id'].startswith('/c/en/'):
			if " " not in e['end']['label'] :
				rel_nodes.add(e['end']['label'])

	if 'view' in obj.keys():
		return obj['view'].get('nextPage', None)
	return None


def get_related(vocab_name, outname):
	print("sleeping for 1 hour so starting fresh with request limits")
	time.sleep(60*60)

	print("starting requests")

	outf = open(os.path.join(CORPORA_PATH, outname), 'w', encoding='utf-8')
	outf.write('word,related_lst\n')
	# outf = open(os.path.join(CORPORA_PATH, outname), 'a', encoding='utf-8')

	# f = open(os.path.join(CORPORA_PATH, vocab_name), 'r', encoding='utf-8')
	# lines = f.readlines()
	# print("need {} words".format(len(lines)))
	start_time = time.time()
	i = 0
	count = 0
	for w in vocab_name:
		if count % 100 == 0 : print(count)
		count += 1

		start_time = check_and_sleep(start_time, i)

		# start_time = time.time()

		# w = l.strip()
		r, i, start_time = query_conceptnet(w, 'RelatedTo', i, start_time)
		outf.write('{},{}\n'.format(w, json.dumps(list(r))))


if __name__ == "__main__":
	vocab = set(brown.words(category=["fiction"]))
	print(len(vocab))
	get_related(vocab, "brown_cn_gold_1.txt")

