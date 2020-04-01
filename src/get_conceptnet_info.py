import sys, os, requests, time, json

from IPython import embed

CORPORA_PATH = '.'#'/Users/emilyallaway/OneDrive/Columbia/Research/corpora/'

REL_DIR = '/Users/emilyallaway/Desktop/big-corpora/related-words/'

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
    obj = obj.json()
    edges = obj['edges']
    for e in edges:
        if e['rel']['label'] != rel: continue
        if e['start']['label'] != w and e['start']['@id'].startswith('/c/en/'):
            rel_nodes.add(e['start']['label'])
        elif e['end']['label'] != w and e['end']['@id'].startswith('/c/en/'):
            rel_nodes.add(e['end']['label'])

    if 'view' in obj.keys():
        return obj['view'].get('nextPage', None)
    return None


def get_related(vocab_name, outname):
    print("sleeping for 1hour so starting fresh with request limits")
    time.sleep(60*60)

    # load completed:
    # done_words = set()
    # done_f = open(os.path.join(CORPORA_PATH, outname), 'r', encoding='utf-8')
    # done_lines = done_f.readlines()
    # for l in done_lines[1:]:
    #     done_words.add(l.strip().split(',')[0])

    print("starting requests")

    outf = open(os.path.join(CORPORA_PATH, outname), 'w', encoding='utf-8')
    outf.write('word,related_lst\n')
    # outf = open(os.path.join(CORPORA_PATH, outname), 'a', encoding='utf-8')

    f = open(os.path.join(CORPORA_PATH, vocab_name), 'r', encoding='utf-8')
    lines = f.readlines()
    print("need {} words".format(len(lines)))
    start_time = time.time()
    i = 0
    for l in lines:
        start_time = check_and_sleep(start_time, i)

        w = l.strip()
        r, i, start_time = query_conceptnet(w, 'RelatedTo', i, start_time)
        outf.write('{},{}\n'.format(w, json.dumps(list(r))))

def load_related():
    word2rel = dict()
    for rf in os.listdir(REL_DIR):
        if rf.startswith('.') or rf.startswith('need'): continue
        print(rf)
        f = open(os.path.join(REL_DIR, rf), 'r')
        lines = f.readlines()
        for l in lines[1:]:
            try:
                temp = l.strip().split(',', 1)
                w = temp[0].lower()
                rel_lst = set(json.loads(temp[1])) - {w}
            except:
                embed()
            word2rel[w] = word2rel.get(w, set()) | rel_lst
    return word2rel



if __name__ == '__main__':
    if sys.argv[1] == '1':
        get_related('need_related_alldefs_CN.txt', 'related-alldefs_CN.txt')
        # get_related('cwn-gi-emolex-dal-vocab-CLEAN.txt', 'related-cwn-gi-emolex-dal-vocab-v3.txt')
        # get_related('need-related.txt', 'related-cwn-gi-emolex-dal-vocab-v2.txt')
        #get_related('cwn-gi-emolex-dal-vocab.txt', 'related-cwn-gi-emolex-dal-vocab.txt')
