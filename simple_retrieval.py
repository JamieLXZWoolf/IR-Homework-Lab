from nltk.corpus import stopwords
import numpy as np
from math import log, sqrt
_STOP_WORDS = set(stopwords.words('english'))
_WORD_MIN_LENGTH = 3

def word_split(text):
    word_list = []
    wcurrent = []
    windex = None
    for i, c in enumerate(text):
        if c.isalnum():
            wcurrent.append(c)
            windex = i
        elif wcurrent:
            word = u''.join(wcurrent)
            word_list.append((windex - len(word) + 1, word))
            wcurrent = []
    if wcurrent:
        word = u''.join(wcurrent)
        word_list.append((windex - len(word) + 1, word))
    return word_list

def words_cleanup(words):
    cleaned_words = []
    for index, word in words:
        if len(word) < _WORD_MIN_LENGTH or word in _STOP_WORDS:
            continue
        cleaned_words.append((index, word))
    return cleaned_words

def words_normalize(words):
    normalized_words = []
    for index, word in words:
        wnormalized = word.lower()
        normalized_words.append((index, wnormalized))
    return normalized_words

def word_index(text):
    words = word_split(text)
    words = words_normalize(words)
    words = words_cleanup(words)
    return words

def get_postings(text):
    posting = {}
    for index, word in word_index(text):
        locations = posting.setdefault(word, [])
        locations.append(index)
    return posting

def inverted_index_add(inverted, doc_id, doc_index):
    for word, locations in doc_index.items():
        indices = inverted.setdefault(word, {})
        indices[doc_id] = locations
    return inverted

# def search(inverted, query):
#     words = [word for _, word in word_index(query) if word in inverted]
#     results = [set(inverted[word].keys()) for word in words]
#     return reduce(lambda x, y: x & y, results) if results else []

def build_inverted_index(docs):
    inverted_index = {}
    docs_count = 0
    id2doc = {}
    with open(docs, "r", encoding="utf-8") as reader:
        for i, line in enumerate(reader.readlines()):
            doc_text = line.strip()
            inverted_index_add(inverted_index, i, get_postings(doc_text))
            docs_count+=1
            id2doc[i] = line.strip()
    return inverted_index, docs_count, id2doc

def term2id(ii):
    t2id = {}
    id2t = {}
    for term, postings in ii.items():
        id = len(t2id)
        t2id[term] = id
        id2t[id] = term
    return t2id, id2t

def norm_denominator(vec):
    sum = 0
    for p in vec:
        sum += p*p
    return sqrt(sum)

class InvertedIndex(object):
    def __init__(self, docs):
        self._index, self._docs_count, self._id2doc =  build_inverted_index(docs)
        print(self._index)
        self._term2id, self._id2term = term2id(self._index)
        self._df_str, self._df_id = self._get_document_frequency()
        self._idf_dict = self._calculate_idf()
        self._term_document_matrix = self._get_t_d_matrix()
        print(self._term_document_matrix, self._id2doc)

    def _get_t_d_matrix(self):
        vectors = []
        for i in range(self._docs_count):
            vectors.append(np.zeros((len(self._index)),dtype=np.int))
        for term, postings in self._index.items():
            id = self._term2id[term]
            for d, o in postings.items():
                vectors[d][id] += len(o)
        return np.array(vectors)

    def _get_document_frequency(self):
        freq_dict_by_str = {}
        freq_dict_by_id = {}
        for t, p in self._index.items():
            freq_dict_by_str[t] = freq_dict_by_id[self._term2id[t]] = len(p)
        return freq_dict_by_str, freq_dict_by_id

    def _calculate_idf(self):
        idf_dict = {}
        for id in self._term2id.values():
            idf_dict[id] = log(self._docs_count/self._df_id[id])
        return idf_dict

    def _query_wf_idf_and_vec(self, query):
        query_postings = get_postings(query)
        term_id_wf_idfs = {}
        out_of_dict_words = set()
        vec = np.zeros((len(self._index)),dtype=np.int)
        for term, posting in query_postings.items():
            wf = len(posting)
            if term not in self._term2id.keys():
                out_of_dict_words.add(term)
                continue
            term_id_wf_idfs[self._term2id[term]] = wf * self._idf_dict[self._term2id[term]]
            vec[self._term2id[term]] += wf
        return term_id_wf_idfs, out_of_dict_words, vec

    def query_and_rank(self, query):
        docs_scores = []
        term_id_wf_idfs, out_of_dict_words, vec = self._query_wf_idf_and_vec(query)
        for doc_id in range(self._docs_count):
            doc_vec = self._term_document_matrix[doc_id]
            cos_norm_denominator = norm_denominator(doc_vec)
            score = 0.0
            max_freq = max(doc_vec)
            for id, wf_idf in term_id_wf_idfs.items():
                tf = doc_vec[id]
                wf = (tf/max_freq) * log(self._docs_count/self._df_id[id])
                di = wf/cos_norm_denominator
                score += di * wf_idf
            docs_scores.append(score)
        ranks = np.argsort(docs_scores)[::-1]
        print(vec)
        for doc_id in ranks:
            print(self._id2doc[doc_id], docs_scores[doc_id])





if __name__ == '__main__':
    ii = InvertedIndex("docs.txt")
    ii.query_and_rank("Effective Online Search Engine Knowledge Graph Data Fusion")