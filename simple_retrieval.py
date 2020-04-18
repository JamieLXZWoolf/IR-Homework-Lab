import os

from pathlib import Path
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

def output_vector(v):
    s = ""
    for i in v:
        s += str(i) + " "
    return s

class InvertedIndex(object):
    def __init__(self, docs):
        self._doc_name = os.path.splitext(docs)[0]
        self._index, self._docs_count, self._id2doc =  build_inverted_index(docs)
        self._term2id, self._id2term = term2id(self._index)
        self._df_str, self._df_id = self._get_document_frequency()
        self._idf_dict = self._calculate_idf()
        self._term_document_matrix = self._get_t_d_matrix()
        self._print_t_d_matrix()
        self._print_inverted_index()
        self._docs_lm_matrix, self._collection_lm_vec  = self._get_docs_and_collection_lan_models()

    def _print_inverted_index(self):
        ii_path = "outputs/{}".format(self._doc_name)
        Path(ii_path).mkdir(parents=True, exist_ok=True)
        with open("{}/inverted_index.txt".format(ii_path), "w", encoding="utf-8") as writer:
            for term, postings in sorted(self._index.items()):
                writer.write("{}:\t{}\t{}\n".format(term, self._term2id[term],str(postings)))

    def _get_t_d_matrix(self):
        vectors = []
        for i in range(self._docs_count):
            vectors.append(np.zeros((len(self._index)),dtype=np.int))
        for term, postings in self._index.items():
            id = self._term2id[term]
            for d, o in postings.items():
                vectors[d][id] += len(o)
        return np.array(vectors)

    def _print_t_d_matrix(self):
        ii_path = "outputs/{}".format(self._doc_name)
        Path(ii_path).mkdir(parents=True, exist_ok=True)
        with open("{}/term_document_matrix.txt".format(ii_path), "w", encoding="utf-8") as writer:
            for d in self._term_document_matrix:
                writer.write("{}\n".format(output_vector(d)))

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
        term_id_wf = {}
        tfs = {}
        for term, posting in query_postings.items():
            if term not in self._term2id.keys():
                out_of_dict_words.add(term)
                continue
            tf = len(posting)
            tfs[self._term2id[term]] = tf
        max_tf = max(tfs.values())
        for i, tf in tfs.items():
            wf = 0.5 + 0.5*tf/max_tf
            term_id_wf[i] = wf
            term_id_wf_idfs[i] = wf * self._idf_dict[i]
            vec[i] += wf
        return term_id_wf_idfs, out_of_dict_words, vec, tfs, term_id_wf

    def _query_lm_and_vec(self, query):
        query_postings = get_postings(query)
        tfs = np.zeros((len(self._index)),dtype=np.int)
        for term, posting in query_postings.items():
            if term not in self._term2id.keys():
                continue
            tf = len(posting)
            tfs[self._term2id[term]] = tf
        query_len = sum(tfs)
        lm = []
        for f in tfs:
            lm.append(1/3 + 2/3 * f/query_len)
        output_dir = "outputs/{}/{}".format(self._doc_name, query)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open("{}/query_lan_model.txt".format(output_dir), "w", encoding="utf-8") as writer:
            writer.write(output_vector(lm))
        return tfs, lm

    def _get_docs_and_collection_lan_models(self):
        lm_matrix = []
        for i in range(self._docs_count):
            lm_vec = []
            for tf in self._term_document_matrix[i]:
                lm_vec.append(1/3 + 2/3 * tf/len(self._term_document_matrix[i]))
            lm_matrix.append(lm_vec)
        c_lm_vec = []
        for id in range(len(self._df_id)):
            c_lm_vec.append(1/3 + 2/3 * (self._df_id[id])/self._docs_count)

        output_dir = "outputs/{}".format(self._doc_name)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open("{}/doc_lan_models.txt".format(output_dir), "w", encoding="utf-8") as writer:
            for d in lm_matrix:
                writer.write("{}\n".format(output_vector(d)))
        with open("{}/col_lan_model.txt".format(output_dir), "w", encoding="utf-8") as writer:
            writer.write(output_vector(c_lm_vec))
        return lm_matrix, c_lm_vec

    def vector_query_and_rank(self, query):
        docs_scores = []
        term_id_wf_idfs, out_of_dict_words, vec, q_tfs, q_wfs = self._query_wf_idf_and_vec(query)
        doc_tfs = {}
        doc_wfs = {}
        doc_norms = {}

        for doc_id in range(self._docs_count):
            doc_tfs[doc_id] = {}
            doc_wfs[doc_id] = {}
            doc_norms[doc_id] = {}
            doc_vec = self._term_document_matrix[doc_id]
            m = max(doc_vec)
            augmented_doc_vec = (tf/m for tf in doc_vec)
            cos_norm_denominator = norm_denominator(augmented_doc_vec)
            score = 0.0
            max_freq = max(doc_vec)
            for id, wf_idf in term_id_wf_idfs.items():
                tf = doc_vec[id]
                doc_tfs[doc_id][id] = tf
                wf = self._idf_dict[id] * tf / max_freq
                doc_wfs[doc_id][id] = wf
                di = wf/cos_norm_denominator
                doc_norms[doc_id][id] = di
                score += di * wf_idf
            docs_scores.append(score)
        ranks = np.argsort(docs_scores)[::-1]

        # output calculation process
        output_dir = "outputs/{}/{}".format(self._doc_name, query)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open("{}/query_calculation.txt".format(output_dir), "w", encoding="utf-8") as writer:
            terms = {self._id2term[i] for i in q_tfs.keys()}
            for t in sorted(terms):
                id = self._term2id[t]
                writer.write("{}:\t{}\t{}\t{}\t{}\t{}\n".format(t, q_tfs[id],q_wfs[id], self._df_id[id],
                                                                self._idf_dict[id], term_id_wf_idfs[id]))
        with open("{}/docs_calculation.txt".format(output_dir), "w", encoding="utf-8") as writer:
            terms = {self._id2term[i] for i in q_tfs.keys()}
            for t in sorted(terms):
                writer.write(t + "\n\n")
                id = self._term2id[t]
                for d in range(self._docs_count):
                    writer.write("\t{}:\t{}\t{}\t{}\n".format(d, doc_tfs[d][id], doc_wfs[d][id], doc_norms[d][id]))
                writer.write("\n\n")
        with open("{}/query_vector.txt".format(output_dir), "w", encoding="utf-8") as writer:
            writer.write(output_vector(vec))
        with open("{}/vector_space_rank_and_scores.txt".format(output_dir), "w", encoding="utf-8") as writer:
            for doc_id in ranks:
                writer.write(self._id2doc[doc_id]+ "\t" + str(docs_scores[doc_id]) + "\n")

    def lm_query_and_rank(self, query):
        query_vec, _ = self._query_lm_and_vec(query)
        valid_terms = []
        for i in range(len(query_vec)):
            if query_vec[i] != 0:
                valid_terms.append(i)
        docs_scores = []
        output_dir = "outputs/{}/{}".format(self._doc_name, query)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open("{}/lm_calculation.txt".format(output_dir), "w", encoding="utf-8") as pw:
            for d in range(self._docs_count):
                pw.write("doc {}: \n\n".format(d))
                score = 1
                for term, t in sorted(((self._id2term[t], t) for t in valid_terms)):
                    pw.write("\t{}".format(term))
                    pt_mc = self._collection_lm_vec[t]
                    pw.write("\t{}".format(pt_mc))
                    pt_dc = self._docs_lm_matrix[d][t]
                    pw.write("\t{}".format(pt_dc))
                    pt_cd = 1/2 * (pt_mc + pt_dc)
                    pw.write("\t{}\n".format(pt_cd))
                    score *= pt_cd
                docs_scores.append(score)
        ranks = np.argsort(docs_scores)[::-1]
        with open("{}/lm_rank_and_scores.txt".format(output_dir), "w", encoding="utf-8") as rw:
            for doc_id in ranks:
                rw.write(self._id2doc[doc_id]+ "\t" + str(docs_scores[doc_id]) + "\n")


if __name__ == '__main__':
    ii = InvertedIndex("new_docs.txt")
    ii.vector_query_and_rank("Computing semantic relatedness using wikipedia-based explicit semantic analysis")
    ii.lm_query_and_rank("Computing semantic relatedness using wikipedia-based explicit semantic analysis")