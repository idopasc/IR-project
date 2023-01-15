import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import sys
from collections import defaultdict, Counter
import itertools
from operator import itemgetter
import nltk
from pathlib import Path
import pickle
import hashlib
nltk.download('stopwords')
from contextlib import closing
import heapq


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


# external function used to read pickle files (like the index file)
def read_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# holds a dict of {doc_id:title} of all the docs in the corpus
doc_id_to_title = read_pkl("./titles.pkl")

# holds all the terms that exist in the body with their idf score {term:idf score}
BIDF = read_pkl("./BIDF.pkl")

# holds all the terms that exist in the title with their idf score
TIDF = read_pkl("./TIDF.pkl")

# holds all the bodies length of all our wiki corpus {doc id: body length}
BDL = read_pkl("./BDL.pkl")

# holds all the titles length of all our wiki corpus {doc id: title length}
TDL = read_pkl("./TDL.pkl")

# holds doc id's bodies and their norm size {doc id: body norm}
B_doc_prepared_tfidf_norm = read_pkl("./BTFIDF_norm.pkl")

# holds doc id's titles and their norm size {doc id: title norm}
T_doc_prepared_tfidf_norm = read_pkl("./TTFIDF_norm.pkl")

#Pandas Dataframe of doc_id's and their page rank score (doc id,page rank score)
page_rank = pd.read_csv("./page_rank.csv", header=None)
page_rank.columns = ["doc_id", "rank"]

# doc_id's and their page view score (doc id,page view score)
page_views = read_pkl("./page_views.pkl")

TUPLE_SIZE = 6  # We're going to pack the doc_id and tf values in this
# many bytes.
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer

BLOCK_SIZE = 1999998


class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """

    def __init__(self, base_dir, name):
        self._base_dir = Path(base_dir)
        self._name = name
        self._file_gen = (open(self._base_dir / f'{name}_{i:04}.bin', 'wb')
                          for i in itertools.count())
        self._f = next(self._file_gen)

    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            # if the current file is full, close and open a new one.
            if remaining == 0:
                self._f.close()
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            locs.append((self._f.name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()


class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """

    def __init__(self):
        self._open_files = {}

    def read(self, locs, n_bytes, base_dir="."):
        b = []
        for f_name, offset in locs:
            if f_name not in self._open_files:
                self._open_files[f_name] = open(base_dir + "/" + f_name, 'rb')
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


# We're going to update and calculate this after each document. This will be usefull for the calculation of AVGDL (utilized in BM25)
class InvertedIndex:
    def __init__(self, docs={}):
        """ Initializes the inverted index and add documents to it (if provided).
        Parameters:
        -----------
          docs: dict mapping doc_id to list of tokens
        """
        # stores document frequency per term
        self.df = Counter()
        # stores total frequency per term
        self.term_total = Counter()
        # stores posting list per term while building the index (internally),
        # otherwise too big to store in memory.
        self._posting_list = defaultdict(list)
        # mapping a term to posting file locations, which is a list of
        # (file_name, offset) pairs. Since posting lists are big we are going to
        # write them to disk and just save their location in this list. We are
        # using the MultiFileWriter helper class to write fixed-size files and store
        # for each term/posting list its list of locations. The offset represents
        # the number of bytes from the beginning of the file where the posting list
        # starts.
        self.posting_locs = defaultdict(list)
        self.DL = {}
        self.base_dir = "."
        if type(docs) == dict:
            for doc_id, tokens in docs.items():
                self.add_doc(doc_id, tokens)
        else:
            for doc_id, tokens in docs:
                self.add_doc(doc_id, tokens)

    def add_doc(self, doc_id, tokens):
        """ Adds a document to the index with a given `doc_id` and tokens. It counts
            the tf of tokens, then update the index (in memory, no storage
            side-effects).
        """
        self.DL[doc_id] = self.DL.get(doc_id, 0) + (len(tokens))
        w2cnt = Counter(tokens)
        self.term_total.update(w2cnt)
        # max_value = max(w2cnt.items(), key=operator.itemgetter(1))[1]
        # frequencies = {key: value/max_value for key, value in frequencies.items()}
        for w, cnt in w2cnt.items():
            self.df[w] = self.df.get(w, 0) + 1
            self._posting_list[w].append((doc_id, cnt))

    def write_index(self, base_dir, name):
        """ Write the in-memory index to disk. Results in the file:
            (1) `name`.pkl containing the global term stats (e.g. df).
        """
        self._write_globals(base_dir, name)

    def write(self, base_dir, name):
        """ Write the in-memory index to disk and populate the `posting_locs`
            variables with information about file location and offset of posting
            lists. Results in at least two files:
            (1) posting files `name`XXX.bin containing the posting lists.
            (2) `name`.pkl containing the global term stats (e.g. df).
        """
        #### POSTINGS ####
        self.posting_locs = defaultdict(list)
        with closing(MultiFileWriter(base_dir, name)) as writer:
            # iterate over posting lists in lexicographic order
            for w in sorted(self._posting_list.keys()):
                self._write_a_posting_list(w, writer, sort=True)
        #### GLOBAL DICTIONARIES ####
        self._write_globals(base_dir, name)

    def _write_globals(self, base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)

    def _write_a_posting_list(self, w, writer, sort=False):
        # sort the posting list by doc_id
        pl = self._posting_list[w]
        if sort:
            pl = sorted(pl, key=itemgetter(0))
        # convert to bytes
        b = b''.join([(int(doc_id) << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                      for doc_id, tf in pl])
        # write to file(s)
        locs = writer.write(b)
        # save file locations to index
        self.posting_locs[w].extend(locs)

    def __getstate__(self):
        """ Modify how the object is pickled by removing the internal posting lists
            from the object's state dictionary.
        """
        state = self.__dict__.copy()
        del state['_posting_list']
        return state

    @staticmethod
    def read_index(base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
            index = pickle.load(f)
            index.base_dir = base_dir
            return index

    @staticmethod
    def delete_index(base_dir, name):
        path_globals = Path(base_dir) / f'{name}.pkl'
        path_globals.unlink()
        for p in Path(base_dir).rglob(f'{name}_*.bin'):
            p.unlink()

    def posting_lists_iter(self):
        """ A generator that reads one posting list from disk and yields
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """
        with closing(MultiFileReader()) as reader:
            for w, locs in self.posting_locs.items():
                # read a certain number of bytes into variable b
                b = reader.read(locs, self.df[w] * TUPLE_SIZE)
                posting_list = []
                # convert the bytes read into `b` to a proper posting list.

                for i in range(self.df[w]):
                    doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                    tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                    posting_list.append((doc_id, tf))

                yield w, posting_list

    def write_a_posting_list(b_w_pl):
        ''' Takes a (bucket_id, [(w0, posting_list_0), (w1, posting_list_1), ...])
        and writes it out to disk as files named {bucket_id}_XXX.bin under the
        current directory. Returns a posting locations dictionary that maps each
        word to the list of files and offsets that contain its posting list.
        Parameters:
        -----------
          b_w_pl: tuple
            Containing a bucket id and all (word, posting list) pairs in that bucket
            (bucket_id, [(w0, posting_list_0), (w1, posting_list_1), ...])
        Return:
          posting_locs: dict
            Posting locations for each of the words written out in this bucket.
        '''
        posting_locs = defaultdict(list)
        bucket, list_w_pl = b_w_pl

        with closing(MultiFileWriter('.', bucket)) as writer:
            for w, pl in list_w_pl:
                # convert to bytes
                b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                              for doc_id, tf in pl])
                # write to file(s)
                locs = writer.write(b)
                # save file locations to index
                posting_locs[w].extend(locs)
        return posting_locs

    def merge_indices(self, base_dir, names, output_name):
        """ A function that merges the (partial) indices built from subsets of
            documents, and writes out the merged posting lists.
        Parameters:
        -----------
            base_dir: str
                Directory where partial indices reside.
            names: list of str
                A list of index names to merge.
            output_name: str
                The name of the merged index.
        """
        indices = [InvertedIndex.read_index(base_dir, name) for name in names]
        iters = [idx.posting_lists_iter() for idx in indices]
        self.posting_locs = defaultdict(list)
        #### POSTINGS: merge & write out ####
        writer = MultiFileWriter(base_dir, output_name)
        pointers = [next(iter, ((chr(sys.maxunicode), None))) for iter in iters]
        finished = [False for _ in range(len(iters))]
        while not all(finished):
            min_w = min(pointers, key=lambda t: t[0])[0]
            idxs = [i for i, x in enumerate(pointers) if x[0] == min_w]
            term = pointers[idxs[0]][0]
            lists = [pointers[idx][1] for idx in idxs]
            merged_gen = heapq.merge(*lists, key=itemgetter(0))
            self._posting_list = {term: list(merged_gen)}
            self._write_a_posting_list(term, writer)
            for idx in idxs:
                pointers[idx] = next(iters[idx], ((chr(sys.maxunicode), None)))
                if pointers[idx][0] == chr(sys.maxunicode):
                    finished[idx] = True
        for index in indices:
            self.df.update(index.df)
            self.DL.update(index.DL)
            self.term_total.update(index.term_total)
        #### GLOBAL DICTIONARIES ####
        self._write_globals(base_dir, output_name)

# loading of body term's document frequency
bdf = read_pkl("./body_index/bdf.pkl")

# loading of body posting locs (location) from the shape <bin,offset>
bpl = read_pkl("./body_index/bpl.pkl")

# InvertedIndex instance
body_index = InvertedIndex()
body_index.df = bdf
body_index.posting_locs = bpl

# defining body index location
body_index.base_dir = "./body_index"

# loading of title term's document frequency
tdf = read_pkl("./title_index/tdf.pkl")

# loading of title posting locs (location) from the shape <bin,offset>
tpl = read_pkl("./title_index/tpl.pkl")

# InvertedIndex instance
title_index = InvertedIndex()
title_index.df = tdf
title_index.posting_locs = tpl

# defining body index location
title_index.base_dir = "./title_index"

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']
all_stopwords = english_stopwords.union(corpus_stopwords)

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE)  ## tokenizer from assignment 3
def tokenize(text):
    return [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]



def read_posting_list(inverted, w):
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, inverted.base_dir)  ## we added the base directory of the relevant InvertedIndex
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list



def autocorrect(tok_query, index):
    """
    In order to correct typo and spelling mistakes, we consrturcted an autocorrect function
    based on jaccard distance method with a specific treshold to evaluate similarity between two words
    Args:
        tok_query: list of the query words
        index: "T" = title index, "B" = body index

    Returns:
    if exist a word which is below the relevant threshold, we will take the nearest one,
    if there isn't, it doesn't autocorrect (doesn't switch the original word)
    """
    corpus_terms = [term for term in tok_query if term in index.df]
    unknown_terms = [term for term in tok_query if term not in corpus_terms]
    for term in unknown_terms:
        closest_term = min([(corp_term, nltk.jaccard_distance(set(nltk.ngrams(term, 2)), set(nltk.ngrams(corp_term, 2)))) for corp_term in index.df if term[0] == corp_term[0]], key=lambda dist: dist[1])
        if closest_term[1] < 0.45:
            corpus_terms.append(closest_term[0])
    return corpus_terms



def cossine_sim(index, query, index_type):
    """
    our fast cossine similarity method, which avoids calculating all the calculations that can be done offline
    index_type argument defining if we are preforming the func on the body or the title index
    query is tokenized list of the query words.
    # we handle spelling mistakes already in the calling function to avoid undefined word here (A word that not in the vocabulary)
    Args:
        index: body/title index
        query: tokenized list of the query words
        index_type: "B" = body, "T" = title
    Returns:
        dictionary with all the relevant docs and their tf-idf score
    """
    if index_type == 'B':
        idf = BIDF
        DL = BDL
        doc_norm = B_doc_prepared_tfidf_norm
    elif index_type == 'T':
        idf = TIDF
        DL = TDL
        doc_norm = T_doc_prepared_tfidf_norm
    else:
        raise TypeError

    relevant_docs_by_score = {}
    query_len = len(query)
    query_Counter = Counter(query)
    tf_idf_weights = {}

    # compute tfidf values for each term in the query
    for term, occs in query_Counter.items():
        tf_norma = occs / query_len
        if term in idf:
            prepared_idf = idf[term]
        else:
            prepared_idf = 0
        tf_idf_weights[term] = tf_norma * prepared_idf

    # compute query norm
    query_norm = np.sqrt(sum(v ** 2 for v in tf_idf_weights.values()))

    for word in query_Counter.keys():

        word_posting = read_posting_list(index, word)
        for doc_id, term_freq in word_posting:
            w_ij = (term_freq / DL[doc_id] * idf[word])
            relevant_docs_by_score[doc_id] = relevant_docs_by_score.get(doc_id, 0) + (w_ij * tf_idf_weights[word]) / (
                        doc_norm[doc_id] * query_norm)

    return sorted(list(relevant_docs_by_score.items()), key=lambda x: x[1], reverse=True)





