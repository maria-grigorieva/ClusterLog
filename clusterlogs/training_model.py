from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
from clusterlogs import Tokens, Regex
import sys

def main():
    # df = pd.read_csv('../samples/fts_mess_panda.csv', index_col=0)
    df = pd.read_csv('../test/error_messages.csv', index_col=0)
    df.set_index('pandaid', inplace=True)
    cleaned_messages = Regex(df['exeerrordiag'].values).process()
    tok = Tokens(cleaned_messages)
    tok.process()

    word2vec = Word2Vec(tok.tokenized,
                         size=300,
                         window=7,
                         min_count=1,
                         workers=4,
                         iter=10)

    word2vec.save('../test/word2vec.model')

if __name__ == "__main__":
    sys.exit(main())