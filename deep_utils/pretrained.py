from pathlib import Path
import pickle


data_root = Path("/home/zf/projects/data")


def get_glove_60B50d_word_embedding():
    return pickle.load(open(data_root / "glove_word_embeddings" / "glove.6B.50d.pkl", "rb"))


if __name__ == '__main__':
    def test_glove():
        emb_dict = get_glove_60B50d_word_embedding()
        print(emb_dict['love'])

    test_glove()