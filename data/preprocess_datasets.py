from data.netflix.preprocess_netflix import preprocess_netflix
from data.movie_lens.preprocess_movie_lens_1m import preprocess_movie_lens
from data.crypto_pairs.preprocess_crypto_pairs import preprocess_crypto_pairs


if __name__ == "__main__":
    preprocess_netflix()
    preprocess_movie_lens()
    preprocess_crypto_pairs()
