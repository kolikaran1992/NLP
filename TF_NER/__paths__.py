from pathlib import Path

path_obj = Path(__file__).absolute().parent

PATH_TO_ROOT_DIR = path_obj.as_posix()

PATH_TO_WORD_EMBEDDINGS = path_obj.joinpath('pre_trained_embeddings')

PATH_TO_GRAPHS = path_obj.joinpath('tensorflow_graphs')

PATH_TO_LOGS = path_obj.joinpath('logs')

PATH_TO_HPARAMS_WORD_EMBEDDING_CHAR_LEVEL = path_obj.joinpath('hparams', 'word_embedding_character_level.json')

PATH_TO_HPARAMS_MAIN_NETWORK = path_obj.joinpath('hparams', 'main_network.json')

PATH_TO_SAVED_MODELS = path_obj.joinpath('saved_models')

PATH_TO_DATA = path_obj.joinpath('data')