import pandas as pd


def load_data(
        ratings_path='data/ratings.jsonl',
        content_path='data/content.jsonl',
        targets_path='data/targets.csv'
    ):
    ratings = pd.read_json(ratings_path, lines=True)
    content = pd.read_json(content_path, lines=True)
    targets = pd.read_csv(targets_path)
    return ratings, content, targets

def preprocessing_content_data(content):
    content['Genre'] = content['Genre'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
    content['Director'] = content['Director'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
    content['Actors'] = content['Actors'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
    content['combined_features'] = content['Genre'].apply(lambda x: ' '.join(x)) + ' ' + \
                                    content['Director'].apply(lambda x: ' '.join(x)) + ' ' + \
                                    content['Actors'].apply(lambda x: ' '.join(x)) + ' ' + \
                                    content['Plot'].fillna('')
    return content