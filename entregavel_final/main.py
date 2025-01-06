"""Recommendation system - assessment activity 2
Registration: 2024661577, 2021037929
Name: Matheus Borges Faria, Lucas de Oliveira Ferreira

To run the code, use the following python version:
conda create --name tp2 python=3.9.18
pip3 install -r requirements.txt
python3 main.py ../data/ratings.jsonl ../data/content.jsonl ../data/targets.csv
"""

import sys
import pandas as pd
import numpy as np
from os import cpu_count
from lightfm.data import Dataset
from lightfm import LightFM
from sklearn.preprocessing import MinMaxScaler

def load_data(ratings_path, content_path, targets_path):
    """Load data from files."""
    ratings = pd.read_json(ratings_path, lines=True)
    content = pd.read_json(content_path, lines=True)
    targets = pd.read_csv(targets_path)
    return ratings, content, targets


def clean_content_data(content):
    """Clean content data."""
    # Drops unnecessary columns
    columns_to_drop = ['Poster', 'Website', 'Response', 'Episode', 'seriesID', 'Season']
    content = content.drop(columns=columns_to_drop).copy()

    # Standardizes NA values
    nan = content.totalSeasons.unique()[0]
    dict_transform_to_na = {
        "Rated":['N/A', 'Not Rated', 'Unrated', 'UNRATED', 'NOT RATED'],
        "all": [nan, 'N/A', 'None', np.nan],
    }

    for na_value in dict_transform_to_na["all"]:
        content = content.replace(na_value, None)

    for na_value in dict_transform_to_na["Rated"]:
        content['Rated'] = content['Rated'].replace(na_value, None)    
    
    # Process and structure the 'Ratings' column
    InternetMovieDatabase_list = []
    Metacritic_list = []
    RottenTomatoes_list = []
    for rating_list in content.Ratings:
        InternetMovieDatabase_list.append(None)
        Metacritic_list.append(None)
        RottenTomatoes_list.append(None)
        for rating_dict in rating_list:
            if rating_dict['Source'] == 'Internet Movie Database':
                InternetMovieDatabase_list[-1] = rating_dict['Value']
            elif rating_dict['Source'] == 'Metacritic':
                Metacritic_list[-1] = rating_dict['Value']
            elif rating_dict['Source'] == 'Rotten Tomatoes':
                RottenTomatoes_list[-1] = rating_dict['Value']
    
    content['Internet Movie Database'] = InternetMovieDatabase_list
    content['Metacritic'] = Metacritic_list
    content['Rotten Tomatoes'] = RottenTomatoes_list
    content.drop(columns=['Ratings'], inplace=True)

    # Standardizes with simple .str.replace() for multiple columns
    content['Year'] = content['Year'].str.replace('–', '')
    
    content['Language'] = content['Language'].str.replace('None, ', '')
    content['Language'] = content['Language'].str.replace(', None', '')
    
    content["Rotten Tomatoes"] = content["Rotten Tomatoes"].str.replace('%', "", regex=True)
    
    content["Metacritic"] = content["Metacritic"].str.replace('/100', "", regex=True)
    
    content["imdbVotes"] = content["imdbVotes"].str.replace(',', "", regex=True)
    
    # Get 'imdbVotes' ranking
    content["Internet Movie Database"] = (
        content["Internet Movie Database"].str.split("/")
        .apply(lambda x: x[0] if x is not None else None)
    )

    # Prepare to convert 'BoxOffice' to integer
    content["BoxOffice"] = content["BoxOffice"].replace(',', "", regex=True)
    content["BoxOffice"] = content["BoxOffice"].apply(
        lambda x: x[1:] if x is not None else None
    )  # Change '$' string to "", replace method didn't work

    # Convert 'Runtime' to integer witch represent minutes
    content["Runtime"] = content["Runtime"].replace(' min', "", regex=True)
    mask = content.Runtime.str.contains('h', na=False)
    content.loc[mask, "Runtime"] = content.loc[mask, "Runtime"].str.split("h").apply(
        lambda x: str(int(x[0]) * 60 + int(x[1]))
    )

    # Convert columns to float
    columns_to_float = ['Runtime', 'Internet Movie Database', 'Metacritic',
                    'Rotten Tomatoes', "imdbVotes", "BoxOffice",
                    "totalSeasons", "imdbRating", "Metascore"]
    content[columns_to_float] = content[columns_to_float].astype(float)

    return content


def content_fill_na(content):
    """Fill NA values in content data."""
    numeric_columns = [
        'Runtime', 'Internet Movie Database', 'Metacritic',
        'Rotten Tomatoes', "imdbVotes", 'BoxOffice',
        'totalSeasons', 'imdbRating', 'Metascore'
    ]
    categoric_columns = ["Rated", "Director", "Production"]
    content[numeric_columns] = content[numeric_columns].fillna(-1)
    content[categoric_columns] = content[categoric_columns].fillna("unknown")
    return content


class LightfmPipeline:
    """LightFM pipeline."""
    model = None
    dataset = None
    item_features_matrix = None
    items_interactions = None
    
    def training_pipeline(self, ratings, content, targets):
        """LightFM training pipeline."""
        # Fit dataset, it create the user, item and item's features mappings
        all_item_ids = list(set(ratings['ItemId'].unique()).union(content['ItemId'].unique()))
        all_item_ids.sort()
        content = content.sort_values("ItemId", ignore_index=True)

        item_features = set()
        for _, row in content.drop(columns=["ItemId"]).iterrows():
            item_features.update(row.values)

        self.dataset = Dataset()
        self.dataset.fit(users=ratings['UserId'].unique(),
                items=list(all_item_ids),
                item_features=item_features)
        
        # Compute item_features_matrix
        features_columns = content.columns.to_list()
        features_columns.pop(0)
        item_features_data = []
        for index, row in content.iterrows():
            item_features_data.append((row["ItemId"], row[features_columns].values.tolist()))
        
        self.item_features_matrix = self.dataset.build_item_features(item_features_data, normalize=True)

        # Build items interactions
        (items_interactions, _) = self.dataset.build_interactions(
            [(row['UserId'], row['ItemId'], row['Rating']) for _, row in ratings.iterrows()]
        )
        self.items_interactions = items_interactions

        # Train LightFM model
        self.model = LightFM(loss='warp', random_state=12012001)
        self.model.fit(self.items_interactions, item_features=self.item_features_matrix,
                epochs=100, num_threads=cpu_count(), verbose=False)
    
    def predict_pipeline(self, targets):
        """LightFM prediction pipeline."""
        user_id_map, user_feature_map, item_id_map, item_feature_map = self.dataset.mapping()
        target_prediction = targets.copy()
        target_prediction["RatingLightFM"] = self.model.predict(
            targets.UserId.map(user_id_map.get).values,
            targets.ItemId.map(item_id_map.get).values,
            item_features=self.item_features_matrix,
            num_threads=cpu_count()
        )
        target_prediction = target_prediction.sort_values(["UserId", "RatingLightFM"], ascending=[True, False])
        return target_prediction

def ensemble_rating(target_prediction, content):
    """Ensample pipeline."""
    new_target_prediction = target_prediction.merge(content, on="ItemId")
    columns = ["imdbRating", "BoxOffice", "Metascore", "RatingLightFM", "imdbVotes"]
    new_target_prediction[columns] = MinMaxScaler().fit_transform(new_target_prediction[columns])
    
    new_target_prediction['Score'] = (
        0.501 * new_target_prediction["imdbRating"]
        + 2.2 * new_target_prediction["BoxOffice"]
        + 0.2 * new_target_prediction["Metascore"]
        + 0.2 * new_target_prediction["RatingLightFM"]
        + 0.1 * new_target_prediction["imdbVotes"]
    )
    new_target_prediction['Score'] = new_target_prediction['Score'].fillna(
        0.501 * new_target_prediction["imdbRating"]
        + 2.2 * new_target_prediction["BoxOffice"]
        + 0.25 * new_target_prediction["Metascore"]
        + 0.25 * new_target_prediction["RatingLightFM"]
    )
    new_target_prediction = new_target_prediction[['UserId', 'ItemId', 'Score']]
    new_target_prediction = new_target_prediction.sort_values(
        ["UserId", "Score"],ascending=[True, False], ignore_index=True
    )
    return new_target_prediction


def final_prediction_output(prediction_dataframe):
    """
    Print predictions for user-item pairs as csv format.

    Parameters
    ----------
    prediction_dataframe : pandas.DataFrame
        The dataframe containing the predictions for user-item pairs.
        Must have columns UserId, ItemId and Rating.
    """
    print("UserId,ItemId")
    for index, row in prediction_dataframe.iterrows():
        print(f"{row['UserId']},{row['ItemId']}")
    return

    
def main():
    """Main function."""
    if len(sys.argv) != 4:
        print("Expected: python3 main.py <ratings.jsonl> <content.jsonl> <targets.csv>")
        sys.exit(1)

    ratings_path = sys.argv[1]
    content_path = sys.argv[2]
    targets_path = sys.argv[3]
    
    ratings, content, targets = load_data(ratings_path, content_path, targets_path)
    content = clean_content_data(content)
    content = content_fill_na(content)

    lightfm_pipeline = LightfmPipeline()
    lightfm_pipeline.training_pipeline(ratings, content, targets)
    
    target_prediction = lightfm_pipeline.predict_pipeline(targets)
    
    target_prediction = ensemble_rating(target_prediction, content)
    
    final_prediction_output(target_prediction)
    
if __name__ == "__main__":
    main()