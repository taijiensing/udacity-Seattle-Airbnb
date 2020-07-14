# Helper Functions

import pandas as pd
import numpy as np

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer,fbeta_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

def convert_to_price(x):
    if type(x)==str:
        x = x.replace('$','')
        x = x.replace(',','')
        x = float(x)
    return x

def convert_to_percent(x):
    if type(x)==str:
        x = x.replace('%','')
        x = float(x)
    return x

def convert_to_binary(x):
    if x == 'f':
        x = x.replace('f','0')
    elif x == 't':
        x = x.replace('t','1')
    else:
        x = '0'
    return int(x)

def days_between(d1, d2):
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2-d1).dt.days)

def count_amenities(x):
    amenities_list = x.replace('{','').replace('}','').replace('"','').replace(' ','_').replace(',',' ')
    amenities_list = amenities_list.split()
    return amenities_list

def clean_data(df):
    '''
    INPUT
    df - raw dataframe
    
    OUTPUT
    Cleaned dataframe
    '''
    
    data_df = df.copy()
    
    # Remove '%' character
    data_df['host_response_rate'] = data_df['host_response_rate'].map(lambda x: convert_to_percent(x))
    data_df['host_acceptance_rate'] = data_df['host_acceptance_rate'].map(lambda x: convert_to_percent(x))    
    
    # Remove '$', ',' characters
    data_df['price'] = data_df['price'].map(lambda x: convert_to_price(x))
    data_df['extra_people'] = data_df['extra_people'].map(lambda x: convert_to_price(x))
    data_df['security_deposit'] = data_df['security_deposit'].map(lambda x: convert_to_price(x))
    data_df['cleaning_fee'] = data_df['cleaning_fee'].map(lambda x: convert_to_price(x))
    
    # Convert thumbnail_url to binary (NaN=0 , else=1)
    data_df['has_thumbnail_url'] = np.where(data_df['thumbnail_url'].isnull(), 0,1)
    data_df.drop('thumbnail_url', axis=1, inplace=True)
    
    # Convert t=1 and f,NaN=0
    data_df['host_is_superhost'] = data_df['host_is_superhost'].map(lambda x: convert_to_binary(x))    
    data_df['host_has_profile_pic'] = data_df['host_has_profile_pic'].map(lambda x: convert_to_binary(x))    
    data_df['host_identity_verified'] = data_df['host_identity_verified'].map(lambda x: convert_to_binary(x))    
    data_df['instant_bookable'] = data_df['instant_bookable'].map(lambda x: convert_to_binary(x))   
    data_df['cancellation_policy'] = data_df['cancellation_policy'].map(lambda x: convert_to_binary(x))   

    # Impute NaN value to 0
    data_df['security_deposit'].fillna(0, inplace = True)
    data_df['cleaning_fee'].fillna(0, inplace = True)    

    # Calculate host's age in days
    data_df['host_since'] = pd.to_datetime(data_df['host_since'], format="%Y-%m-%d")
    data_df['host_since'] = data_df[['host_since']].apply(lambda x: days_between(x, '2016-01-04'))
   
    # Count number of Amenities
    data_df['array_amenities'] = data_df['amenities'].map(lambda x: count_amenities(x))
    data_df['total_amenities'] = data_df['amenities'].map(lambda x: len(count_amenities(x)))

    # Drop amenities
    data_df.drop('amenities', axis=1, inplace=True)
    data_df.drop('array_amenities', axis=1, inplace=True)
    
    return data_df


def process_features(feature_df):
    '''
    Impute Missing Values, Dummy Encoding for Categorical Variables, and Feature Scaling
    '''

    # Categorical variables
    cat_feat = list(feature_df.select_dtypes(include = ['object']).columns)    
    
    # Encode Categorical Variables
    feature_df = pd.get_dummies(feature_df, columns= cat_feat)    
    
    # Impute Missing Values
    fill_NaN           = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputed_df         = pd.DataFrame(fill_NaN.fit_transform(feature_df))
    imputed_df.columns = feature_df.columns
    imputed_df.index   = feature_df.index    
    
     # Feature scaling
    num_feat            = list(imputed_df.select_dtypes(include = ['int64','float64']).columns)
    scaler              = MinMaxScaler()
    scaled_df           = imputed_df.copy()
    scaled_df[num_feat] = scaler.fit_transform(imputed_df[num_feat])   
    
    # Replace certain characters in feature names
    scaled_df.columns = scaled_df.columns.str.replace(' ', '_')
    scaled_df.columns = scaled_df.columns.str.replace('&', 'and')
    scaled_df.columns = scaled_df.columns.str.replace('/', '_')
    scaled_df.columns = scaled_df.columns.str.replace('-', '_')
    
    return scaled_df
    
def boost_classifier(clf, parameters, feature_df, labels):

    # Split 'features' and 'label' into train/testi sets
    X_train, X_test, y_train, y_test = train_test_split(feature_df, labels, test_size = 0.2, random_state = 0)

    # Make an fbeta_score scoring object using make_scorer()
    scorer = make_scorer(fbeta_score, beta=0.5)

    # Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
    grid_obj = GridSearchCV(clf, parameters, scorer, n_jobs=-1)

    # Fit the grid search object to the training data and find the optimal parameters using fit()
    grid_fit = grid_obj.fit(X_train,y_train)

    # Get the estimator
    best_clf = grid_fit.best_estimator_

    return best_clf, X_train, X_test, y_train, y_test    
    
    
def prediction_scores(clf, X_train, X_test, y_train, y_test):

    # Model prediction
    test_preds = (clf.fit(X_train, y_train)).predict(X_test)
    train_preds = (clf.fit(X_train, y_train)).predict(X_train)
    
    # Model accuracy
    test_accuracy = accuracy_score(y_test, test_preds)
    train_accuracy = accuracy_score(y_train, train_preds)
    
    return test_accuracy, train_accuracy   
    
def print_scores(test_accuracy, train_accuracy):

    print("Accuracy score on testing data: {:.4f}".format(test_accuracy))
    print("Accuracy score on training data: {:.4f}".format(train_accuracy))