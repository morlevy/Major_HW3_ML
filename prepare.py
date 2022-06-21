import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from feature_engine.imputation import EndTailImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def filter_zip(item):
    if pd.isna(item):
        return item

    item = (str(item)).replace(" ", "")
    zip_start_index = len(item) - 5
    return item[zip_start_index:]


def filter_country(item):
    if pd.isna(item):
        return item

    item = (str(item)).replace(" ", "")
    zip_start_index = len(item) - 5
    country_start_index = zip_start_index - 2
    return item[country_start_index:zip_start_index]


def longitude_filter(item):
    if pd.isna(item):
        return item

    items = str(item)
    return items[2:items.find(',') - 1]


def latitude_filter(item):
    if pd.isna(item):
        return item

    items = str(item)
    return items[items.find(',') + 3:-2]


def symptom_cough(item):
    if pd.isna(item): return 0
    if 'cough' in item: return 1
    return 0


def symptom_low_appetite(item):
    if pd.isna(item): return 0
    if 'low_appetite' in item: return 1
    return 0


def symptom_sore_throat(item):
    if pd.isna(item): return 0
    if 'sore_throat' in item: return 1
    return 0


def symptom_shortness_of_breath(item):
    if pd.isna(item): return 0
    if 'shortness_of_breath' in item: return 1
    return 0


def symptom_fever(item):
    if pd.isna(item): return 0
    if 'fever' in item: return 1
    return 0


def day(x):
    if pd.isna(x): return x
    x = str(x)
    return x[:2]


def month(x):
    if pd.isna(x): return x
    x = str(x)
    return x[3:5]


def year(x):
    if pd.isna(x): return x
    x = str(x)
    return x[6:]


class ConvertNumericAndExtractFeatures:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def sex(self):
        self.dataframe = pd.get_dummies(self.dataframe, columns=['sex'], prefix=['sex'])

    def blood_type(self):
        self.dataframe = pd.get_dummies(self.dataframe, columns=['blood_type'], prefix=['blood'])

    def address(self):
        addresses = pd.DataFrame(self.dataframe, columns=['address'])
        country = addresses.applymap(filter_country)
        zip_code = addresses.applymap(filter_zip)

        self.dataframe['country'] = country
        self.dataframe['zip_code'] = zip_code
        del self.dataframe['address']

        dict_countries = (self.dataframe['country'].value_counts()).to_dict()
        i = 1
        for key in dict_countries.keys():
            dict_countries[key] = i
            i = i + 1

        def convert_country_to_int(x):
            if pd.isna(x):
                return x
            return dict_countries[x]

        country_to_int = country.applymap(convert_country_to_int)
        del self.dataframe['country']
        self.dataframe['country'] = country_to_int

    def current_location(self):
        current_location = pd.DataFrame(self.dataframe, columns=['current_location'])
        longitude = current_location.applymap(longitude_filter)
        latitude = current_location.applymap(latitude_filter)

        self.dataframe['longitude'] = pd.to_numeric(longitude['current_location'])
        self.dataframe['latitude'] = pd.to_numeric(latitude['current_location'])
        del self.dataframe['current_location']

    def symptoms(self):
        symptoms = pd.DataFrame(self.dataframe, columns=['symptoms'])
        cough = symptoms.applymap(symptom_cough)
        low_appetite = symptoms.applymap(symptom_low_appetite)
        sore_throat = symptoms.applymap(symptom_sore_throat)
        shortness_of_breath = symptoms.applymap(symptom_shortness_of_breath)
        fever = symptoms.applymap(symptom_fever)

        self.dataframe["symptom_cough"] = cough
        self.dataframe["symptom_low_appetite"] = low_appetite
        self.dataframe["symptom_sore_throat"] = sore_throat
        self.dataframe["symptom_shortness_of_breath"] = shortness_of_breath
        self.dataframe["symptom_fever"] = fever
        del self.dataframe['symptoms']

    def pcr_date(self):
        pcr_date = pd.DataFrame(self.dataframe, columns=['pcr_date'])
        day_col = pcr_date.applymap(day)
        month_col = pcr_date.applymap(month)
        year_col = pcr_date.applymap(year)

        self.dataframe['day'] = day_col
        self.dataframe['month'] = month_col
        self.dataframe['year'] = year_col
        del self.dataframe['pcr_date']

    def convert_to_numeric(self):
        # day, month, year and zip_code are from object type - so we will convert them to int
        self.dataframe['day'] = (
            self.dataframe['day'].fillna(0)
                .astype(int)
                .astype(object)
                .where(self.dataframe['day'].notnull())
        )

        self.dataframe['month'] = (
            self.dataframe['month'].fillna(0)
                .astype(int)
                .astype(object)
                .where(self.dataframe['month'].notnull())
        )

        self.dataframe['year'] = (
            self.dataframe['year'].fillna(0)
                .astype(int)
                .astype(object)
                .where(self.dataframe['year'].notnull())
        )

        self.dataframe['zip_code'] = (
            self.dataframe['zip_code'].fillna(0)
                .astype(int)
                .astype(object)
                .where(self.dataframe['zip_code'].notnull())
        )

    def run_all(self):
        self.sex()
        # self.blood_type()
        # self.address()
        # self.current_location()
        self.symptoms()
        # self.pcr_date()
        # self.convert_to_numeric()

        return self.dataframe.copy()


class Imputation:
    def __init__(self, data, train, random_sample_list, mean_list, median_list, most_frequent_list, end_tail_norm_list):
        self.data = data
        self.train = train
        self.random_sample_list = random_sample_list
        self.mean_list = mean_list
        self.median_list = median_list
        self.most_frequent_list = most_frequent_list
        self.end_tail_norm_list = end_tail_norm_list

    def mean_imputation(self):
        for feature in self.mean_list:
            imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer_mean.fit(self.train[[feature]])
            self.data[feature] = imputer_mean.transform(self.data[[feature]])

    def most_frequent_imputation(self):
        for feature in self.most_frequent_list:
            imputer_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            imputer_mean.fit(self.train[[feature]])
            self.data[feature] = imputer_mean.transform(self.data[[feature]])

    def median_imputation(self):
        for feature in self.median_list:
            imputer_mean = SimpleImputer(missing_values=np.nan, strategy='median')
            imputer_mean.fit(self.train[[feature]])
            self.data[feature] = imputer_mean.transform(self.data[[feature]])

    def convert_dataframe_sex_feature(self, dataframe):
        conditions = [
            (dataframe['sex_M'] == 0) & (dataframe['sex_F'] == 0),
            (dataframe['sex_M'] == 1),
            (dataframe['sex_F'] == 1)
        ]
        values = [None, 'M', 'F']

        dataframe['sex'] = np.select(conditions, values)

    def random_sample_imputation_on_sex_feature(self):
        conditions = [
            (self.data['sex_M'] == 0) & (self.data['sex_F'] == 0),
            (self.data['sex_M'] == 1),
            (self.data['sex_F'] == 1)
        ]
        values = [None, 'M', 'F']
        self.data['sex'] = np.select(conditions, values)

        conditions = [
            (self.train['sex_M'] == 0) & (self.train['sex_F'] == 0),
            (self.train['sex_M'] == 1),
            (self.train['sex_F'] == 1)
        ]
        values = [None, 'M', 'F']
        self.train['sex'] = np.select(conditions, values)

        self.data['sex_imputed'] = self.data['sex'].copy()
        self.train['sex_imputed'] = self.train['sex'].copy()

        random_sample_train = self.train['sex'].dropna().sample(self.train['sex'].isnull().sum(),
                                                                random_state=0)
        random_sample_data = self.data['sex'].dropna().sample(self.data['sex'].isnull().sum(),
                                                              random_state=0)
        random_sample_train.index = self.train[self.train['sex'].isnull()].index
        random_sample_data.index = self.data[self.data['sex'].isnull()].index

        self.train.loc[self.train['sex'].isnull(), 'sex_imputed'] = random_sample_train
        self.data.loc[self.data['sex'].isnull(), 'sex_imputed'] = random_sample_data

        del self.train['sex']
        self.train.rename(columns={'sex_imputed': 'sex'}, inplace=True)

        del self.data['sex']
        self.data.rename(columns={'sex_imputed': 'sex'}, inplace=True)

        del self.data['sex_M']
        del self.data['sex_F']
        self.data = pd.get_dummies(self.data, columns=['sex'], prefix=['sex'])

        del self.train['sex_M']
        del self.train['sex_F']
        self.train = pd.get_dummies(self.train, columns=['sex'], prefix=['sex'])

    def random_sample_imputation_on_blood_type_feature(self):
        conditions = [
            (self.data['blood_A+'] == 0) & (self.data['blood_A-'] == 0) &
            (self.data['blood_AB-'] == 0) & (self.data['blood_B+'] == 0) &
            (self.data['blood_B-'] == 0) & (self.data['blood_O+'] == 0) &
            (self.data['blood_O-'] == 0) & (self.data['blood_AB+'] == 0),
            (self.data['blood_A+'] == 1),
            (self.data['blood_A-'] == 1),
            (self.data['blood_AB-'] == 1),
            (self.data['blood_B+'] == 1),
            (self.data['blood_B-'] == 1),
            (self.data['blood_O+'] == 1),
            (self.data['blood_O-'] == 1),
            (self.data['blood_AB+'] == 1)
        ]
        values = [None, 'A+', 'A-', 'AB-', 'B+', 'B-', 'O+', 'O-', 'AB+']
        self.data['blood_type'] = np.select(conditions, values)

        conditions = [
            (self.train['blood_A+'] == 0) & (self.train['blood_A-'] == 0) &
            (self.train['blood_AB-'] == 0) & (self.train['blood_B+'] == 0) &
            (self.train['blood_B-'] == 0) & (self.train['blood_O+'] == 0) &
            (self.train['blood_O-'] == 0) & (self.train['blood_AB+'] == 0),
            (self.train['blood_A+'] == 1),
            (self.train['blood_A-'] == 1),
            (self.train['blood_AB-'] == 1),
            (self.train['blood_B+'] == 1),
            (self.train['blood_B-'] == 1),
            (self.train['blood_O+'] == 1),
            (self.train['blood_O-'] == 1),
            (self.train['blood_AB+'] == 1)
        ]
        values = [None, 'A+', 'A-', 'AB-', 'B+', 'B-', 'O+', 'O-', 'AB+']
        self.train['blood_type'] = np.select(conditions, values)

        self.data['blood_type_imputed'] = self.data['blood_type'].copy()
        self.train['blood_type_imputed'] = self.train['blood_type'].copy()

        random_sample_train = self.train['blood_type'].dropna().sample(self.train['blood_type'].isnull().sum(),
                                                                       random_state=0)
        random_sample_data = self.data['blood_type'].dropna().sample(self.data['blood_type'].isnull().sum(),
                                                                     random_state=0)
        random_sample_train.index = self.train[self.train['blood_type'].isnull()].index
        random_sample_data.index = self.data[self.data['blood_type'].isnull()].index

        self.train.loc[self.train['blood_type'].isnull(), 'blood_type_imputed'] = random_sample_train
        self.data.loc[self.data['blood_type'].isnull(), 'blood_type_imputed'] = random_sample_data

        del self.train['blood_type']
        self.train.rename(columns={'blood_type_imputed': 'blood_type'}, inplace=True)

        del self.data['blood_type']
        self.data.rename(columns={'blood_type_imputed': 'blood_type'}, inplace=True)

        blood_type_list = ['blood_A+', 'blood_A-', 'blood_AB-', 'blood_AB+', 'blood_O+', 'blood_O-', 'blood_B+', 'blood_B-']
        for blood_type in blood_type_list:
            del self.data[blood_type]
            del self.train[blood_type]

        self.data = pd.get_dummies(self.data, columns=['blood_type'], prefix=['blood'])
        self.train = pd.get_dummies(self.train, columns=['blood_type'], prefix=['blood'])

    def random_sample_imputation(self):
        for feature in self.random_sample_list:
            if feature != 'sex' and feature != 'blood_type':
                self.data[f'{feature}_imputed'] = self.data[feature].copy()
                self.train[f'{feature}_imputed'] = self.train[feature].copy()

                random_sample_train = self.train[feature].dropna().sample(self.train[feature].isnull().sum(),
                                                                          random_state=0)
                random_sample_data = self.data[feature].dropna().sample(self.data[feature].isnull().sum(),
                                                                        random_state=0)
                random_sample_train.index = self.train[self.train[feature].isnull()].index
                random_sample_data.index = self.data[self.data[feature].isnull()].index

                self.train.loc[self.train[feature].isnull(), f'{feature}_imputed'] = random_sample_train
                self.data.loc[self.data[feature].isnull(), f'{feature}_imputed'] = random_sample_data

                del self.train[feature]
                self.train.rename(columns={f'{feature}_imputed': feature}, inplace=True)

                del self.data[feature]
                self.data.rename(columns={f'{feature}_imputed': feature}, inplace=True)

            if feature == 'sex':
                self.random_sample_imputation_on_sex_feature()

            # if feature == 'blood_type':
            #    self.random_sample_imputation_on_blood_type_feature()

    def end_tail_norm_imputation(self):
        for feature in self.end_tail_norm_list:
            imputer_end_tail = EndTailImputer(imputation_method='iqr', tail='left')
            imputer_end_tail.fit(self.train[[feature]])
            self.data[feature] = imputer_end_tail.transform(self.data[[feature]])

    def run_all(self):
        self.mean_imputation()
        self.median_imputation()
        self.random_sample_imputation()
        self.end_tail_norm_imputation()
        self.most_frequent_imputation()

        return self.data


class Normalization:
    def __init__(self, dataframe, minmax_list, standardization_list):
        self.dataframe = dataframe
        self.minmax_list = minmax_list
        self.standardization_list = standardization_list

    def min_max_scaling(self):
        for feature in self.minmax_list:
            # fit scaler on training data
            norm = MinMaxScaler()
            norm.fit(self.dataframe[[feature]])
            # transform training data
            self.dataframe[feature] = norm.transform(self.dataframe[[feature]])

    def standardization(self):
        for feature in self.standardization_list:
            # fit on training data column
            scale = StandardScaler()
            scale.fit(self.dataframe[[feature]])
            # transform the training data column
            self.dataframe[feature] = scale.transform(self.dataframe[[feature]])

    def run_all(self):
        self.min_max_scaling()
        self.standardization()

        return self.dataframe


def prepare_data(data, training_data):
    """
    Preform clean process as summarizes in Q28
    First part is convert non-numeric features of the data to numeric features, and extract new features.
    Second part is to apply imputation on all required features.
    Third part is to apply normalization on all features.
    And last part - drop all the un-selected features.

    Input - data - dataframe of the data we want to clean
            training_data - dataframe of the data
    """

    """convert non-numeric features of the data to numeric features, and extract new features"""
    data_numeric_class = ConvertNumericAndExtractFeatures(data.copy())
    data = data_numeric_class.run_all()

    data_training_numeric_class = ConvertNumericAndExtractFeatures(training_data.copy())
    training_data = data_training_numeric_class.run_all()

    """Apply imputation on all required features"""
    # random_sample_imputation_features_list = ['age', 'num_of_siblings', 'conversations_per_day', 'sport_activity',
    #                                           'PCR_05', 'sex', 'blood_type', 'zip_code', 'day', 'month', 'year']
    # mean_imputation_features_list = ['household_income', 'PCR_01', 'PCR_02', 'PCR_06', 'PCR_07', 'PCR_09', ]
    # median_imputation_features_list = ['weight', 'latitude']
    # most_frequent_imputation_features_list = ['happiness_score', 'sugar_levels', 'PCR_03', 'country']
    # end_tail_norm_imputation_features_list = ['longitude']

    random_sample_imputation_features_list = ['sport_activity', 'PCR_05']
    mean_imputation_features_list = ['PCR_01', 'PCR_02', 'PCR_06', 'PCR_07', 'PCR_09']
    median_imputation_features_list = []
    most_frequent_imputation_features_list = ['sugar_levels', 'PCR_03']
    end_tail_norm_imputation_features_list = []

    # dataframe, random_sample_list, mean_list, median_list, most_frequent_list, end_tail_norm_list
    data_imputation_class = Imputation(data.copy(), training_data.copy(), random_sample_imputation_features_list,
                                       mean_imputation_features_list, median_imputation_features_list,
                                       most_frequent_imputation_features_list, end_tail_norm_imputation_features_list)
    data = data_imputation_class.run_all()

    """"Apply normalization on all features"""
    minmax_normalized_features = ['PCR_01', 'PCR_02', 'PCR_04', 'PCR_06', 'PCR_08', 'PCR_10', 'sport_activity', 'PCR_05']

    standardization_normalized_features = ['sugar_levels', 'PCR_03', 'PCR_07', 'PCR_09']

    data_normalization = Normalization(data.copy(), minmax_normalized_features, standardization_normalized_features)
    data = data_normalization.run_all()

    """drop features we don't need"""
    features_list = ['PCR_01', 'PCR_04', 'PCR_02', 'PCR_05', 'PCR_06', 'PCR_08', 'PCR_10', 'PCR_09', 'sugar_levels',
                     'sport_activity', 'symptom_shortness_of_breath', 'symptom_sore_throat', 'symptom_cough',
                     'covid_score', 'symptom_fever', 'spread_score', 'blood_type']

    return data[features_list]


if __name__ == '__main__':
    dataset = pd.read_csv('HW3_data.csv')

    yoav_id = '212617864'
    mor_id = '211810452'
    random_state = int(yoav_id[-1]) + int(yoav_id[-2]) + int(mor_id[-1]) + int(mor_id[-2])
    train, test = train_test_split(dataset, test_size=0.2, random_state=random_state)

    train_clean = prepare_data(train, train)
    test_clean = prepare_data(test, train)

    train_clean.to_csv(
        r'C:\Users\javits\Technion\Introduction To Machine Learning - 236756\HW\Major_HW3\Major_HW3_ML\train_clean.csv',
        index=False)
    test_clean.to_csv(
        r'C:\Users\javits\Technion\Introduction To Machine Learning - 236756\HW\Major_HW3\Major_HW3_ML\test_clean.csv',
        index=False)