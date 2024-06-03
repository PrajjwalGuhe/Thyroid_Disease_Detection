import pandas as pd
import numpy as np
import sys
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from src.ThyroidDiseaseDetection.logger import logging
from src.ThyroidDiseaseDetection.exception import customexception
from dataclasses import dataclass
from src.ThyroidDiseaseDetection.utils.utils import save_object
from sklearn.preprocessing import LabelEncoder

@dataclass
class DatatransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DatatransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation Initiated')
            categorical_col = ['sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured', 'referral_source']
            numerical_col = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']

            sex = ['F', 'M']
            on_thyroxine = ['t', 'f']
            query_on_thyroxine = ['t', 'f']
            on_antithyroid_medication = ['t', 'f']
            sick = ['t', 'f']
            pregnant = ['t', 'f']
            thyroid_surgery = ['t', 'f']
            I131_treatment = ['t', 'f']
            query_hypothyroid = ['t', 'f']
            query_hyperthyroid = ['t', 'f']
            lithium = ['t', 'f']
            goitre = ['t', 'f']
            tumor = ['t', 'f']
            hypopituitary = ['t', 'f']
            psych = ['t', 'f']
            TSH_measured = ['t', 'f']
            T3_measured = ['t', 'f']
            TT4_measured = ['t', 'f']
            T4U_measured = ['t', 'f']
            FTI_measured = ['t', 'f']
            TBG_measured = ['f']
            referral_source = ['SVHC', 'other', 'SVI', 'STMW', 'SVHD']
            binary_class = ['N', 'P']

            logging.info('Pipeline initiated')

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[sex, on_thyroxine, query_on_thyroxine, on_antithyroid_medication, sick, pregnant, thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid, lithium, goitre, tumor, hypopituitary, psych, TSH_measured, T3_measured, TT4_measured, T4U_measured, FTI_measured, TBG_measured, referral_source])),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_col),
                ('cat_pipeline', cat_pipeline, categorical_col)
            ])

            logging.info('Pipeline completed')
            return preprocessor

        except Exception as e:
            logging.info('Error in data transformation')
            raise customexception(e, sys)

    def standardize_column_names(self, df):
        df.columns = df.columns.str.replace(' ', '_', regex=True)
        return df

    def replace_placeholders(self, df):
        df.replace('?', np.nan, inplace=True)
        return df

    def calculate_missing_tt4_percentage(self, df):
        missing_tt4_count = df['TT4'].isnull().sum()
        total_elements = df.shape[0] * df.shape[1]
        percentage_missing_tt4 = (missing_tt4_count / total_elements) * 100
        logging.info(f'Percentage of missing TT4 values: {percentage_missing_tt4}%')
        return percentage_missing_tt4

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            # Standardize column names
            train_df = self.standardize_column_names(train_df)
            test_df = self.standardize_column_names(test_df)

            # Replace placeholders with NaN
            train_df = self.replace_placeholders(train_df)
            test_df = self.replace_placeholders(test_df)

            logging.info(f'Train DataFrame Columns: {train_df.columns.tolist()}')
            logging.info(f'Test DataFrame Columns: {test_df.columns.tolist()}')

            # Calculate missing TT4 percentage
            self.calculate_missing_tt4_percentage(train_df)
            self.calculate_missing_tt4_percentage(test_df)

            # Verify the column existence
            required_columns = ['on_thyroxine', 'binaryClass']
            for col in required_columns:
                if col not in train_df.columns:
                    raise ValueError(f"Required column '{col}' is not present in the training dataset")
                if col not in test_df.columns:
                    raise ValueError(f"Required column '{col}' is not present in the testing dataset")

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'binaryClass'
            drop_columns = [target_column_name]
            label_encoder = LabelEncoder()

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = label_encoder.fit_transform(train_df[target_column_name])

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = label_encoder.fit_transform(test_df[target_column_name])

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessing object on training and testing datasets')

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception occurred in the initiate data transformation")
            raise customexception(e, sys)
