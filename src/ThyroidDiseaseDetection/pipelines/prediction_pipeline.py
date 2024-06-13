import os
import sys
import pandas as pd
from src.ThyroidDiseaseDetection.exception import customexception
from src.ThyroidDiseaseDetection.logger import logging
from src.ThyroidDiseaseDetection.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")
            
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            
            scaled_data=preprocessor.transform(features)
            
            pred=model.predict(scaled_data)
            
            return pred
            
            
        
        except Exception as e:
            raise customexception(e,sys)
    
    
    
class CustomData:
    def __init__(self, data):
        self.sex = data['sex']
        self.on_thyroxine = data['on_thyroxine']
        self.query_on_thyroxine = data['query_on_thyroxine']
        self.on_antithyroid_medication = data['on_antithyroid_medication']
        self.sick = data['sick']
        self.pregnant = data['pregnant']
        self.thyroid_surgery = data['thyroid_surgery']
        self.I131_treatment = data['I131_treatment']
        self.query_hypothyroid = data['query_hypothyroid']
        self.query_hyperthyroid = data['query_hyperthyroid']
        self.lithium = data['lithium']
        self.goitre = data['goitre']
        self.tumor = data['tumor']
        self.hypopituitary = data['hypopituitary']
        self.psych = data['psych']
        self.TSH_measured = data['TSH_measured']
        self.T3_measured = data['T3_measured']
        self.TT4_measured = data['TT4_measured']
        self.T4U_measured = data['T4U_measured']
        self.FTI_measured = data['FTI_measured']
        self.TBG_measured = data['TBG_measured']
        self.referral_source = data['referral_source']
        self.age = data['age']
        self.TSH = data['TSH']
        self.T3 = data['T3']
        self.TT4 = data['TT4']
        self.T4U = data['T4U']
        self.FTI = data['FTI']

    def get_data_as_dataframe(self):
        try:
            data = {
                'sex': [self.sex],
                'on_thyroxine': [self.on_thyroxine],
                'query_on_thyroxine': [self.query_on_thyroxine],
                'on_antithyroid_medication': [self.on_antithyroid_medication],
                'sick': [self.sick],
                'pregnant': [self.pregnant],
                'thyroid_surgery': [self.thyroid_surgery],
                'I131_treatment': [self.I131_treatment],
                'query_hypothyroid': [self.query_hypothyroid],
                'query_hyperthyroid': [self.query_hyperthyroid],
                'lithium': [self.lithium],
                'goitre': [self.goitre],
                'tumor': [self.tumor],
                'hypopituitary': [self.hypopituitary],
                'psych': [self.psych],
                'TSH_measured': [self.TSH_measured],
                'T3_measured': [self.T3_measured],
                'TT4_measured': [self.TT4_measured],
                'T4U_measured': [self.T4U_measured],
                'FTI_measured': [self.FTI_measured],
                'TBG_measured': [self.TBG_measured],
                'referral_source': [self.referral_source],
                'age': [self.age],
                'TSH': [self.TSH],
                'T3': [self.T3],
                'TT4': [self.TT4],
                'T4U': [self.T4U],
                'FTI': [self.FTI]
            }
            df = pd.DataFrame(data)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise customexception(e,sys)