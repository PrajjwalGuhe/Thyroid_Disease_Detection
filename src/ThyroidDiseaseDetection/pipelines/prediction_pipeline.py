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
    def __init__(self,
                 sex: str,
                 on_thyroxine: str,
                 query_on_thyroxine: str,
                 on_antithyroid_medication: str,
                 sick: str,
                 pregnant: str,
                 thyroid_surgery: str,
                 I131_treatment: str,
                 query_hypothyroid: str,
                 query_hyperthyroid: str,
                 lithium: str,
                 goitre: str,
                 tumor: str,
                 hypopituitary: str,
                 psych: str,
                 TSH_measured: str,
                 T3_measured: str,
                 TT4_measured: str,
                 T4U_measured: str,
                 FTI_measured: str,
                 TBG_measured: str,
                 referral_source: str,
                 age: float,
                 TSH: float,
                 T3: float,
                 TT4: float,
                 T4U: float,
                 FTI: float):
        
        self.sex = sex
        self.on_thyroxine = on_thyroxine
        self.query_on_thyroxine = query_on_thyroxine
        self.on_antithyroid_medication = on_antithyroid_medication
        self.sick = sick
        self.pregnant = pregnant
        self.thyroid_surgery = thyroid_surgery
        self.I131_treatment = I131_treatment
        self.query_hypothyroid = query_hypothyroid
        self.query_hyperthyroid = query_hyperthyroid
        self.lithium = lithium
        self.goitre = goitre
        self.tumor = tumor
        self.hypopituitary = hypopituitary
        self.psych = psych
        self.TSH_measured = TSH_measured
        self.T3_measured = T3_measured
        self.TT4_measured = TT4_measured
        self.T4U_measured = T4U_measured
        self.FTI_measured = FTI_measured
        self.TBG_measured = TBG_measured
        self.referral_source = referral_source
        self.age = age
        self.TSH = TSH
        self.T3 = T3
        self.TT4 = TT4
        self.T4U = T4U
        self.FTI = FTI

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
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
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise customexception(e,sys)