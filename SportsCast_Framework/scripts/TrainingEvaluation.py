from .SavingReading import SavingReading
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
from .Model import Model
from .MultiARIMA import MultiARIMA
import logging
import os


class TrainingEvaluation:

    def __init__(self,Model_Cls:Model,use_exog_feat:bool,hparams:str="",train_ds_all:ListDataset=None):
        '''
        Parameters
        ----
        Model_Cls: class of model to train (DeepAR or MultiARIMA). Chosen "on-the-fly"

        use_exog_feat: whether to use exogenous features for training and evaluation

        train_ds_all: training dataset for ALL players

        hparams: string of hyperparameters for naming results with a unique suffix

        Returns
        ----
        True if success, None otherwise
        '''
        self.saver_reader = SavingReading()
        self.Model_Cls = Model_Cls
        self.use_exog_feat = use_exog_feat
        self.model = None if train_ds_all is None else Model_Cls(train_ds_all,transform='none')
        #Set hparams str    
        #hparams = f"transform=__,..." #TODO
        self.hparams = hparams

    def train(self,train_ds_dir:str=os.getcwd()+"/data/train_ds",train_ds_fname:str="train_ds_all", \
                model_train_dir:str=os.getcwd()+"/data/models/trained", model_train_save_filename:str="model_trained", \
                trn_params_sffix:str=""):
        '''
        Parameters
        ----
        train_ds_dir: directory for training data

        train_ds_fname: filename of same

        model_train_dir: directory to save trained model

        model_train_save_filename: filename for same

        Returns
        ----
        True if success, None otherwise
        '''


        if self.model is None:
            #Get training data
            try:
                train_ds_all = self.saver_reader.read(file_ext='.p',read_name=train_ds_fname+trn_params_sffix,full_read_dir=train_ds_dir,bool_read_s3=False)
            except AssertionError as err:
                logging.error(f'Couldn\'t open path: {err}. Can\'t create model to train')
                return None
            
            #Instatiate models
            self.model = self.Model_Cls(train_ds_all,transform='none') #TODO: ADD MORE HPARAM ARGUMENTS

        
        self.model.create(use_exog_feat=self.use_exog_feat) #Create models w/ or w/o features
        #Fit to dataset
        self.model.fit()
        #Save model
        if not self.saver_reader.save(self.model,model_train_save_filename+self.hparams,model_train_dir):
            logging.warn(f'Failed to save trained model!')

        #TODO: possibly evaluate training error
        return True


    def evaluate(self,model_train_dir:str=os.getcwd()+"/data/models/trained", model_train_save_filename:str="model_trained",\
                test_ds_dir:str=os.getcwd()+"/data/test_ds",test_ds_fname:str="test_ds_all",\
                modelresult_dir:str=os.getcwd()+"/data/models", modelresult_filename:str="model_results",
                test_params_sffix:str="", retrain_horizon:int=0):
        '''
        Parameters
        ----
        model_train_dir: directory to open trained model

        model_train_save_filename: filename for same

        test_ds_dir: directory for testing data

        test_ds_fname: filename for same

        modelresult_dir: directory for saving model+result dataframe (see MultiARIMA.py)

        modelresult_filename: filename for same

        test_params_sffix: string to append to filename to specify parameters for testing

        retrain_horizon: horizon (see MultiARIMA.py)

        Returns
        ----
        True if success, None otherwise
        '''


        if model_train_save_filename == "model_trained":
            model_train_save_filename = model_train_save_filename + self.hparams
        if modelresult_filename == "model_results":
            modelresult_filename = modelresult_filename + self.hparams

        assert self.model is not None, "Create model first! (and ideally also train)"

        #Get testing data
        try:
            test_ds_all = self.saver_reader.read(file_ext='.p',read_name=test_ds_fname+test_params_sffix,full_read_dir=test_ds_dir,bool_read_s3=False)
        except AssertionError as err:
            logging.error(f'Couldn\'t open path: {err}. Can\'t create model to train')
            return None

        self.model.evaluate(test_ds_all,horizon=retrain_horizon) #Run evaluation

        #Save tested aggregate model and results
        if not self.saver_reader.save(self.model.model_results_df,modelresult_filename+self.hparams,modelresult_dir):
            logging.warn(f'Failed to save trained model!')