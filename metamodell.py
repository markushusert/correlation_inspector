import sklearn.model_selection
import pandas as pd
import numpy as np
from functools import wraps


class metamodell:
    def __init__(self,data,regressor,nr_inputs,evaluate_function,fieldnames=None) -> None:
        self.data=data
        self.nr_inputs=nr_inputs
        self.regressor=regressor
        self.fieldnames=fieldnames
        self.evaluate_function=evaluate_function

        self.inputs=data[:,:self.nr_inputs]
        self.oututs=data[:,self.nr_inputs:]


    def test(self,reduce_function):
        """
        evaluates a trained metamodell returns 1d-array of error-quantities of the modell
        
        1-predicts the test-inputs
        2-calls self.evaluate_function(row_predict,row) and store result in array for each test-specimen
        3.returns the reduction of the resulting array using reduce_function, eg reduce_function=np.amax
        """
        self.outputs_test_predicted=self.regressor.predict(self.inputs_test)
        outputnames=[name for i,name in enumerate(self.fieldnames) if i>=self.nr_inputs]
        self.outputs_test_predicted_df=pd.DataFrame(self.outputs_test_predicted,columns=outputnames)
        self.outputs_test_df=pd.DataFrame(self.outputs_test,columns=outputnames)
        nr_test_samples=self.outputs_test_df.shape[0]
        for i in range(nr_test_samples):
            errors=self.evaluate_function(self.outputs_test_predicted_df.iloc[i,:],self.outputs_test_df.iloc[i,:])
            if i==0:
                error_array=np.zeros((nr_test_samples,len(errors)))
            error_array[i,:]=errors
        return reduce_function(error_array,0)
    def train(self):
        self.regressor.fit(self.inputs_train,self.oututs_train)
    def split_data(self,train_idxs=None,test_idxs=None,test_ratio=None):
        '''
        splits data randomly into train- and test-data
        '''
        nr_points=self.data.shape[0]
        if train_idxs is None or test_idxs is None:
            if test_ratio is None:
                raise Exception("provide either train_idxs and test_idxs or test_ratio")
            #idx to split by are not given, so generate new ones
            self.data_train,self.data_test,self.idx_train,self.idx_test=sklearn.model_selection.train_test_split(self.data,range(nr_points),test_size= self.test_ratio)
        else:
            
            #use given indices to split data
            self.data_train=self.data[train_idxs,:]
            self.data_test=self.data[test_idxs,:]
            self.idx_train=train_idxs
            self.idx_test=test_idxs
        self.inputs_train=self.data_train[:,:self.nr_inputs]
        self.oututs_train=self.data_train[:,self.nr_inputs:]
        self.inputs_test=self.data_test[:,:self.nr_inputs]
        self.outputs_test=self.data_test[:,self.nr_inputs:]
        return self.idx_train,self.idx_test