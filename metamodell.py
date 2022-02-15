import sklearn.model_selection
import pandas as pd
import numpy as np
from functools import wraps
from IPython.display import display,Markdown

class metamodell:
	def __init__(self,data,regressor,nr_inputs,evaluate_function,fieldnames=None,outputblocks=None) -> None:
		self.data=data
		self.nr_inputs=nr_inputs
		self.nr_outputs=self.data.shape[1]-nr_inputs
		self.regressor=regressor
		self.fieldnames=fieldnames
		self.evaluate_function=evaluate_function

		self.inputs=data[:,:self.nr_inputs]
		self.oututs=data[:,self.nr_inputs:]

		#outputblocks describes which groups of outputs shall be fitted together if possible
		#for example if the argument is (3,3,1)
		#self.outputblocks will be
		#[
		#(0,3)
		#(3,6)
		#(6,1)
		# ]
		if outputblocks is None:
			self.outputblocks=[(0,self.nr_outputs)]
		else:
			count=0
			self.outputblocks=[]
			for blocksize in outputblocks:
				self.outputblocks.append(count,count+blocksize)
				count+=blocksize
			

		self.outputs_need_update=True
		self.outputnames=[name for i,name in enumerate(self.fieldnames) if i>=self.nr_inputs]

	def update_outputs(self):
		"""
		update predicted outputs if needed, returns True/False wether update was done
		"""
		if self.outputs_need_update:
			self.outputs_test_predicted=self.regressor.predict(self.inputs_test)
			
			self.outputs_test_predicted_df=pd.DataFrame(self.outputs_test_predicted,columns=self.outputnames)
			
			return True
		else:
			return False
	def calculate_error_data(self):
		"""
		calculates errors for each test-datapoint and stores them in self.error_array
		"""
		new_data_split=self.update_outputs()
		nr_test_samples=self.outputs_test_df.shape[0]
		if not hasattr(self,"error_array") or new_data_split:
			for i in range(nr_test_samples):
				errors=self.evaluate_function(self.outputs_test_predicted_df.iloc[i,:],self.outputs_test_df.iloc[i,:])
				if i==0:
					self.error_array=np.zeros((nr_test_samples,len(errors)))
				self.error_array[i,:]=errors

	def test(self,reduce_function):
		"""
		evaluates a trained metamodell returns 1d-array of error-quantities of the modell
		
		1-predicts the test-inputs
		2-calls self.evaluate_function(row_predict,row) and store result in array for each test-specimen
		3.returns the reduction of the resulting array using reduce_function, eg reduce_function=np.amax
		"""
		self.calculate_error_data()
		return reduce_function(self.error_array,0)
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
		self.outputs_test_df=pd.DataFrame(self.outputs_test,columns=self.outputnames)
		#reset predicted outputs since thex changed with new training data
		self.outputs_need_update=True
		return self.idx_train,self.idx_test

def tryout_metamodells(list_name_and_regressor,data,nr_inputs,evaluate_row_func,error_size,metamodell_fields,nr_tries=10,ratio_test=0.25,reduce_functions=[np.amax,np.average,np.amin]):
	"""
	trains different kinds of metamodells on a given dataset and gives statistikal overview which metamodell fits best
	
	list_name_and_regressor=list of tuples of metamodell-name and sklearn.regressor
	data=np.array of size(n_points,n_features)
	nr_inputs=first nr_inputs columns of data array are considered inputs
	evaluate_row_func=function that returns a tuple of errors when comparing a predicted datapoint to an actual datapoint
	error_size=length of tuple returned by evaluate_row_func
	metamodell_fields=list of strings giving the name for each column
	nr_tries=how many times each metamodell is to be retrained
	ratio_test=ratio of available datapoints to be used for testing
	reduce_functions=functions to be called on array of errors for each testpoint, to create resultant errors
	"""
	
	nr_points=data.shape[0]
	nr_modells=len(list_name_and_regressor)
	error_data=np.zeros((nr_modells,error_size,len(reduce_functions),nr_tries))
	for iter_try in range(nr_tries):
		#how many datapoints shall be used for testing
		idx_train,idx_test=sklearn.model_selection.train_test_split(range(nr_points),test_size=ratio_test)
		#create modells
		modells=[]
		modell_names=[]
		for name,Regressor in list_name_and_regressor:
			modell=metamodell(data,Regressor,nr_inputs,evaluate_row_func,metamodell_fields)
			modell.split_data(idx_train,idx_test)
			print(f"training modell:{name}")
			modell.train()
			modells.append(modell)
			modell_names.append(name)
		#evaluate modells
		for iter_reduce_func,reduce_function in enumerate(reduce_functions):
			#array_modell_error=np.zeros((nr_modells,error_size))
			#df_modell_error=pd.DataFrame(array_modell_error,names)
			
			#print(reduce_function.__name__)
			for iter_modell,(name,modell) in enumerate(zip(modell_names,modells)):
				#df_modell_error.loc[name]=modell.test(reduce_function)
				print(f"evaluating modell:{name}")
				error_data[iter_modell,:,iter_reduce_func,iter_try]=modell.test(reduce_function)
			#display(df_modell_error)
	error_data_avg=np.mean(error_data,3)# mean over all tries
	error_data_std=np.std(error_data,3)#deviation over all tries

	#display gathered data:
	for iter_reduce_func,reduce_function in enumerate(reduce_functions):
		display(Markdown((f"# {reduce_function.__name__} of errors of all test-points")))
		df_avg=pd.DataFrame(error_data_avg[:,:,iter_reduce_func],modell_names)
		avg_style=df_avg.style.set_caption(f"average over {nr_tries} different training-sets")
		df_std=pd.DataFrame(error_data_std[:,:,iter_reduce_func],modell_names)
		std_style=df_std.style.set_caption(f"standart deviation over {nr_tries} different training-sets")
		display(avg_style)
		display(std_style)
def printfun():
	print("hi i am from your module")