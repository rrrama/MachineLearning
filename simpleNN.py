import numpy as np

class SimpleNN():
	
	weights=[]
	biases=[]
	vecAct=None

	def sigmoid(self,item):
		return 1/(1+np.exp(-item))

	def relu(self):
		pass

	def tanh(self):
		pass

	functions={}


	def __init__(self,layout,fromFile=False,fileName='',activation='sigmoid',learnRate=0.5,batchSize=1):
		functions={
			'sigmoid':self.sigmoid,
			'relu':np.vectorize(self.relu),
			'tanh':np.vectorize(self.tanh)
		}
		if fromFile:
			#load weights from file
			raise ValueError("i am lazy, not yet implemented")
		else:
			
			self.biases=[np.random.rand(i,1) for i in layout[1:]]
			self.weights=[np.random.rand(layout[index+1],value) for index,value in enumerate(layout[:-1])]

		self.vecAct = functions[activation]
		self.learnRate = learnRate

	


	def forwardPass(self,inpVec):
		currentVec=np.copy(inpVec)
		for index,layer in enumerate(self.weights):
			currentVec = self.vecAct(np.add(np.dot(layer,currentVec),self.biases[index]))
		return currentVec

	def cost(self,actual,target):
		return np.sum(np.multiply(np.square(np.subtract(target,actual)),0.5))


	def train(self,fromFile=False,trainingData='trainingData.txt',inputData = [],targetData=[]):
		if fromFile:
			pass
		else:
			if inputData==[] or targetData==[]:
				raise ValueError("No input or target data for training!")

			

			

	def save(self,filename):
		pass


if __name__=="__main__":
	nn = SimpleNN([2,3,2])
	print(nn.forwardPass(np.array([[2],[3]])))
	input()