import numpy;
import pylab;
import random;
import math;

def sigmoid(x):
	return 1/(1+math.e**(-x))

##计算输出节点的delta
def odelta(x,y):
	# print(x,"(1-",x,")(",y,"-",x,")=",end='')
	# print(x*(1-x)*(y-x))
	return y-x
	# return x*(1-x)*(y-x)

##计算隐藏层节点的delta
def hdelta(x,y):
	# print(x,"(1-",x,")(",y,")=",end='')
	# print(x*(1-x)*y)
	return x*(1-x)*y

##计算损失函数
def MSE(x,y):
	return numpy.mean(numpy.square(x-y)*0.5)

class BPNN:
	def __init__(self,inputs,hiddens,outputs,data):
		##初始化节点数目
		##输入层
		self.ni=inputs
		##隐藏层 
		self.nh=hiddens
		##输出层
		self.no=outputs
		#初始化权重矩阵为0-1之间的随机浮点数(最后一行给偏置项，所以要多加一行)
		self.w_to_hidden=numpy.zeros([self.ni+1,self.nh])
		self.w_to_output=numpy.zeros([self.nh+1,self.no])
		for i in range(self.nh):
			for j in range(self.ni+1):
				self.w_to_hidden[j][i]=random.uniform(0,1)
		# print("初始化的w_to_hidden:")
		# print(self.w_to_hidden)
		for i in range(self.no):
			for j in range(self.nh+1):
				self.w_to_output[j][i]=random.uniform(0,1)
		##初始化各个节点的输入输出值
		##输入层
		self.ai=numpy.ones(self.ni+1)
		#用于矩阵运算的输入层输出
		self.ai_m=numpy.ones([data.shape[1],data.shape[0]])
		#增加了偏置行的矩阵
		self.ai_m1=numpy.ones([data.shape[1]+1,data.shape[0]])
		##隐藏层
		self.ah=numpy.ones(self.nh+1)
		##用于矩阵运算的隐藏层输出
		self.ah_m=numpy.ones([self.nh,data.shape[0]])
		##增加了偏置行的隐藏层输出矩阵
		self.ah_m1=numpy.ones([self.nh+1,data.shape[0]])
		##输出层
		self.ao=numpy.ones(self.no)
		##初始化各个节点的delta
		self.do=numpy.ones(self.no)
		self.dh=numpy.ones(self.nh)
		##初始化各个节点的梯度:
		self.go=numpy.ones([self.nh+1,self.no])
		self.gh=numpy.ones([self.ni+1,self.nh])
		##初始化一个预测结果向量
		self.trainres=numpy.ones(1)
	'''
	前向传递矩阵运算
	'''
	def forward_matrix(self,inputs):
		tmp=numpy.ones([1,inputs.shape[0]])
		self.ai_m=inputs.T  ##输入层矩阵
		self.ai_m1=numpy.row_stack((self.ai_m,tmp)) #给输入层矩阵增加一行偏置行
		self.ah_m=sigmoid(numpy.dot(self.w_to_hidden.T,self.ai_m1)) #隐藏层的输出矩阵
		self.ah_m1=numpy.row_stack((self.ah_m,tmp))  #给隐藏层的输出矩阵增加一行偏置行
		self.trainres=numpy.dot(self.w_to_output.T,self.ah_m1) #结果矩阵
	'''
	前向传递普通运算
	'''
	def forward(self,inputs):
		if len(inputs)!=self.ni:
			print("输入层节点数量与数据量不同")
		##将输入数据读进网络
		for i in range(self.ni):
			self.ai[i]=inputs[i]
		#数据进入隐藏层
		for i in range(self.nh):
			sum=0
			for j in range(self.ni+1):
				sum+=self.ai[j]*self.w_to_hidden[j][i]
			self.ah[i]=sum
		#数据进入输出层
		for i in range(self.no):
			sum=0
			for j in range(self.nh+1):
				sum+=self.ah[j]*self.w_to_output[j][i]
			self.ao[i]=sigmoid(sum)
	'''
	后向传递矩阵运算
	'''
	def backward_matrix(self,inputs,step,labels):
		output_delta_v=labels-self.trainres  #行向量  1*8000
		output_gradient_m=numpy.dot(self.ah_m1,output_delta_v.T)  #每一项都是所有样本delta*前一项对应输出的和
		tmp_w_to_output=self.w_to_output[0:8,:]  #去除偏置行
		hidden_delta_m=hdelta(self.ah_m.T,numpy.dot(output_delta_v.T,tmp_w_to_output.T))   #后一项的维度是8000*隐藏层节点数
		hidden_gradient_m=numpy.dot(self.ai_m1,hidden_delta_m)
		self.w_to_output+=step*output_gradient_m/inputs.shape[0]
		self.w_to_hidden+=step*hidden_gradient_m/inputs.shape[0]
	'''
	后向传递普通运算
	'''
	def backward(self,labels,step):
		##计算各个输出节点的反向梯度
		for i in range(self.no):
			self.do[i]=odelta(self.ao[i],labels[i])
			for j in range(self.nh+1):
				self.go[j][i]=self.do[i]*self.ah[j]
		##计算隐藏层节点的反向梯度
		for i in range(self.nh):
			sum=0
			for j in range(self.no):
				sum+=self.do[j]*self.w_to_output[i][j]
			self.dh[i]=hdelta(self.ah[i],sum)
			for j in range(self.ni+1):
				self.gh[j][i]=self.dh[i]*self.ai[j]
		##更新w_to_output
		for i in range(self.no):
			for j in range(self.nh+1):
				self.w_to_output[j][i]+=step*self.go[j][i]
		##更新w_to_hidden
		for i in range(self.nh):
			for j in range(self.ni+1):
				self.w_to_hidden[j][i]+=step*self.gh[j][i]

	##训练网络，每次随机抽取一个样本作为训练样本，迭代一定次数
	def train(self,trainset,labels,iteration,step):
		size=trainset.shape[0]
		train_mse=[]
		for i in range(iteration):
			'''
			随机梯度下降
			'''
			rand=random.randint(0,size-1)  #产生数据样本数范围内的随机数
			sample=trainset[rand,:]  #抽出样本
			label=numpy.zeros(1)   #被抽出样本的正确label
			label[0]=labels[rand]
			self.forward(sample)   #前向  
			self.backward(label,step)  #后向
			self.forward_matrix(trainset)
			将该样本的标签转换成一个1维向量作为backward的参数
			'''
			批梯度下降
			'''
			self.forward_matrix(trainset)
			self.backward_matrix(trainset,step,labels)
			#跑一遍全局数据
			self.forward_matrix(trainset)
			train_mse.append(MSE(self.trainres,labels))   #求每次迭代的误差均值
		pylab.plot(train_mse)
	'''
	验证函数，将每一次迭代的验证MSE结果画出来
	'''
	def validate(self,trainset,validationset,labelst,labelsv,iteration,step):
		'''
		画出验证集上的MSE函数随迭代次数增加而变化的图像
		'''
		size=validationset.shape[0]
		validation_mse=[]
		for i in range(iteration):
			self.forward_matrix(trainset)
			self.backward_matrix(trainset,step,labelst)
			self.forward_matrix(validationset)
			validation_mse.append(MSE(self.trainres,labelsv))
		pylab.plot(validation_mse)

	def normalvalidate(self,trainset,validationset,labelst,labelsv,iteration,step):
		'''
		画出在验证集上预测值与正确值各自的图像
		'''
		size=validationset.shape[0]
		validation_mse=[]
		for i in range(iteration):
			self.forward_matrix(trainset)
			self.backward_matrix(trainset,step,labelst)
		self.forward_matrix(validationset)

'''
数据预处理，对每一列进行标准归一化
'''
def preprocess(data):
	return (data-numpy.mean(data,axis=0))/(numpy.std(data,axis=0)+0.000001)  ##+0.000001是为了防止分母为0

if __name__=="__main__":
	##普通数据集
	# data=preprocess(numpy.loadtxt('D:\\train1.csv',skiprows=1,delimiter=',',usecols=(2,3,4,5,6,7,8,9,10,11,12,13),dtype=float))
	# labels=numpy.loadtxt('D:\\train1.csv',skiprows=1,delimiter=',',usecols=(14),dtype=float)
	# bpnn=BPNN(data.shape[1],8,1,data)
	# bpnn.train(data,labels,2000,0.001)
	# pylab.show()

	#预测
	data=preprocess(numpy.loadtxt('D:\\train1.csv',skiprows=1,delimiter=',',usecols=(2,3,4,5,6,7,8,9,10,11,12,13),dtype=float))
	labels=numpy.loadtxt('D:\\train1.csv',skiprows=1,delimiter=',',usecols=(14),dtype=float)
	test=preprocess(numpy.loadtxt('D:\\test.csv',skiprows=1,delimiter=',',usecols=(2,3,4,5,6,7,8,9,10,11,12,13),dtype=float))
	bpnn=BPNN(data.shape[1],8,1,data)
	for i in range(2000):
		bpnn.forward_matrix(data)
		bpnn.backward_matrix(data,0.001,labels)
	bpnn.forward_matrix(test)
	file=open("D:\\result.txt",'w')
	for i in range(test.shape[0]):
		file.write(str(bpnn.trainres[0][i]))
		file.write('\n')

	##normal validation
	# data=numpy.loadtxt('D:\\train1.csv',skiprows=1,delimiter=',',usecols=(2,3,4,5,6,7,8,9,10,11,12,13),dtype=float)
	# labels=numpy.loadtxt('D:\\train1.csv',skiprows=1,delimiter=',',usecols=(14),dtype=float)
	# trainset=preprocess(data)
	# validationset=preprocess(data[8146:8620,:])
	# labels_crossvalid=labels[8146:8620]
	# pylab.plot(labels_crossvalid,color='g',label='Data')
	# bpnn=BPNN(trainset.shape[1],8,1,trainset)
	# bpnn.normalvalidate(trainset,validationset,labels,labels_crossvalid,2000,0.001)
	# pylab.plot(bpnn.trainres[0],color='b',label = 'Predict')
	# pylab.show()

	##cross validation交叉验证
	# data=numpy.loadtxt('D:\\train1.csv',skiprows=1,delimiter=',',usecols=(2,3,4,5,6,7,8,9,10,11,12,13),dtype=float)
	# labels=numpy.loadtxt('D:\\train1.csv',skiprows=1,delimiter=',',usecols=(14),dtype=float)
	# for i in range(8):	
	# 	validationset=preprocess(data[i*1000:i*1000+1000,:])   ##每次取全部数据集的i*1000到i*1000+1000的数据做验证集
	# 	trainset=preprocess(numpy.delete(data,numpy.s_[i*1000:i*1000+1000:1],axis=0))  ##剔除作为验证集的数据之后剩下的数据作为训练集
	# 	bpnn=BPNN(trainset.shape[1],8,1,trainset)
	# 	labels_crosstrain=numpy.delete(labels,numpy.s_[i*1000:i*1000+1000:1],axis=0)   ##获取与数据集对应的label
	# 	labels_crossvalid=labels[i*1000:i*1000+1000]
	# 	bpnn.validate(trainset,validationset,labels_crosstrain,labels_crossvalid,2000,0.001)  
	# 	# bpnn.train(trainset,labels_crosstrain,2000,0.001)
	# pylab.show()

