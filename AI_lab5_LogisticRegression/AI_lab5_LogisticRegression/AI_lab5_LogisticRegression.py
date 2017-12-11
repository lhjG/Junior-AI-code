import numpy;
import pylab;
def sigmoid(h):
	return 1/(1+numpy.e**(-h))
def LR(step,upperlimit):
	train = numpy.loadtxt(open("D:train.csv", "rb"), delimiter=",", skiprows=0) ##将训练集全部数据读作矩阵
	test = numpy.loadtxt(open("D:test.csv", "rb"), delimiter=",", skiprows=0) 
	train_amount=train.shape[0]
	train_label_vector=numpy.zeros([train_amount,1])
	train_label_vector[:,0]=train[:,6]   ##通过截取train的最后一列将所有训练集的标签读作向量
	train_add_one_matrix = numpy.ones([train_amount,7])
	test_add_one_matrix = numpy.ones([test.shape[0],7])
	train_add_one_matrix[:,1:7]=train[:,0:6] ##在训练集矩阵的最前面加一列1，组成新的矩阵用于运算
	test_add_one_matrix[:,1:7]=test[:,0:6]
	w = numpy.zeros([1,7])   ##初始化w
	for i in range(upperlimit):
		h = numpy.dot(train_add_one_matrix,w.T)
		diff=step*numpy.dot((sigmoid(h)-train_label_vector).T,train_add_one_matrix)
		w-=diff
		'''step=0.00001*numpy.sum(abs(diff))    动态调整步长'''
	p_vector=sigmoid(numpy.dot(test_add_one_matrix,w.T))   ##预测的所有验证集属于正类的概率组成的向量
	print(w)
	test_amount=test.shape[0]
	f=open("D:result.txt","w")
	for i in range(len(p_vector)):   ##计算正确率
		if p_vector[i]<0.5:
			p_vector[i]=0
		elif p_vector[i]>0.5:
			p_vector[i]=1
		print(p_vector[i])
		f.write(str(int(p_vector[i][0])))
		f.write('\n')
####################################注释掉的部分是交叉验证的代码##################################	
	'''train = numpy.loadtxt(open("D:train.csv", "rb"), delimiter=",", skiprows=0)
	test = numpy.loadtxt(open("D:train.csv", "rb"), delimiter=",", skiprows=0)'''
	'''for i in range(8):  ##交叉验证选取i*1000-i*1000+1000的数据为验证集
		train = numpy.loadtxt(open("D:train.csv", "rb"), delimiter=",", skiprows=0)   ##将训练集全部数据读作矩阵
		test=numpy.ones([1000,41])    
		test[0:1000,:]=train[i*1000:i*1000+1000,:]   ##截取train的  i*1000——i*1000+1000 行的数据为验证集 
		train=numpy.delete(train,numpy.s_[i*1000:i*1000+1000:1],axis=0) ##剔除训练集中被选取作为验证集的行
		train_amount=train.shape[0]
		train_label_vector=numpy.zeros([train_amount,1])
		train_label_vector[:,0]=train[:,40]   ##通过截取train的最后一列将所有训练集的标签读作向量
		train_add_one_matrix = numpy.ones([train_amount,41])
		train_add_one_matrix[:,1:41]=train[:,0:40]  ##在训练集矩阵的最前面加一列1，组成新的矩阵用于运算
		w = numpy.ones([1,41])   ##初始化w
		for i in range(upperlimit):
			h = numpy.dot(train_add_one_matrix,w.T)
			diff=step*numpy.dot((sigmoid(h)-train_label_vector).T,train_add_one_matrix)
			w-=diff
			step=0.00001*numpy.sum(abs(diff))
		p_vector=sigmoid(numpy.dot(test,w.T))   ##预测的所有验证集属于正类的概率组成的向量
		sum=0
		test_amount=test.shape[0]
		for i in range(len(p_vector)):   ##计算正确率
			if p_vector[i]<0.5:
				p_vector[i]=0
			elif p_vector[i]>0.5:
				p_vector[i]=1
			if p_vector[i]==test[i][40]:
				sum+=1
		global result
		result.append(sum/test_amount)
	#print(result)'''
if __name__=="__main__":
	LR(1,1)
	'''for i in range(10):
		result=[]
		LR(0.00001*(i+1),100)
		if i ==0:
			plot0=pylab.plot(result,'r',label='0.00001')
		if i ==1:
			plot1=pylab.plot(result,'g',label='0.00002')
		if i ==2:
			plot2=pylab.plot(result,'b',label='0.00003')
		if i ==3:
			plot3=pylab.plot(result,'c',label='0.00004')
		if i ==4:
			plot4=pylab.plot(result,'m',label='0.00005')
		if i ==5:
			plot5=pylab.plot(result,'y',label='0.00006')
		if i ==6:
			plot6=pylab.plot(result,'k',label='0.00007')
		if i ==7:
			plot7=pylab.plot(result,'m',linewidth='2.5',label='0.00008')
		if i ==8:
			plot8=pylab.plot(result,'y',linewidth='2.5',label='0.00009')
		if i ==9:
			plot9=pylab.plot(result,'k',linewidth='2.5',label='0.0001')
	pylab.legend(loc='upper left')
	pylab.show()'''
	'''result=[]
	LR(0.00001,100)
	plot0=pylab.plot(result,'r',label='100')
	result=[]
	LR(0.00001,500)
	plot1=pylab.plot(result,'g',label='500')
	result=[]
	LR(0.00001,1000)
	plot2=pylab.plot(result,'b',label='1000')
	result=[]
	LR(0.00001,5000)
	plot3=pylab.plot(result,'k',label='5000')
	result=[]
	LR(0.00001,10000)
	plot4=pylab.plot(result,'m',label='10000')
pylab.legend(loc='upper left')
pylab.show()'''
		