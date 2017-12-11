import math;
import copy;
import numpy;
import pylab;
from collections import Counter;

class Sample:     ##一个样本是一个sample，里面的attributes用来盛装属性值，label就是标签
	def __init__(self):
		self.attributes = []
		self.label=0
class Node:
	def __init__(self):
		self.dataset=[]
		self.children=[]
		self.attribute_value=0
		self.attribute_or_decision=0
		self.parent=Sample()
		self.mark=[]
		self.level=0
		
def make_decision(decision_tree,test_sample):
	if len(decision_tree.children)==0:
		return decision_tree.attribute_or_decision
	else:
		size_of_chidren = len(decision_tree.children)
		for j in range(size_of_chidren):
			if test_sample.attributes[decision_tree.attribute_or_decision]==decision_tree.children[j].attribute_value:
				return make_decision(decision_tree.children[j],test_sample)
def read_data(dataset_train,dataset_test):
	train_string = dataset_train.read()
	test_string = dataset_test.read()
	train_linebreak_splits = train_string.split('\n')
	test_linebreak_splits = test_string.split('\n')
	train_linebreak_splits.pop()
	test_linebreak_splits.pop()
	for linebreak_split in train_linebreak_splits:
		attributes_and_label = linebreak_split.split(',')
		sample = Sample()
		sample.label=attributes_and_label[-1]
		attributes_and_label.pop() ##删掉label 便于sample.attributes的遍历数据赋值
		for attribute in attributes_and_label:
			sample.attributes.append(attribute)
		train_sample_list.append(sample)
	for linebreak_split in test_linebreak_splits:
		attributes_and_label = linebreak_split.split(',')
		sample = Sample()
		sample.label=attributes_and_label[-1]
		attributes_and_label.pop() ##删掉label 便于sample.attributes的遍历赋值
		for attribute in attributes_and_label:
			sample.attributes.append(attribute)
		test_sample_list.append(sample)
	global train_amount
	train_amount=len(train_linebreak_splits)
	global test_amount
	test_amount = len(test_linebreak_splits)
def choose_attribute(root,method):
	if method==1:  #ID3
		ID3_list=[]
		origin_entropy=0
		conditional_entropy_list=[]
		label_list=[]
		label_amount_dictionary={}
		whether_label_same=1
		whether_all_attributes_same=1
		for i in range(len(root.dataset)):
			if root.dataset[i].label not in label_list:
				label_list.append(root.dataset[i].label)
				label_amount_dictionary[root.dataset[i].label]=1
			else:
				label_amount_dictionary[root.dataset[i].label]+=1
		for i in range(len(label_list)):
			p = float(label_amount_dictionary[label_list[i]]/len(root.dataset))
			origin_entropy+=float((-p)*(math.log(p))/math.log(2))
		if len(root.dataset)==0:   ##数据集为空
			return -1
		for i in range(len(root.dataset)-1):
			if root.dataset[i].label!=root.dataset[i+1].label:
				whether_label_same=0
				break
		if whether_label_same:   ##所有数据的label相同
			return -2
		if len(root.dataset[0].attributes)==numpy.sum(root.mark):  ##没有可用属性
			return -3
		##print("目前数据集共有：",len(dataset[0].attributes),"个属性")
		for i in range(len(root.dataset[0].attributes)):  ##遍历所有属性
			if root.mark[i]==1:  ##如果该属性被使用过
				conditional_entropy_list.append(999)  ##给这个被使用过的属性一个很大的条件熵，以免它被选中
				continue
			current_attribute_conditional_entropy=0
			value_list=[]            ##每个属性可能取到的值的不重复列表
			value_amount_dictionary={}    ##每个可能取到的值的出现次数字典（用来算该值的概率）
			##print("当前结点共有",len(dataset),"个数据样本")
			for j in range(len(root.dataset)):
				##print(i,' ',j)
				if root.dataset[j].attributes[i] not in value_list:   ##这个样例的属性i的值没有被放进value_list就放进去，这样保证值不重复
					value_list.append(root.dataset[j].attributes[i])    
					value_amount_dictionary[root.dataset[j].attributes[i]]=1     ##对每一个不一样的值进行计数
				else:
					value_amount_dictionary[root.dataset[j].attributes[i]]+=1
			if len(value_list)==1:
				conditional_entropy_list.append(origin_entropy)
				continue
			else:
				whether_all_attributes_same=0
			for j in range(len(value_list)):  ##遍历所有可能取到的值
				p1 = float(value_amount_dictionary[value_list[j]]/len(root.dataset))  ##属性i取值为value_list[j]这个值的概率
				p2= 0    ##用来计算对于属性i的取值value_list[j]来说熵是多少  即对属性i取值为value_list[j]的情况下的-p*log(p)求和
				tmp_label_amount_dictionary={}
				label_probability_list=[]
				cnt=0
				for k in range(len(root.dataset)):
					if root.dataset[k].attributes[i]==value_list[j]:  ##计算所有样本中值为value_list[j]的总数，作为计算-p*log(p)中的p时的分母
						cnt+=1
				for k in range(len(label_list)):
					tmp_label_amount_dictionary[label_list[k]]=0   ##初始化属性i的取值value_list[j]中各个label的总数
				for k in range(len(label_list)):   ##遍历各种可能出现的label
					for l in range(len(root.dataset)):   ##遍历全部样本
						if root.dataset[l].attributes[i]==value_list[j]:
							if root.dataset[l].label == label_list[k]:    ##如果样本l的label值等于label_list[k]
								tmp_label_amount_dictionary[label_list[k]]+=1     ##对在值等于value_list[j]的情况下，标签为label_list[k]出现的次数进行计数，作为计算-p*log(p)中的p时的分子
					label_probability_list.append(float(tmp_label_amount_dictionary[label_list[k]]/cnt))
				for k in range(len(label_list)):
					if label_probability_list[k]!=0:
						p2+=-label_probability_list[k]*float(math.log(label_probability_list[k])/math.log(2))  ##即对属性i取值为value_list[j]的情况下的-p*log(p)求和
				current_attribute_conditional_entropy+=p1*p2    ##p1和p2的积就是属性i的条件熵 
			conditional_entropy_list.append(current_attribute_conditional_entropy)
		if whether_all_attributes_same:    ##所有数据在所有属性取值相同
			return -3
		for i in range(len(root.dataset[0].attributes)):
			ID3_list.append(origin_entropy-conditional_entropy_list[i])
		print(ID3_list)
		return conditional_entropy_list.index(min(conditional_entropy_list))  ##由于对于根节点，数据集的原始熵都是相同的，所以只要找出条件熵的最小值的索引（对应属性的索引）
	if method==2:  #C4.5
		origin_entropy=0
		conditional_entropy_list=[]
		splitinfo_list=[]
		infogainratio_list=[]
		label_list=[]
		label_amount_dictionary={}
		whether_label_same=1
		whether_all_attributes_same=1
		##########################计算origin_entropy##########################
		for i in range(len(root.dataset)):
			if root.dataset[i].label not in label_list:
				label_list.append(root.dataset[i].label)
				label_amount_dictionary[root.dataset[i].label]=1
			else:
				label_amount_dictionary[root.dataset[i].label]+=1
		for i in range(len(label_list)):
			p = float(label_amount_dictionary[label_list[i]]/len(root.dataset))
			origin_entropy+=float((-p)*(math.log(p))/math.log(2))
		#######################################################################
		if len(root.dataset)==0:
			return -1
		for i in range(len(root.dataset)-1):
			if root.dataset[i].label!=root.dataset[i+1].label:
				whether_label_same=0
				break
		if whether_label_same:
			return -2
		if len(root.dataset[0].attributes)==numpy.sum(root.mark):
			return -3
		##print("目前数据集共有：",len(dataset[0].attributes),"个属性")
		for i in range(len(root.dataset[0].attributes)):  ##遍历所有属性
			##print(i)
			if root.mark[i]==1:  ##如果该属性被使用过
				conditional_entropy_list.append(999)  ##给这个被使用过的属性一个很大的条件熵，以免它被选中
				splitinfo_list.append(999)  ##给这个被使用过的属性一个很大的特征熵，以免它被选中
				continue
			current_attribute_splitinfo=0
			current_attribute_conditional_entropy=0
			value_list=[]            ##每个属性可能取到的值的不重复列表
			value_amount_dictionary={}    ##每个可能取到的值的出现次数字典（用来算该值的概率）
			##print("当前结点共有",len(dataset),"个数据样本")
			for j in range(len(root.dataset)):
				##print(i,' ',j)
				if root.dataset[j].attributes[i] not in value_list:   ##这个样例的属性i的值没有被放进value_list就放进去，这样保证值不重复
					value_list.append(root.dataset[j].attributes[i])    
					value_amount_dictionary[root.dataset[j].attributes[i]]=1     ##对每一个不一样的值进行计数
				else:
					value_amount_dictionary[root.dataset[j].attributes[i]]+=1
			for j in range(len(value_list)):
				p = float(value_amount_dictionary[value_list[j]]/len(root.dataset))
				current_attribute_splitinfo+=float((-p)*(math.log(p))/math.log(2))
			if len(value_list)==1:
				splitinfo_list.append(999)
				conditional_entropy_list.append(origin_entropy)    ##如果该属性只有一种取值，那么splitinfo将为0，条件熵等于原始熵
				continue
			else:
				whether_all_attributes_same=0
			splitinfo_list.append(current_attribute_splitinfo)
			for j in range(len(value_list)):  ##遍历所有可能取到的值
				p1 = float(value_amount_dictionary[value_list[j]]/len(root.dataset))  ##属性i取值为value_list[j]这个值的概率
				p2= 0    ##用来计算对于属性i的取值value_list[j]来说熵是多少  即对属性i取值为value_list[j]的情况下的-p*log(p)求和
				tmp_label_amount_dictionary={}
				label_probability_list=[]
				cnt=0
				for k in range(len(root.dataset)):
					if root.dataset[k].attributes[i]==value_list[j]:  ##计算所有样本中值为value_list[j]的总数，作为计算-p*log(p)中的p时的分母
						cnt+=1
				for k in range(len(label_list)):
					tmp_label_amount_dictionary[label_list[k]]=0   ##初始化属性i的取值value_list[j]中各个label的总数
				for k in range(len(label_list)):   ##遍历各种可能出现的label
					for l in range(len(root.dataset)):   ##遍历全部样本
						if root.dataset[l].attributes[i]==value_list[j]:
							if root.dataset[l].label == label_list[k]:    ##如果样本l的label值等于label_list[k]
								tmp_label_amount_dictionary[label_list[k]]+=1     ##对在值等于value_list[j]的情况下，标签为label_list[k]出现的次数进行计数，作为计算-p*log(p)中的p时的分子
					label_probability_list.append(float(tmp_label_amount_dictionary[label_list[k]]/cnt))
				for k in range(len(label_list)):
					if label_probability_list[k]!=0:
						p2+=-label_probability_list[k]*float(math.log(label_probability_list[k])/math.log(2))  ##即对属性i取值为value_list[j]的情况下的-p*log(p)求和
				current_attribute_conditional_entropy+=p1*p2    ##p1和p2的积就是属性i的条件熵 
			conditional_entropy_list.append(current_attribute_conditional_entropy)
		for i in range(len(root.dataset[0].attributes)):
			infogainratio_list.append((origin_entropy-conditional_entropy_list[i])/splitinfo_list[i])
		if whether_all_attributes_same:
			return -3
		#print(infogainratio_list)
		return infogainratio_list.index(max(infogainratio_list))  #找出所有属性的信息增益率中的最大值的索引（对应属性的索引）
	if method==3:  #GINI
		GINI_list=[]
		label_list=[]
		label_amount_dictionary={}
		whether_label_same=1
		whether_all_attributes_same=1
		if len(root.dataset)==0:
			return -1
		for i in range(len(root.dataset)-1):
			if root.dataset[i].label!=root.dataset[i+1].label:
				whether_label_same=0
				break
		if whether_label_same:
			return -2
		if len(root.dataset[0].attributes)==numpy.sum(root.mark):
			return -3
		##print("目前数据集共有：",len(dataset[0].attributes),"个属性")
		for i in range(len(root.dataset)):
			if root.dataset[i].label not in label_list:
				label_list.append(root.dataset[i].label)
		for i in range(len(root.dataset[0].attributes)):  ##遍历所有属性
			if root.mark[i]==1:  ##如果该属性被使用过
				GINI_list.append(999)  ##给这个被使用过的属性一个很大的GINI指数，以免它被选中
				continue
			current_attribute_GINI=0
			value_list=[]            ##每个属性可能取到的值的不重复列表
			value_amount_dictionary={}    ##每个可能取到的值的出现次数字典（用来算该值的概率）
			##print("当前结点共有",len(dataset),"个数据样本")
			for j in range(len(root.dataset)):
				##print(i,' ',j)
				if root.dataset[j].attributes[i] not in value_list:   ##这个样例的属性i的值没有被放进value_list就放进去，这样保证值不重复
					value_list.append(root.dataset[j].attributes[i])    
					value_amount_dictionary[root.dataset[j].attributes[i]]=1     ##对每一个不一样的值进行计数
				else:
					value_amount_dictionary[root.dataset[j].attributes[i]]+=1
			if len(value_list)==1:
				GINI_list.append(999)
				continue
			else:
				whether_all_attributes_same=0
			for j in range(len(value_list)):  ##遍历所有可能取到的值
				p1 = float(value_amount_dictionary[value_list[j]]/len(root.dataset))  ##属性i取值为value_list[j]这个值的概率
				p2= 0    ##用来计算对于属性i的取值value_list[j]来说条件GINI是多少  
				tmp_label_amount_dictionary={}
				label_probability_list=[]
				cnt=0
				for k in range(len(root.dataset)):
					if root.dataset[k].attributes[i]==value_list[j]:  ##计算所有样本中值为value_list[j]的总数
						cnt+=1
				for k in range(len(label_list)):
					tmp_label_amount_dictionary[label_list[k]]=0   ##初始化属性i的取值value_list[j]中各个label的总数
				for k in range(len(label_list)):   ##遍历各种可能出现的label
					for l in range(len(root.dataset)):   ##遍历全部样本
						if root.dataset[l].attributes[i]==value_list[j]:
							if root.dataset[l].label == label_list[k]:    ##如果样本l的label值等于label_list[k]
								tmp_label_amount_dictionary[label_list[k]]+=1     ##对在值等于value_list[j]的情况下，标签为label_list[k]出现的次数
					label_probability_list.append(float(tmp_label_amount_dictionary[label_list[k]]/cnt))
				for k in range(len(label_list)):
					p2+=label_probability_list[k]*(1-label_probability_list[k])  ##即对属性i取值为value_list[j]的情况下的p*(1-p)求和
				current_attribute_GINI+=p1*p2    ##p1和p2的积就是属性i的GINI指数
			GINI_list.append(current_attribute_GINI)
		if whether_all_attributes_same:
			return -3
		print(GINI_list)
		return GINI_list.index(min(GINI_list))  ##找出所有属性的GINI指数的最小值的索引（对应属性的索引）
def create_tree(root):
	chosen_attribute=choose_attribute(root,1)
	print("选择了第",chosen_attribute,"个属性")
	if chosen_attribute > -1:   ##不是叶节点，可以继续划分
		root.mark[chosen_attribute]=1    ##标记此属性被使用
		root.attribute_or_decision=chosen_attribute
		chosen_attribute_value_list=[]
		for i in range(len(root.dataset)):
			if root.dataset[i].attributes[chosen_attribute] not in chosen_attribute_value_list:   ##遇到了一个新的值，以为着将要创建一个新的结点
				chosen_attribute_value_list.append(root.dataset[i].attributes[chosen_attribute])
				node = Node()
				node.mark=copy.deepcopy(root.mark)   ##继承父结点的mark
				node.attribute_value=root.dataset[i].attributes[chosen_attribute]
				node.parent=root
				node.level=root.level+1   ##更新层数
				for j in range(i,len(root.dataset)):    ##从当前数据开始往后遍历，遇到这个属性的值相等的数据就加入新结点的数据集
					if root.dataset[j].attributes[chosen_attribute]==root.dataset[i].attributes[chosen_attribute]:
						sample=Sample()
						sample=copy.deepcopy(root.dataset[j])
						node.dataset.append(sample)    ##扩充新结点的数据集
				root.children.append(node)    ##将新的结点作为root的子结点
		for i in range(test_amount):   ##扫完训练集之后再扫一遍测试集，找到测试集中有但是训练集中没有的值
			if test_sample_list[i].attributes[chosen_attribute] not in chosen_attribute_value_list:
				chosen_attribute_value_list.append(test_sample_list[i].attributes[chosen_attribute])
				node = Node()
				node.attribute_value=test_sample_list[i].attributes[chosen_attribute]
				node.parent=root    ##这些结点没有数据集就不用添加数据了
				node.level=root.level+1
				root.children.append(node)
		if len(root.children)!=0:   ##如果当前结点还有子结点
			for i in range(len(root.children)):
				print("当前结点为第",root.children[i].level,"层第",i,"个结点:")
				print("当前数据集:")
				for j in range(len(root.children[i].dataset)):   ##分别以每一个子结点为根节点继续建树
					print(root.children[i].dataset[j].attributes)
				print("当前mark:",root.children[i].mark)
				create_tree(root.children[i])
	if chosen_attribute==-1:    ##当前数据集为空
		lst=[]
		for i in range(len(root.parent.dataset)):
			lst.append(root.parent.dataset[i].label)
		root.attribute_or_decision=Counter(lst).most_common(1)[0][-2]
	if chosen_attribute==-2:    ##所有数据标签相同
		root.attribute_or_decision=root.dataset[0].label
	if chosen_attribute==-3:   ##没有属性可以继续使用或者所有数据所有属性的值都相同
		lst=[]
		for i in range(len(root.dataset)):
			lst.append(root.dataset[i].label)
		root.attribute_or_decision=Counter(lst).most_common(1)[0][-2]   ##投票，找出lst中出现次数最多的（label，出现次数）元组，然后取出label
def cross_validation(train_linebreak_splits,it):
	for i in range(total):
		if i >=it*100 and i<it*100+100:   ##取索引值为it*100 到 it*100+99 的所有数据作为验证集
			attributes_and_label = train_linebreak_splits[i].split(',')
			sample = Sample()
			sample.label=attributes_and_label[-1]
			answer_list.append(sample.label)
			attributes_and_label.pop() ##删掉label 便于sample.attributes的遍历赋值
			for attribute in attributes_and_label:
				sample.attributes.append(attribute)
			test_sample_list.append(sample)     ##构建验证集
		else:
			attributes_and_label = train_linebreak_splits[i].split(',')
			sample = Sample()
			sample.label=attributes_and_label[-1]
			attributes_and_label.pop() ##删掉label 便于sample.attributes的遍历数据赋值
			for attribute in attributes_and_label:
				sample.attributes.append(attribute)
			train_sample_list.append(sample)    ##构建训练集
		
train_amount=0          ##跑结果
test_amount=0
train_sample_list=[]
test_sample_list=[]
validation_result_list=[]
train_set = open("D:\\train_set1.csv","r")
test_set = open("D:\\validation_set2.csv","r")
# train_set = open("D:\\train.csv","r")
# test_set = open("D:\\test.csv","r")
read_data(train_set,test_set)
root = Node()
root.dataset=train_sample_list
print("当前结点为第0层第0个结点")
print("当前数据集:")
for i in range(len(root.dataset)):
	print(root.dataset[i].attributes)
for i in range(len(train_sample_list[0].attributes)):
	root.mark.append(0)
print("当前mark:",root.mark)
create_tree(root)
decision_list=[]
for i in range(test_amount):
	decision_list.append(make_decision(root,test_sample_list[i]))
output=open("D:\\15352225_liuhongji.txt","w")
for i in range(test_amount):
	# output.write(str(decision_list[i]))
	# output.write('\n')
	print(decision_list[i])

          ##交叉验证
# validation_result_list=[]
# train_set = open("D:\\train.csv","r")
# train_string = train_set.read()
# train_linebreak_splits = train_string.split('\n')
# train_linebreak_splits.pop()
# total=len(train_linebreak_splits)
# for i in range(7):
	# answer_list=[]
	# train_sample_list=[]
	# test_sample_list=[]
	# cross_validation(train_linebreak_splits,i)
	# train_amount=len(train_sample_list)
	# test_amount=len(test_sample_list)
	# root = Node()
	# root.dataset=train_sample_list
	# for i in range(len(train_sample_list[0].attributes)):
		# root.mark.append(0)
	# create_tree(root)
	# decision_list=[]
	# for i in range(test_amount):
		# decision_list.append(make_decision(root,test_sample_list[i]))
	# right_amount=0
	# for i in range(test_amount):
		# if answer_list[i]==decision_list[i]:
			# right_amount+=1
	# validation_result_list.append(float(right_amount/test_amount))
# pylab.plot(validation_result_list)
# pylab.show()




