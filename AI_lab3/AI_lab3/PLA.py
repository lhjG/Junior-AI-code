import numpy;
train_amount = 0
test_amount=0
predict_list=[]
def PLA_naive(inputstream1,inputstream2,upper_limit):
    train_string = inputstream1.read()
    test_string = inputstream2.read()
    linebreak_splits = train_string.split('\n')
    linebreak_splits.pop()
    linebreak_splits_1 = test_string.split('\n')
    linebreak_splits_1.pop()
    global train_amount
    train_amount=len(linebreak_splits)
    global test_amount
    test_amount=len(linebreak_splits_1)
    train_matrix = numpy.ones((train_amount,8),float)
    test_matrix = numpy.ones((test_amount,8),float)
    for i in range(train_amount):
        features_and_label = linebreak_splits[i].split(',')
        for j in range(7):
            train_matrix[i][j+1]=float(features_and_label[j])
    for i in range(test_amount):
        features_and_label_1 = linebreak_splits_1[i].split(',')
        for j in range(6):
            test_matrix[i][j+1]=float(features_and_label_1[j])


    w = numpy.ones(7)##################################初始化w
    ##w=[-3,2,2,0]
    for i in range(upper_limit):
        for j in range(train_amount):
            temp=numpy.zeros(7)
            for k in range(7):
                temp[k]=train_matrix[j][k]    ###############向量化train_matrix的一行方便下面做点乘
                ans = numpy.sign(numpy.dot(temp,w)) #######################向量点乘
            if ans != train_matrix[j][7]:   ########################预测错误
			    ##print("在第",j+1,"个训练样本处预测错误,此时的w:",w,"错误样本的特征向量:",temp,"正确答案:",test_matrix[j][4])
                w = w+train_matrix[j][7]*temp   ##################更新w，w+=label*特征向量
                break              ##################只要更新了w就从第一个训练样本开始从头遍历
    print("最终使用的w：",w)
    for i in range(test_amount):
        temp=numpy.zeros(7)
        for j in range(7):
            temp[j]=test_matrix[i][j]
	    ##print("第",i+1,"个预测文本的特征向量：",temp)
        predict_list.append(numpy.sign(numpy.dot(w,temp)))
    ##output = open("D:\\validation概率预测结果.csv","w")
    TP,TN,FP,FN=0,0,0,0
    for i in range(test_amount):
        ##output.write(str(predict_list[i]))
        ##output.write('\n')
        print("第",i+1,"个样例的预测结果",predict_list[i])
        if predict_list[i]==test_matrix[i][7]:
		    ##print("预测正确！")
            if predict_list[i]==1:
                TP+=1
            else :
                TN+=1
        else:
		    ##print("预测错误！")
            if predict_list[i]==1:
                FP+=1
            else:
                FN+=1
    Accuracy=(TP+TN)/(TP+TN+FP+FN)
    if (TP+FN)!=0:
        Recall=TP/(TP+FN)
    else:
        Recall=0
    if (TP+FP)!=0:
        Precision=TP/(TP+FP)
    else:
        Precision=0
    if (Precision+Recall)!=0:
        F1 = 2*Precision*Recall/(Precision+Recall)
    else:
        F1=0
    '''print("TP:",TP,"TN:",TN,"FP:",FP,"FN:",FN)
    print("Accuracy：",Accuracy)
    print("Recall：",Recall)
    print("Precision：",Precision)
    print("F1:",F1)'''
    #output.close()
def PLA_pocket(inputstream1,inputstream2,upper_limit):
    train_string = inputstream1.read()
    test_string = inputstream2.read()
    linebreak_splits = train_string.split('\n')
    linebreak_splits.pop()
    linebreak_splits_1 = test_string.split('\n')
    linebreak_splits_1.pop()
    global train_amount
    train_amount=len(linebreak_splits)
    global test_amount
    test_amount=len(linebreak_splits_1)
    train_matrix = numpy.ones((train_amount,67),float)
    test_matrix = numpy.ones((test_amount,67),float)
    for i in range(train_amount):
        features_and_label = linebreak_splits[i].split(',')
        for j in range(66):
            train_matrix[i][j+1]=float(features_and_label[j])
    for i in range(test_amount):
        features_and_label_1 = linebreak_splits_1[i].split(',')
        for j in range(66):
            test_matrix[i][j+1]=float(features_and_label_1[j])
    ##w_in_pocket=[-3,2,2,0]
    w_in_pocket = numpy.ones(66)################################初始化口袋内的w


 ##################################计算初始化口袋内w对训练集的预测正确率########################
    predict_list_1=[]   
    for i in range(train_amount):
        temp=numpy.zeros(66)
        for j in range(66):
            temp[j]=train_matrix[i][j]
        predict_list_1.append(numpy.sign(numpy.dot(w_in_pocket,temp)))
    right_amount=0
    for i in range(train_amount):
        if predict_list_1[i]==train_matrix[i][66]:
            right_amount+=1
    right_rate_p = right_amount/train_amount
###############################################################################################


    w=numpy.ones(66) ################初始化普通的w
    ##print("迭代上限:",upper_limit)
    for i in range(upper_limit): #######################迭代上限
        ##print("当前迭代次数：",i+1)
        for j in range(train_amount):
            temp=numpy.zeros(66)
            for k in range(66):
                temp[k]=train_matrix[j][k]    #######################向量化train_matrix的一行，方便下面做向量点乘
            ans = numpy.sign(numpy.dot(temp,w))#################普通w和特征向量点乘的符号
            if ans != train_matrix[j][66]:   ##################预测错误
                w = w+train_matrix[j][66]*temp    ################更新普通的w
                ##print("在第",j+1,"个训练样本处预测错误,更新后的普通w为:",w)
                ##print("当前口袋中的w的预测正确率:",right_rate_p)
    ##########################计算普通w的预测正确率（也要从头遍历整个训练集）####################################
                predict_list_2=[]
                for i in range(train_amount):
                    temp=numpy.zeros(66)
                    for j in range(66):
                        temp[j]=train_matrix[i][j]
                    predict_list_2.append(numpy.sign(numpy.dot(w,temp)))
                right_amount=0
                for i in range(train_amount):
                    if predict_list_2[i]==train_matrix[i][66]:
                        right_amount+=1
                right_rate_w = right_amount/train_amount
                ##print("当前w的预测正确率:",right_rate_w)
                if right_rate_w>right_rate_p:   #######################如果当前普通的w的正确率高，就替换掉口袋中的w和口袋中w的正确率
                    w_in_pocket=w
                    right_rate_p=right_rate_w
                    ##print("口袋中的w更新为：",w_in_pocket)
                break ##################################只要遇到预测错误就重头开始遍历训练集，迭代次数++
              
    #output = open("D:\\validation概率预测结果.csv","w")
    ##print(w_in_pocket)
    for i in range(test_amount):
        temp=numpy.zeros(66)
        for j in range(66):
            temp[j]=test_matrix[i][j]
        predict_list.append(numpy.sign(numpy.dot(w_in_pocket,temp)))
        ##print(numpy.sign(numpy.dot(w_in_pocket,temp)))
    TP,TN,FP,FN=0,0,0,0
    for i in range(test_amount):
        ##output.write(str(predict_list[i]))
        ##output.write('\n')
        if predict_list[i]==test_matrix[i][66]:
            if predict_list[i]==1:
                TP+=1
            else:
                TN+=1
        else:
            if predict_list[i]==1:
                FP+=1
            else:
                FN+=1
    Accuracy=(TP+TN)/(TP+TN+FP+FN)
    if (TP+FN)!=0:
        Recall=TP/(TP+FN)
    else:
        Recall=0
    if (TP+FP)!=0:
        Precision=TP/(TP+FP)
    else:
        Precision=0
    if (Precision+Recall)!=0:
        F1 = 2*Precision*Recall/(Precision+Recall)
    else:
        F1=0
    print("TP:",TP,"TN:",TN,"FP:",FP,"FN:",FN)
    print("Accuracy：",Accuracy)
    print("Recall：",Recall)
    print("Precision：",Precision)
    print("F1:",F1)
    ##output.close()
train_set = open("D:\\train_set1.csv","r")
test_set = open("D:\\validation_set2.csv","r")
PLA_naive(train_set,test_set,10000)

