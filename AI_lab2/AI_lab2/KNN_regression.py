import numpy;
import csv;
import math;
words_amount=0
words = []
emotions_of_validation=[]
emotion_probabilities_of_train_dictionary={}
test_emotion_probabilities=[]
item_emotion_probabilities=[0,0,0,0,0,0,float]
def use_One_hot(inputstream1,inputstream2):
    train_string = inputstream1.read()
    linebreak_splits = train_string.split('\n')
    linebreak_splits.pop()
    del linebreak_splits[0]
    sentences_and_emotion_probabilities = []
    sentences=[]
    train_amount = 0
    for linebreak_split in linebreak_splits:
        train_amount+=1
        sentence_and_emotion_probabilities = linebreak_split.split(',')
        emotion_probabilities_of_train_dictionary.setdefault(train_amount-1,[]).append(sentence_and_emotion_probabilities[1])
        emotion_probabilities_of_train_dictionary.setdefault(train_amount-1,[]).append(sentence_and_emotion_probabilities[2])
        emotion_probabilities_of_train_dictionary.setdefault(train_amount-1,[]).append(sentence_and_emotion_probabilities[3])
        emotion_probabilities_of_train_dictionary.setdefault(train_amount-1,[]).append(sentence_and_emotion_probabilities[4])
        emotion_probabilities_of_train_dictionary.setdefault(train_amount-1,[]).append(sentence_and_emotion_probabilities[5])
        emotion_probabilities_of_train_dictionary.setdefault(train_amount-1,[]).append(sentence_and_emotion_probabilities[6])
        sentences_and_emotion_probabilities.append(sentence_and_emotion_probabilities)
    for sentence_and_emotion_probabilities in sentences_and_emotion_probabilities:
        sentence = sentence_and_emotion_probabilities[0].split(' ')
        sentences.append(sentence)
        for each_word_in_sentence in sentence:
            bool_if_word_exist = each_word_in_sentence in words
            if bool_if_word_exist==False:
                global words_amount
                words_amount+=1
                words.append(each_word_in_sentence)   
    One_hot = numpy.zeros((train_amount,words_amount))
    for i in range(train_amount):
        for j in range(words_amount):
            bool_if_word_in_this_train = words[j] in sentences[i]
            if bool_if_word_in_this_train:
                One_hot[i][j]=1
    train_string_1 = inputstream2.read()
    linebreak_splits_1 = train_string_1.split('\n')
    linebreak_splits_1.pop()
    del linebreak_splits_1[0]
    test_amount=len(linebreak_splits_1)
    emotion_probabilities_of_train_dictionary_1={}
    test_matrix=numpy.zeros((train_amount,words_amount))
    for j in range(test_amount):
        sentence_and_emotion_probabilities_1 = linebreak_splits_1[j].split(',')
        emotion_probabilities_of_train_dictionary_1.setdefault(j,[]).append(sentence_and_emotion_probabilities_1[1])
        emotion_probabilities_of_train_dictionary_1.setdefault(j,[]).append(sentence_and_emotion_probabilities_1[2])
        emotion_probabilities_of_train_dictionary_1.setdefault(j,[]).append(sentence_and_emotion_probabilities_1[3])
        emotion_probabilities_of_train_dictionary_1.setdefault(j,[]).append(sentence_and_emotion_probabilities_1[4])
        emotion_probabilities_of_train_dictionary_1.setdefault(j,[]).append(sentence_and_emotion_probabilities_1[5])
        emotion_probabilities_of_train_dictionary_1.setdefault(j,[]).append(sentence_and_emotion_probabilities_1[6])
        sentence_1 = sentence_and_emotion_probabilities_1[0].split(' ')
        for i in range(words_amount):
            if words[i] in sentence_1:
               test_matrix[j][i]=1   
    KNN(25,3,One_hot,test_matrix)
def use_TFIDF(inputstream1,inputstream2):
    train_string = inputstream1.read()
    linebreak_splits = train_string.split('\n')
    linebreak_splits.pop()
    del linebreak_splits[0]
    sentences_and_emotion_probabilities = []
    sentences=[]
    train_amount = 0
    text_length_list=[]
    row_text_word_appeartimes_dictionary_dictionary={}
    rows_cnt=0
    word_appeartimes_in_all_text_dictionary={}
    for linebreak_split in linebreak_splits:
        train_amount+=1
        sentence_and_emotion_probabilities = linebreak_split.split(',')
        emotion_probabilities_of_train_dictionary.setdefault(train_amount-1,[]).append(sentence_and_emotion_probabilities[1])
        emotion_probabilities_of_train_dictionary.setdefault(train_amount-1,[]).append(sentence_and_emotion_probabilities[2])
        emotion_probabilities_of_train_dictionary.setdefault(train_amount-1,[]).append(sentence_and_emotion_probabilities[3])
        emotion_probabilities_of_train_dictionary.setdefault(train_amount-1,[]).append(sentence_and_emotion_probabilities[4])
        emotion_probabilities_of_train_dictionary.setdefault(train_amount-1,[]).append(sentence_and_emotion_probabilities[5])
        emotion_probabilities_of_train_dictionary.setdefault(train_amount-1,[]).append(sentence_and_emotion_probabilities[6])
        sentences_and_emotion_probabilities.append(sentence_and_emotion_probabilities)
    for sentence_and_emotion_probabilities in sentences_and_emotion_probabilities:
        rows_cnt+=1
        word_appertimes_dictionary={}
        sentence = sentence_and_emotion_probabilities[0].split(' ')
        text_length_list.append(len(sentence))
        sentences.append(sentence)
        for each_word_in_sentence in sentence:
            bool_if_word_exist = each_word_in_sentence in words
            if bool_if_word_exist==False:
                global words_amount
                words_amount+=1
                words.append(each_word_in_sentence)   
            if each_word_in_sentence in word_appertimes_dictionary:
                word_appertimes_dictionary[each_word_in_sentence]+=1
            else:
                word_appertimes_dictionary[each_word_in_sentence]=1
        row_text_word_appeartimes_dictionary_dictionary[rows_cnt-1] = word_appertimes_dictionary
    
    for word in words:
       word_appeartimes_in_all_text_dictionary[word]=0
    for sentence in sentences:
        for word in words:
            if word in sentence:
                word_appeartimes_in_all_text_dictionary[word]+=1
    TFIDF = numpy.zeros((train_amount,words_amount))
    for i in range(train_amount):
        for j in range(words_amount):
            if words[j] in row_text_word_appeartimes_dictionary_dictionary[i].keys():
                TFIDF[i][j]=(row_text_word_appeartimes_dictionary_dictionary[i][words[j]]/text_length_list[i])*math.log(train_amount/(1 + word_appeartimes_in_all_text_dictionary[words[j]]))
            else:
                TFIDF[i][j]=0
    '''output = open("D:\\matrix.txt","w")
    output.write("这是训练集的TFIDF矩阵！！！！！！！！！！！！！！！！")
    for i in range(len(TFIDF)):
        output.write('\n')
        for j in range(len(TFIDF[1])):
            output.write(str(TFIDF[i][j]))
            output.write(' ')'''
    train_string_1 = inputstream2.read()
    row_text_word_appeartimes_dictionary_dictionary_1={}
    word_appeartimes_in_all_text_dictionary_1={}
    text_length_list_1=[]
    linebreak_splits_1 = train_string_1.split('\n')
    linebreak_splits_1.pop()
    del linebreak_splits_1[0]
    test_amount=len(linebreak_splits_1)
    emotion_probabilities_of_train_dictionary_1={}
    test_matrix=numpy.zeros((train_amount,words_amount))
    for j in range(test_amount):
        word_appertimes_dictionary_1={}
        sentence_and_emotion_probabilities_1 = linebreak_splits_1[j].split(',')
        emotion_probabilities_of_train_dictionary_1.setdefault(j,[]).append(sentence_and_emotion_probabilities_1[1])
        emotion_probabilities_of_train_dictionary_1.setdefault(j,[]).append(sentence_and_emotion_probabilities_1[2])
        emotion_probabilities_of_train_dictionary_1.setdefault(j,[]).append(sentence_and_emotion_probabilities_1[3])
        emotion_probabilities_of_train_dictionary_1.setdefault(j,[]).append(sentence_and_emotion_probabilities_1[4])
        emotion_probabilities_of_train_dictionary_1.setdefault(j,[]).append(sentence_and_emotion_probabilities_1[5])
        emotion_probabilities_of_train_dictionary_1.setdefault(j,[]).append(sentence_and_emotion_probabilities_1[6])
        sentence_1 = sentence_and_emotion_probabilities_1[0].split(' ')
        for word in words:
            word_appeartimes_in_all_text_dictionary_1[word]=0
        for word in words:
            if word in sentence_1:
                word_appeartimes_in_all_text_dictionary_1[word]+=1
        for each_word_in_sentence_1 in sentence_1:
            if each_word_in_sentence_1 in word_appertimes_dictionary_1:
                word_appertimes_dictionary_1[each_word_in_sentence_1]+=1
            else:
                word_appertimes_dictionary_1[each_word_in_sentence_1]=1
        row_text_word_appeartimes_dictionary_dictionary_1[j]=word_appertimes_dictionary_1
        text_length_list_1.append(len(sentence_1))
        for i in range(words_amount):
            if words[i] in row_text_word_appeartimes_dictionary_dictionary_1[j].keys():
                test_matrix[j][i]=(row_text_word_appeartimes_dictionary_dictionary_1[j][words[i]]/text_length_list_1[j])*math.log(test_amount/(1+word_appeartimes_in_all_text_dictionary_1[words[i]]))  
            else:
                test_matrix[j][i]=0
    KNN(13,3,TFIDF,test_matrix)
def KNN(k,length_kind,train_matrix,test_matrix):
    output = open("D:\\validation概率预测结果.csv","w")
    for i in range(len(test_matrix)):
        row_distance_dictionary={}
        distances_dictionary = {}
        distances_list=[]
        rows_of_selected=[]
################################################################################依次计算预测样本[i]与训练文本[j]的距离并记录下来
        for j in range(len(train_matrix)):
            if length_kind==1:                   #####################################曼哈顿距离
                distance = numpy.sum(numpy.abs(test_matrix[i]-train_matrix[j]))
            if length_kind==2:                   ####################################欧式距离
                distance = numpy.sqrt(numpy.sum(numpy.square(test_matrix[i]-train_matrix[j])))
            if length_kind==3:                   #####################################余弦相似度
                length_product = (numpy.sqrt(test_matrix[i].dot(test_matrix[i])))*(numpy.sqrt(train_matrix[j]).dot(train_matrix[j]))
                if length_product != 0:
                    distance = test_matrix[i].dot(train_matrix[j])/ length_product
                else:
                    distance = 1
            row_distance_dictionary[j]=distance
            distances_dictionary.setdefault(distance,[]).append(j)
            if distance not in distances_list:
                distances_list.append(distance)
        distances_list.sort()                    #################################距离排序
####################################################################################记录排序好的各个距离对应的行号
        for distance in distances_list:
            for row_number in distances_dictionary[distance]:
                rows_of_selected.append(row_number)
####################################################################找出距离最近的K个向量，用距离的倒数做权值乘以对应向量对应情感的概率
        for r in range(k):
            if length_kind==3:                     ##################如果是余弦相似度，应该找出值最大的K个，其余的距离应该是最小的K个
                if row_distance_dictionary[rows_of_selected[r]] != 0:  ########################以防距离为0，距离为0就直接赋值，不用再除以距离作为权重
                    item_emotion_probabilities[0]+=(float(emotion_probabilities_of_train_dictionary[rows_of_selected[-(r+1)]][0])/row_distance_dictionary[rows_of_selected[-(r+1)]])
                    item_emotion_probabilities[1]+=(float(emotion_probabilities_of_train_dictionary[rows_of_selected[-(r+1)]][1])/row_distance_dictionary[rows_of_selected[-(r+1)]])
                    item_emotion_probabilities[2]+=(float(emotion_probabilities_of_train_dictionary[rows_of_selected[-(r+1)]][2])/row_distance_dictionary[rows_of_selected[-(r+1)]])
                    item_emotion_probabilities[3]+=(float(emotion_probabilities_of_train_dictionary[rows_of_selected[-(r+1)]][3])/row_distance_dictionary[rows_of_selected[-(r+1)]])
                    item_emotion_probabilities[4]+=(float(emotion_probabilities_of_train_dictionary[rows_of_selected[-(r+1)]][4])/row_distance_dictionary[rows_of_selected[-(r+1)]])
                    item_emotion_probabilities[5]+=(float(emotion_probabilities_of_train_dictionary[rows_of_selected[-(r+1)]][5])/row_distance_dictionary[rows_of_selected[-(r+1)]])
                else:
                    item_emotion_probabilities[0]=float(emotion_probabilities_of_train_dictionary[rows_of_selected[-(r+1)]][0])
                    item_emotion_probabilities[1]=float(emotion_probabilities_of_train_dictionary[rows_of_selected[-(r+1)]][1])
                    item_emotion_probabilities[2]=float(emotion_probabilities_of_train_dictionary[rows_of_selected[-(r+1)]][2])
                    item_emotion_probabilities[3]=float(emotion_probabilities_of_train_dictionary[rows_of_selected[-(r+1)]][3])
                    item_emotion_probabilities[4]=float(emotion_probabilities_of_train_dictionary[rows_of_selected[-(r+1)]][4])
                    item_emotion_probabilities[5]=float(emotion_probabilities_of_train_dictionary[rows_of_selected[-(r+1)]][5])
            else:                             ###################################曼哈顿和欧式距离
                if row_distance_dictionary[rows_of_selected[r]] != 0:
                    item_emotion_probabilities[0]+=(float(emotion_probabilities_of_train_dictionary[rows_of_selected[r]][0])/row_distance_dictionary[rows_of_selected[r]])
                    item_emotion_probabilities[1]+=(float(emotion_probabilities_of_train_dictionary[rows_of_selected[r]][1])/row_distance_dictionary[rows_of_selected[r]])
                    item_emotion_probabilities[2]+=(float(emotion_probabilities_of_train_dictionary[rows_of_selected[r]][2])/row_distance_dictionary[rows_of_selected[r]])
                    item_emotion_probabilities[3]+=(float(emotion_probabilities_of_train_dictionary[rows_of_selected[r]][3])/row_distance_dictionary[rows_of_selected[r]])
                    item_emotion_probabilities[4]+=(float(emotion_probabilities_of_train_dictionary[rows_of_selected[r]][4])/row_distance_dictionary[rows_of_selected[r]])
                    item_emotion_probabilities[5]+=(float(emotion_probabilities_of_train_dictionary[rows_of_selected[r]][5])/row_distance_dictionary[rows_of_selected[r]])
                else:
                    item_emotion_probabilities[0]=float(emotion_probabilities_of_train_dictionary[rows_of_selected[r]][0])
                    item_emotion_probabilities[1]=float(emotion_probabilities_of_train_dictionary[rows_of_selected[r]][1])
                    item_emotion_probabilities[2]=float(emotion_probabilities_of_train_dictionary[rows_of_selected[r]][2])
                    item_emotion_probabilities[3]=float(emotion_probabilities_of_train_dictionary[rows_of_selected[r]][3])
                    item_emotion_probabilities[4]=float(emotion_probabilities_of_train_dictionary[rows_of_selected[r]][4])
                    item_emotion_probabilities[5]=float(emotion_probabilities_of_train_dictionary[rows_of_selected[r]][5])

##########################################################################归一化，使得各概率之和为1
            probability_sum = item_emotion_probabilities[0]+item_emotion_probabilities[1]+item_emotion_probabilities[2]+item_emotion_probabilities[3]+item_emotion_probabilities[4]+item_emotion_probabilities[5]
            if probability_sum !=0:
                item_emotion_probabilities[0]/=probability_sum
                item_emotion_probabilities[1]/=probability_sum
                item_emotion_probabilities[2]/=probability_sum
                item_emotion_probabilities[3]/=probability_sum
                item_emotion_probabilities[4]/=probability_sum
                item_emotion_probabilities[5]/=probability_sum
#########################################################################
        test_emotion_probabilities.append(item_emotion_probabilities)
        output.write(str(item_emotion_probabilities[0]))
        output.write(',')
        output.write(str(item_emotion_probabilities[1]))
        output.write(',')
        output.write(str(item_emotion_probabilities[2]))
        output.write(',')
        output.write(str(item_emotion_probabilities[3]))
        output.write(',')
        output.write(str(item_emotion_probabilities[4]))
        output.write(',')
        output.write(str(item_emotion_probabilities[5]))
        output.write('\n')
    output.close()




'''output = open("D:\\matrix.txt","w")
output.write("这是训练集的One-hot矩阵！！！！！！！！！！！！！！！！")
for i in range(len(train_set_One_hot)):
    output.write('\n')
    for j in range(len(train_set_One_hot[1])):
        output.write(str(train_set_One_hot[i][j]))
    for j in range(6):
        output.write(str(emotion_probabilities_of_train_dictionary[i][j]))
        output.write(' ')
output.write('\n')
output.write("这是验证集的One-hot矩阵！！！！！！！！！！！！！！！！")
for i in range(len(test_matrix)):
    output.write('\n')
    for j in range(len(test_matrix[1])):
        output.write(str(test_matrix[i][j]))
    for j in range(6):
        output.write(str(emotion_probabilities_of_train_dictionary_1[i][j]))
        output.write(' ')'''
train_set = open("D:\\train_set1.csv","r")
test_set = open("D:\\validation_set2.csv","r")
use_One_hot(train_set,test_set)


