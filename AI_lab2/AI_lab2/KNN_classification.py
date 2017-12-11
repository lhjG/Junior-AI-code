import numpy;
import math;
emotions_of_train=[]
words = []
words_amount=0
emotions_of_validation=[]
def use_One_hot(inputstream1,inputstream2,k,length_kind):
    train_string = inputstream1.read()
    linebreak_splits = train_string.split('\n')
    linebreak_splits.pop()
    del linebreak_splits[0]
    sentence_and_emotions = []
    sentences=[]
    train_amount = 0
    for linebreak_split in linebreak_splits:
        train_amount+=1
        sentence_and_emotion = linebreak_split.split(',')
        emotions_of_train.append(sentence_and_emotion[1])
        sentence_and_emotions.append(sentence_and_emotion)
    for sentence_and_emotion in sentence_and_emotions:
        sentence = sentence_and_emotion[0].split(' ')
        sentences.append(sentence)
        for each_word_in_sentence in sentence:
            bool_if_word_exist = each_word_in_sentence in words
            if bool_if_word_exist==False:
                global words_amount
                words_amount+=1
                words.append(each_word_in_sentence)   
    One_hot = numpy.zeros((train_amount,words_amount),int)
    for i in range(train_amount):
        for j in range(words_amount):
            bool_if_word_in_this_train = words[j] in sentences[i]
            if bool_if_word_in_this_train:
                One_hot[i][j]=1
    train_string_1 = inputstream2.read()
    linebreak_splits_1 = train_string_1.split('\n')
    linebreak_splits_1.pop()
    del linebreak_splits_1[0]
    global test_amount
    test_amount=len(linebreak_splits_1)
    test_matrix = numpy.zeros((test_amount,words_amount),int)
    for j in range(test_amount):
        sentence_and_emotion_1 = linebreak_splits_1[j].split(',')
        emotions_of_validation.append(sentence_and_emotion_1[1])
        sentence_1 = sentence_and_emotion_1[0].split(' ')
        for i in range(words_amount):
            if words[i] in sentence_1:
               test_matrix[j][i]=1
    KNN(k,length_kind,One_hot,test_matrix)
def use_TFIDF(inputstream1,inputstream2,k,length_kind):
    train_string = inputstream1.read()
    linebreak_splits = train_string.split('\n')
    linebreak_splits.pop()
    del linebreak_splits[0]
    sentence_and_emotions = []
    sentences=[]
    row_text_appeartimes_dictionary_dictionary={}
    row_text_appeartimes_dictionary_dictionary_1={}
    word_appeartimes_in_all_text_dictionary={}
    word_appeartimes_in_all_text_dictionary_1={}
    text_length_list=[]
    text_length_list_1=[]
    train_amount = 0
    rows_cnt=0
    for linebreak_split in linebreak_splits:
        train_amount+=1
        sentence_and_emotion = linebreak_split.split(',')
        emotions_of_train.append(sentence_and_emotion[1])
        sentence_and_emotions.append(sentence_and_emotion)
    for sentence_and_emotion in sentence_and_emotions:
        rows_cnt+=1
        word_appeartimes_dictionary={}
        sentence = sentence_and_emotion[0].split(' ')
        text_length_list.append(len(sentence))
        sentences.append(sentence)
        for each_word_in_sentence in sentence:
            if each_word_in_sentence in word_appeartimes_dictionary.keys():
                word_appeartimes_dictionary[each_word_in_sentence]+=1
            else:
                word_appeartimes_dictionary[each_word_in_sentence]=1
            bool_if_word_exist = each_word_in_sentence in words
            if bool_if_word_exist==False:
                global words_amount
                words_amount+=1
                words.append(each_word_in_sentence)  
        row_text_appeartimes_dictionary_dictionary[rows_cnt-1]=word_appeartimes_dictionary
    for word in words:
       word_appeartimes_in_all_text_dictionary[word]=0
    for sentence in sentences:
        for word in words:
            if word in sentence:
                word_appeartimes_in_all_text_dictionary[word]+=1
    TFIDF = numpy.zeros((train_amount,words_amount))
    for i in range(train_amount):
        for j in range(words_amount):
            if words[j] in row_text_appeartimes_dictionary_dictionary[i].keys():
                '''print("该词语出现了多少次")
                print(row_text_appeartimes_dictionary_dictionary[i][words[j]])
                print("文本长度")
                print(text_length_list[i])
                print("文本总数")
                print(train_amount)
                print("总共出现了多少次")
                print(word_appeartimes_in_all_text_dictionary[words[j]])'''
                TFIDF[i][j]=(row_text_appeartimes_dictionary_dictionary[i][words[j]]/text_length_list[i])*math.log(train_amount/(1 + word_appeartimes_in_all_text_dictionary[words[j]]))
                ##print(TFIDF[i][j])
                ##print(row_text_appeartimes_dictionary_dictionary[i][words[j]]/text_length_list[i])*math.log(train_amount/(1 + word_appeartimes_in_all_text_dictionary[words[j]]))
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
    linebreak_splits_1 = train_string_1.split('\n')
    linebreak_splits_1.pop()
    del linebreak_splits_1[0]
    global test_amount
    test_amount=len(linebreak_splits_1)
    test_matrix = numpy.zeros((test_amount,words_amount),int)
    for j in range(test_amount):
        sentence_and_emotion_1 = linebreak_splits_1[j].split(',')
        emotions_of_validation.append(sentence_and_emotion_1[1])
        word_appeartimes_dictionary_1={}
        sentence_1 = sentence_and_emotion_1[0].split(' ')
        for word in words:
            word_appeartimes_in_all_text_dictionary_1[word]=0
        for word in words:
            if word in sentence_1:
                word_appeartimes_in_all_text_dictionary_1[word]+=1
        for each_word_in_sentence in sentence_1:
            if each_word_in_sentence in word_appeartimes_dictionary_1.keys():
                word_appeartimes_dictionary_1[each_word_in_sentence]+=1
            else:
                word_appeartimes_dictionary_1[each_word_in_sentence]=1
        row_text_appeartimes_dictionary_dictionary_1[j]=word_appeartimes_dictionary_1
        text_length_list_1.append(len(sentence_1))
        for i in range(words_amount):
            if words[i] in row_text_appeartimes_dictionary_dictionary_1[j].keys():
               test_matrix[j][i]=(row_text_appeartimes_dictionary_dictionary_1[j][words[i]]/text_length_list_1[j])*math.log(test_amount/(1 + word_appeartimes_in_all_text_dictionary_1[words[i]]))
            else:
               test_matrix[j][i]=0
    KNN(k,length_kind,TFIDF,test_matrix) 
def use_TF(inputstream1,inputstream2,k,length_kind):
    train_string = inputstream1.read()
    linebreak_splits = train_string.split('\n')
    linebreak_splits.pop()
    del linebreak_splits[0]
    sentence_and_emotions = []
    sentences=[]
    row_text_appeartimes_dictionary_dictionary={}
    row_text_appeartimes_dictionary_dictionary_1={}
    word_appeartimes_in_all_text_dictionary={}
    word_appeartimes_in_all_text_dictionary_1={}
    text_length_list=[]
    text_length_list_1=[]
    train_amount = 0
    rows_cnt=0
    for linebreak_split in linebreak_splits:
        train_amount+=1
        sentence_and_emotion = linebreak_split.split(',')
        emotions_of_train.append(sentence_and_emotion[1])
        sentence_and_emotions.append(sentence_and_emotion)
    for sentence_and_emotion in sentence_and_emotions:
        rows_cnt+=1
        word_appeartimes_dictionary={}
        sentence = sentence_and_emotion[0].split(' ')
        text_length_list.append(len(sentence))
        sentences.append(sentence)
        for each_word_in_sentence in sentence:
            if each_word_in_sentence in word_appeartimes_dictionary.keys():
                word_appeartimes_dictionary[each_word_in_sentence]+=1
            else:
                word_appeartimes_dictionary[each_word_in_sentence]=1
            bool_if_word_exist = each_word_in_sentence in words
            if bool_if_word_exist==False:
                global words_amount
                words_amount+=1
                words.append(each_word_in_sentence)  
        row_text_appeartimes_dictionary_dictionary[rows_cnt-1]=word_appeartimes_dictionary
    for word in words:
       word_appeartimes_in_all_text_dictionary[word]=0
    for sentence in sentences:
        for word in words:
            if word in sentence:
                word_appeartimes_in_all_text_dictionary[word]+=1
    TF = numpy.zeros((train_amount,words_amount))
    for i in range(train_amount):
        for j in range(words_amount):
            if words[j] in row_text_appeartimes_dictionary_dictionary[i].keys():
                '''print("该词语出现了多少次")
                print(row_text_appeartimes_dictionary_dictionary[i][words[j]])
                print("文本长度")
                print(text_length_list[i])
                print("文本总数")
                print(train_amount)
                print("总共出现了多少次")
                print(word_appeartimes_in_all_text_dictionary[words[j]])'''
                TF[i][j]=row_text_appeartimes_dictionary_dictionary[i][words[j]]/text_length_list[i]
                ##print(TFIDF[i][j])
                ##print(row_text_appeartimes_dictionary_dictionary[i][words[j]]/text_length_list[i])*math.log(train_amount/(1 + word_appeartimes_in_all_text_dictionary[words[j]]))
            else:
                TF[i][j]=0
    output = open("D:\\matrix.txt","w")
    output.write("这是训练集的TF矩阵！！！！！！！！！！！！！！！！")
    for i in range(len(TF)):
        output.write('\n')
        for j in range(len(TF[1])):
            output.write(str(TF[i][j]))
            output.write(' ')
    train_string_1 = inputstream2.read()
    linebreak_splits_1 = train_string_1.split('\n')
    linebreak_splits_1.pop()
    del linebreak_splits_1[0]
    global test_amount
    test_amount=len(linebreak_splits_1)
    test_matrix = numpy.zeros((test_amount,words_amount))
    for j in range(test_amount):
        sentence_and_emotion_1 = linebreak_splits_1[j].split(',')
        emotions_of_validation.append(sentence_and_emotion_1[1])
        word_appeartimes_dictionary_1={}
        sentence_1 = sentence_and_emotion_1[0].split(' ')
        for word in words:
            word_appeartimes_in_all_text_dictionary_1[word]=0
        for word in words:
            if word in sentence_1:
                word_appeartimes_in_all_text_dictionary_1[word]+=1
        for each_word_in_sentence in sentence_1:
            if each_word_in_sentence in word_appeartimes_dictionary_1.keys():
                word_appeartimes_dictionary_1[each_word_in_sentence]+=1
            else:
                word_appeartimes_dictionary_1[each_word_in_sentence]=1
        row_text_appeartimes_dictionary_dictionary_1[j]=word_appeartimes_dictionary_1
        text_length_list_1.append(len(sentence_1))
        for i in range(words_amount):
            if words[i] in row_text_appeartimes_dictionary_dictionary_1[j].keys():
               test_matrix[j][i]=row_text_appeartimes_dictionary_dictionary_1[j][words[i]]/text_length_list_1[j]
            else:
               test_matrix[j][i]=0
    output.write(("这是测试集的TF矩阵！！！！！！！！！！！！！！！！"))
    for i in range(len(test_matrix)):
        output.write('\n')
        for j in range(len(test_matrix[i])):
            output.write(str(test_matrix[i][j]))
            output.write(' ')
    KNN(k,length_kind,TF,test_matrix) 
def KNN(k,length_kind,train_matrix,test_matrix):
    for i in range(len(test_matrix)):
        distances_dictionary = {}
        distances_list=[]
        distances_list_copy=[]
        emotions_of_selected=[]
        emotions_of_selected_dictionary={}
        rows_of_selected=[]
#########################################################################依次记录预测样本[i]与所有训练样本的向量距离并记录在距离列表里面
        for j in range(len(train_matrix)):
            if length_kind==1:                     #############################曼哈顿距离
                distance = numpy.sum(numpy.abs(test_matrix[i]-train_matrix[j]))
            if length_kind==2:                   ###############################欧式距离
                distance = numpy.sqrt(numpy.sum(numpy.square(test_matrix[i]-train_matrix[j])))
            distances_dictionary.setdefault(distance,[]).append(j)
            distances_list_copy.append(distance)
            if distance not in distances_list:
                distances_list.append(distance)
        distances_list.sort()                                    ########距离排序（从小到大）
#######################################################################################按照排好序的距离顺序，记录每一个距离对应的的训练集行号
        for distance in distances_list:
            for row_number in distances_dictionary[distance]:
                rows_of_selected.append(row_number)
###################################################################################找出最前面（距离最小）的K个行号
        for r in range(k):
            emotions_of_selected.append(emotions_of_train[rows_of_selected[r]])
            '''print("距离第",r+1,"近的行号:",rows_of_selected[r])
            print("该行与目标距离:",distances_list_copy[rows_of_selected[r]])
            print("该行情感:",emotions_of_train[rows_of_selected[r]])'''
#################################################################################计算选中的k个训练文本中各情感出现的次数
        for emotion_of_selected in emotions_of_selected:
            if emotion_of_selected in emotions_of_selected_dictionary.keys():
                emotions_of_selected_dictionary[emotion_of_selected]+=1
            else:
                emotions_of_selected_dictionary[emotion_of_selected]=1
##################################################################################找出选中的k个训练文本中出现次数最多的情感
        emotions_of_test.append(max(emotions_of_selected_dictionary,key=emotions_of_selected_dictionary.get))
    output = open("D:\\validation概率预测结果.csv","w")
    for i in range(test_amount):
        output.write(emotions_of_test[i])
        output.write('\n')
    output.close()


        
'''for j in range(30):
    train_set = open("D:\\train_set1.csv","r")
    test_set = open("D:\\validation_set2.csv","r")
    test_amount=0
    emotions_of_test = []
    use_One_hot(train_set,test_set,j+1,2)
    right_amount=0
    for i in range(test_amount):
        if emotions_of_test[i]==emotions_of_validation[i]:
            right_amount+=1
    right_rate=right_amount/test_amount
    print(right_rate)'''

test_amount=0
emotions_of_test = []
train_set = open("D:\\train_set1.csv","r")
test_set = open("D:\\validation_set2.csv","r")
use_One_hot(train_set,test_set,13,2)
'''right_amount=0
print("接下来是预测的结果:")
for i in range(test_amount):
    print(emotions_of_test[i])'''
'''for i in range(test_amount):
    if emotions_of_test[i]==emotions_of_validation[i]:
        right_amount+=1
    right_rate=right_amount/test_amount
print("正确率为：",right_rate)'''