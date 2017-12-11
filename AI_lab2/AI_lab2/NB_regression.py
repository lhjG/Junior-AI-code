import numpy;
emotion_probability_dictionary_list=[]
emotions_of_train=[]
emotions=[]
train_amount=0
emotion_probabilities_of_train_dictionary={}
words_amount=0
words=[]
sentences=[]
row_text_word_appeartimes_dictionary_dictionary={}
text_length_list=[]
text_length_list_1=[]
def use_TF(inputstream1,inputstream2):
    train_string = inputstream1.read()
    linebreak_splits = train_string.split('\n')
    linebreak_splits.pop()
    xxx_and_emotions=linebreak_splits[0].split(',')
    for i in range(len(xxx_and_emotions)-1):
        emotions.append(xxx_and_emotions[i+1])
    del linebreak_splits[0]
    sentences_and_emotion_probabilities = []
    rows_cnt=0
    word_appeartimes_in_all_text_dictionary={}
    for linebreak_split in linebreak_splits:
        global train_amount
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
    train_string_1 = inputstream2.read()
    row_text_word_appeartimes_dictionary_dictionary_1={}
    word_appeartimes_in_all_text_dictionary_1={}
    linebreak_splits_1 = train_string_1.split('\n')
    linebreak_splits_1.pop()
    del linebreak_splits_1[0]
    global test_amount
    test_amount=len(linebreak_splits_1)
    emotion_probabilities_of_train_dictionary_1={}
    test_matrix=numpy.zeros((test_amount,words_amount))
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
                test_matrix[j][i]=(row_text_word_appeartimes_dictionary_dictionary_1[j][words[i]]/text_length_list_1[j])
            else:
                test_matrix[j][i]=0
    NaiveBayes(test_matrix) 
def NaiveBayes(test_matrix):
    for i in range(test_amount):
        ##print("第",i+1,"个样例")
#########################################记录测试样本向量中不为0的维度

        word_list=[]
        for j in range(words_amount):
            if test_matrix[i][j] != 0:
                word_list.append(j)

#################################################初始化情感->概率字典

        emotions_probability={}
        for j in range(len(emotions)):
            emotions_probability[emotions[j]]=0

###############################################################################
        for j in range(len(emotions)):
            for k in range(train_amount):
                temp=1
                for l in range(len(word_list)):
#################################################计算在训练样本[k]中测试样本[i]不为0的维度的值之积

                    if words[word_list[l]] in row_text_word_appeartimes_dictionary_dictionary[k].keys():
                        numerator=row_text_word_appeartimes_dictionary_dictionary[k][words[word_list[l]]]+0.001
                        denominator=text_length_list[k]+words_amount*0.001
                        temp*=numerator/denominator
                    else:
                        temp*=1/(text_length_list[k]+words_amount)

#######################################################################上述结果再乘以训练文本[k]的情感[j]的概率

                temp*=float(emotion_probabilities_of_train_dictionary[k][j])

#######################################################################把所有训练样本的结果累加起来
                emotions_probability[emotions[j]] += temp
                ##print(emotions[j],":",emotions_probability[emotions[j]])
###########################################################################归一化使得各情感的概率和为1                
        sum=0
        for j in range(len(emotions)):
            sum+=float(emotions_probability[emotions[j]])
        for j in range(len(emotions)):
            if sum!=0:
                emotions_probability[emotions[j]] = float(emotions_probability[emotions[j]]/sum)
        emotion_probability_dictionary_list.append(emotions_probability)
####################################################################################################
    output = open("D:\\validation概率预测结果.csv","w")
    for i in range(test_amount):
        output.write(str(emotion_probability_dictionary_list[i][emotions[0]]))
        output.write(',')
        output.write(str(emotion_probability_dictionary_list[i][emotions[1]]))
        output.write(',')
        output.write(str(emotion_probability_dictionary_list[i][emotions[2]]))
        output.write(',')
        output.write(str(emotion_probability_dictionary_list[i][emotions[3]]))
        output.write(',')
        output.write(str(emotion_probability_dictionary_list[i][emotions[4]]))
        output.write(',')
        output.write(str(emotion_probability_dictionary_list[i][emotions[5]]))
        output.write('\n')
    output.close()
test_amount=0
train_set = open("D:\\train_set1.csv","r")
test_set = open("D:\\validation_set2.csv","r")
use_TF(train_set,test_set)