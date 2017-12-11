import numpy;
import math;
emotions_of_train=[]
words = []
words_amount=0
emotions_of_validation=[]
emotion_each_word_total_dictionary={}
emotion_thesaurus_dictionary={}
emotion_non_repetitive_thesaurus_dictionary={}
emotions_non_repetitively_list=[]
test_emotion_predict_result_list=[]
total_words_amount=0
train_amount = 0
def use_One_hot(inputstream1,inputstream2):
    train_string = inputstream1.read()
    linebreak_splits = train_string.split('\n')  
    linebreak_splits.pop()
    del linebreak_splits[0]
    sentence_and_emotions = []
    sentences=[]
    for linebreak_split in linebreak_splits:
        global train_amount
        train_amount+=1
        sentence_and_emotion = linebreak_split.split(',')
        if sentence_and_emotion[1] not in emotions_non_repetitively_list:
            emotions_non_repetitively_list.append(sentence_and_emotion[1])
        emotions_of_train.append(sentence_and_emotion[1])
        sentence_and_emotions.append(sentence_and_emotion)
    for sentence_and_emotion in sentence_and_emotions:
        sentence = sentence_and_emotion[0].split(' ')
        global total_words_amount
        total_words_amount+=len(sentence)
        sentences.append(sentence)
        for each_word_in_sentence in sentence:
            bool_if_word_exist = each_word_in_sentence in words
            if bool_if_word_exist==False:
                global words_amount
                words_amount+=1
                words.append(each_word_in_sentence)
    for i in range(len(emotions_of_train)):
        if emotions_of_train[i] not in emotion_thesaurus_dictionary.keys():
            thesaurus=[]
            thesaurus_non_repetitive=[]
            for j in range(train_amount):
                if emotions_of_train[j]==emotions_of_train[i]:
                    for k in range(len(sentences[j])):
                        thesaurus.append(sentences[j][k])
                        if sentences[j][k] not in thesaurus_non_repetitive:
                            thesaurus_non_repetitive.append(sentences[j][k])
            emotion_thesaurus_dictionary[emotions_of_train[i]]=thesaurus
            emotion_non_repetitive_thesaurus_dictionary[emotions_of_train[i]]=thesaurus_non_repetitive
    for i in range(len(emotions_non_repetitively_list)):
        word_total_dictionary={}
        for j in range(words_amount):    
            if words[j] in emotion_thesaurus_dictionary[emotions_non_repetitively_list[i]]:
                cnt = 0
                for k in range(len(emotion_thesaurus_dictionary[emotions_non_repetitively_list[i]])):
                    if emotion_thesaurus_dictionary[emotions_non_repetitively_list[i]][k]==words[j]:
                        cnt+=1
                word_total_dictionary[words[j]]=cnt
            else:
                word_total_dictionary[words[j]]=0
        emotion_each_word_total_dictionary[emotions_non_repetitively_list[i]]=word_total_dictionary
    One_hot = numpy.zeros((train_amount,words_amount),int)
    for i in range(train_amount):
        for j in range(words_amount):
            bool_if_word_in_this_train = words[j] in sentences[i]
            if bool_if_word_in_this_train:
                One_hot[i][j]=1
    print(One_hot)
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
    NaiveBayes(test_matrix)
def NaiveBayes(test_matrix):
    for i in range(test_amount):
        emotion_probability_dictionary={}                          ############键为"情感",值为该预测样本该情感的"概率"
        exist_list=[]                                              ############在该预测样本向量中不为0的维度
        for j in range(words_amount):
            if test_matrix[i][j]==1:
                exist_list.append(words[j])            
        for j in range(len(emotions_non_repetitively_list)):        ####遍历情感列表    
            p_this_emotion_condition_probability=1  

            
###############################################下面在计算p(y|x)的条件概率的分子

############################################p_this_emotion = 情感[j]在整个训练文本的概率（多项式模型）

            p_this_emotion = len(emotion_thesaurus_dictionary[emotions_non_repetitively_list[j]])/total_words_amount

#################################################计算分子中的条件概率  

            for k in range(len(exist_list)):   

######################################情感[j]中第0、1、2、3、4、....个预测样本不为0的维度的词语出现总数 + 1/ 情感[j]中出现的词语总数 + 训练集不重复的词语总数
                print(exist_list[k],":")
                numerator=emotion_each_word_total_dictionary[emotions_non_repetitively_list[j]][exist_list[k]]+1
                denominator = len(emotion_thesaurus_dictionary[emotions_non_repetitively_list[j]])+words_amount
                print(numerator," ",denominator)
                p_this_emotion_condition_probability*=float(numerator/denominator)
                
                
####################################################测试样本[i]属于情感[j]的概率         
            cnt=0
            for k in range(train_amount):
                if emotions_of_train[k]==emotions_non_repetitively_list[j]:
                    cnt+=1
            p_this_emotion=cnt/train_amount
            emotion_probability_dictionary[emotions_non_repetitively_list[j]]=p_this_emotion_condition_probability * p_this_emotion
            print("第",i+1,"个预测样本属于情感",emotions_non_repetitively_list[j],"的概率为:",emotion_probability_dictionary[emotions_non_repetitively_list[j]])

####################################################测试样本[i]的情感是各个情感的概率最大者            
        test_emotion_predict_result_list.append(max(emotion_probability_dictionary,key=emotion_probability_dictionary.get))
        print("第",i+1,"个预测样本预测情感为:",max(emotion_probability_dictionary,key=emotion_probability_dictionary.get))
        output = open("D:\\validation概率预测结果.csv","w")
    for i in range(test_amount):
        output.write(test_emotion_predict_result_list[i])
        output.write('\n')
    output.close()

test_amount=0
train_set = open("D:\\train_set1.csv","r")
test_set = open("D:\\validation_set2.csv","r")
use_One_hot(train_set,test_set)
'''right_amount=0
for i in range(test_amount):
    if test_emotion_predict_result_list[i]==emotions_of_validation[i]:
        right_amount+=1
right_rate = right_amount/test_amount
print(right_rate)'''