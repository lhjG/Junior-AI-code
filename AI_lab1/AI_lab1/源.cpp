#include <string>  
#include <stdio.h>  
#include<iostream>
#include<fstream>
#include<vector>
#include<map>
#include<cmath>

using namespace std;



struct ThreeElementMap
{
	int row;
	int col;
	int value;
};
int create_Onehot(vector<string> vstring,vector<map<string,int>> vmap,double** matrix,int rows,int cols)
{
	int cnt = 0;
	for (int i = 0; i <= rows - 1; i++)
	{
		for (int j = 0; j <= cols - 1; j++)
		{
			matrix[i][j] = vmap[i][vstring[j]];//排序为j的单词在排序为i的map容器中的映射值     （0/1）
			if (vmap[i][vstring[j]] == 1)	cnt++;
		}
	}
	return cnt;
}
void print_matrix(double **matrix, int rows, int cols)
{
	for (int i = 0; i <= rows - 1; i++)
	{
		for (int j = 0; j <= cols - 1; j++)
			cout << matrix[i][j]<<' ';
		cout << endl;
	}
}
void create_TF(vector<string> vstring, vector<map<string, int>> vmap, vector<int> vint,double** matrix,int rows,int cols)
{
	for (int i = 0; i <= rows - 1; i++)
	{
		for (int j = 0; j <= cols - 1; j++)
			matrix[i][j] = (double) vmap[i][vstring[j]] / vint[i];//排序为j的单词在排序为i的map容器的映射值（该单词在该行出现的次数）/该行单词数目
	}
}
void create_TFIDF(double** TFmatrix, double** TFIDFmatrix, int article_amount, vector<string> vstring,map<string,int> vmap,int cols)
{
	for (int i = 0; i <= article_amount - 1; i++)
	{
		for (int j = 0; j <= cols - 1; j++)
		{
			double log_value = (double) log((double) article_amount / (vmap[vstring[j]])) / log(2);//文本总数/排序为j的单词出现过的文本数
			TFIDFmatrix[i][j] = (double) TFmatrix[i][j] * log_value;
		}
	}
}
void create_ThreeElementTable(double **matrix,int rows,int cols,int **table,int sum)
{
	int cnt=0;
	table[0][0] = rows;
	table[1][0] = cols;
	table[2][0] = sum;
	for (int i = 0; i <= rows - 1; i++)
	{
		for (int j = 0; j <= cols - 1; j++)
		{
			if (matrix[i][j] == 1)
			{
				cnt++;
				table[cnt + 2][0] = i;
				table[cnt + 2][1] = j;
				table[cnt + 2][2] = 1;
			}
		}
	}
}
void print_ThreeElementTable(int **table,int rows)
{
	cout << table[0][0] << endl << table[1][0] << endl << table[2][0] << endl;
	for (int i = 3; i <= rows - 1; i++)
	{
		for (int j = 0; j <= 2; j++)
			cout << table[i][j] << ' ';
		cout << endl;
	}
}
void cmp(ThreeElementMap *th, int size)
{
	int row_temp, col_temp, value_temp;
	for (int j = 0; j <= size - 1; j++)
	{
		for (int i = 0; i <= size - 2; i++)
		{
			if (th[i].row > th[i + 1].row)
			{
				row_temp = th[i].row;
				th[i].row = th[i + 1].row;
				th[i + 1].row = row_temp;
				col_temp = th[i].col;
				th[i].col = th[i + 1].col;
				th[i + 1].col = col_temp;
				value_temp = th[i].value;
				th[i].value = th[i + 1].value;
				th[i + 1].value = value_temp;
			}
			else if (th[i].row == th[i + 1].row)
			{
				if (th[i].col > th[i + 1].col)
				{
					col_temp = th[i].col;
					th[i].col = th[i + 1].col;
					th[i + 1].col = col_temp;
					value_temp = th[i].value;
					th[i].value = th[i + 1].value;
					th[i + 1].value = value_temp;
				}
				else if (th[i].col == th[i + 1].col)
				{
					th[i].value += th[i + 1].value;
					th[i + 1].row = 10000;
				}
			}
		}
	}
}
int** Matrix_addition(int **table1, int **table2)
{
	int size1 = table1[2][0];
	int size2 = table2[2][0];
	int size = size1 + size2;
	ThreeElementMap *th = new ThreeElementMap[size];
	for (int i = 0; i <= size1 - 1; i++)
	{
		th[i].row = table1[i + 3][0];
		th[i].col = table1[i + 3][1];
		th[i].value = table1[i + 3][2];
	}
	for (int i = 0; i <= size2 - 1; i++)
	{
		th[i+size1].row = table2[i + 3][0];
		th[i+size1].col = table2[i + 3][1];
		th[i+size1].value = table2[i + 3][2];
	}
	/***
	for (int i = 0; i <= size - 1; i++)
	{
		cout << th[i].row << ' ' << th[i].col << ' ' << th[i].value << endl;
	}***/
	cmp(th, size);
	/***
	for (int i = 0; i <= size - 1; i++)
	{
		cout << th[i].row << ' ' << th[i].col << ' ' << th[i].value << endl;
	}***/
	int cnt = 0;
	for (int i = 0; i <= size - 1; i++)
		if (th[i].row != 10000)	cnt++;
	size = cnt;
	//cout << size << endl;
	/********
	for (int i = 0; i <= size - 1; i++)
	{
		cout << th[i].row<<' '<< th[i].col<<' '<< th[i].value<<endl;
	}********/
	int **table = new int*[size + 3];
	for (int i = 0; i <= size + 2; i++)
		table[i] = new int[3];
	table[0][0] = table1[0][0];
	table[1][0] = table1[1][0];
	table[2][0] = size;
	//cout << table[0][0] << ' ' << table[1][0] << ' ' << table[2][0] << endl;
	for (int i = 0; i <= size - 1; i++)
	{
		table[i+3][0] = th[i].row;
		table[i+3][1] = th[i].col;
		table[i+3][2] = th[i].value;
	}
	return table;
}
void print_ThreeElementTable(int **t)
{
	cout << t[0][0] << endl<<t[1][0]<<endl<<t[2][0]<<endl;
	for (int i = 0; i <= t[2][0] - 1; i++)
		cout << t[i + 3][0] << ' ' << t[i + 3][1] << ' ' << t[i + 3][2] << endl;
}


int main()
{
	vector<string> strings, words;
	vector<map<string, int>> all_if_appeared_in_a_row,all_appear_times_in_a_row;
	ifstream datatxt;
	datatxt.open("D:\\test1.txt");
	string datarow;
	map<string,int> if_record_word, appear_times_in_all_article;
	vector<int> words_each_row;
	int rows_cnt=0,words_cnt=0;
	while (getline(datatxt, datarow))
	{
		rows_cnt++;  //记录行数
		int datarow_size = datarow.size();
		char *char_of_datarow = new char[datarow_size + 1];
		for (int i = 0; i <= datarow_size - 1; i++)
			char_of_datarow[i] = datarow[i];
		char_of_datarow[datarow_size] = '\0';
		const char *delimit = "\t"; //第一次分割字符串，分割符为Tab
		char *p, *buf;
		buf = NULL;
		p = strtok_s(char_of_datarow, delimit, &buf);
		while (p)
		{
			strings.push_back(p);  //将分割后的三个字符串片段一次压入strings向量
			p = strtok_s(NULL, delimit, &buf);
		}
		int section_amounts = strings.size();
		int usefulsection_size = strings[section_amounts - 1].size();
		char *s1 = new char[usefulsection_size + 1];
		for (int i = 0; i <= usefulsection_size - 1; i++)
			s1[i] = strings[section_amounts - 1][i];//取出第一次分割后的最后一段字符串
		s1[usefulsection_size] = '\0';
		delimit = " ";//第二次分割的分隔符为空格
		p = strtok_s(s1, delimit, &buf);//对刚取出的字符串进行第二次分割，分隔符为空格，分成单词
		map<string, int> if_appeared_in_a_row, appear_times_in_a_row, appear_in_an_article_flag;
		int row_words = 0;
		while (p)//此循环为依次读取分割后的单词
		{
			if (appear_in_an_article_flag[p] == 0)//如果此单词在这篇文章里面从未出现过
			{
				appear_times_in_all_article[p]++;//此单词在所有文本里的出现次数++  用于TF-IDF矩阵
				appear_in_an_article_flag[p] = 1;//标记此单词已经在这篇文章里出现过（被计算过）
			}
			if_appeared_in_a_row[p] = 1;//此单词在这篇文章里出现过
			appear_times_in_a_row[p]++; //此单词在这篇文章里的出现次数++
			row_words++;//这篇文章里的单词数目++
			if (if_record_word[p] == 0)//如果这个单词被记录过   用于不重复记录所有文章里的单词
			{
				words.push_back(p);//压入代表所有单词的集合的向量words
				if_record_word[p] = 1;//标记此单词已经被记录过
				words_cnt++;//不重复的单词数目++
			}
			p = strtok_s(NULL, delimit, &buf);
		}
		all_if_appeared_in_a_row.push_back(if_appeared_in_a_row);//将代表每一行的映射集合起来
		all_appear_times_in_a_row.push_back(appear_times_in_a_row);
		words_each_row.push_back(row_words);
	}
	double **Onehot = new double*[rows_cnt];//动态创建二元数组来创建空的One-hot  TF  TFIDF矩阵
	double **TF = new double*[rows_cnt];
	double **TFIDF = new double*[rows_cnt];
	for (int i = 0; i <= rows_cnt - 1; i++)
	{
		Onehot[i] = new double[words_cnt];
		TF[i] = new double[words_cnt];
		TFIDF[i] = new double[words_cnt];
	}
	char order;
	int value_amount;
	cout << "是否需要创建One-hot矩阵，需要请输入‘y’，否则输入‘n’：";
	cin >> order;
	if (order == 'y')
	{
		value_amount = create_Onehot(words, all_if_appeared_in_a_row, Onehot, rows_cnt, words_cnt);
		print_matrix(Onehot, rows_cnt, words_cnt);
	}
	cout << "是否需要创建TF矩阵，需要请输入‘y’，否则输入‘n’：";
	cin >> order;
	if (order == 'y')
	{
		create_TF(words, all_appear_times_in_a_row, words_each_row, TF, rows_cnt, words_cnt);
		print_matrix(TF, rows_cnt, words_cnt);
	}
	cout << "是否需要创建TF-IDF矩阵，需要请输入‘y’，否则输入‘n’：";
	cin >> order;
	if (order == 'y')
	{
		create_TF(words, all_appear_times_in_a_row, words_each_row, TF, rows_cnt, words_cnt);
		create_TFIDF(TF, TFIDF, rows_cnt, words, appear_times_in_all_article, words_cnt);
		print_matrix(TFIDF, rows_cnt, words_cnt);
	}
	cout<< "是否需要创建三元顺序表，需要请输入‘y’，否则输入‘n’：";
	cin >> order;
	if (order == 'y')
	{
		value_amount = create_Onehot(words, all_if_appeared_in_a_row, Onehot, rows_cnt, words_cnt);
		int **ThreeElementTable = new int*[value_amount + 3];
		for (int i = 0; i <= value_amount + 2; i++)
			ThreeElementTable[i] = new int[3];
		create_ThreeElementTable(Onehot, rows_cnt,words_cnt,ThreeElementTable,value_amount);
		print_ThreeElementTable(ThreeElementTable, value_amount + 3);
	}
	cout << "是否需要计算三元表矩阵加法，需要请输入‘y’，否则输入‘n’：";
	cin >> order;
	if (order == 'y')
	{
		int row1,row2,col1, col2, value_number1,value_number2;
		cout << "请依次输入两个需要相加的三元顺序表：" ;
		cin >> row1 >> col1 >> value_number1;//读取作为加数的两个三元顺序表
		int **table1 = new int*[value_number1 + 3];
		int **table2 = new int*[value_number1 + 3];
		for (int i = 0; i <= value_number1 + 2; i++)
			table1[i] = new int[3];
		table1[0][0] = row1;
		table1[1][0] = col1;
		table1[2][0] = value_number1;
		for (int i = 0; i <= value_number1 - 1; i++)
			cin >> table1[i+3][0] >> table1[i+3][1] >> table1[i+3][2];
		cin >> row2 >> col2 >> value_number2;
		for (int i = 0; i <= value_number2 + 2; i++)
			table2[i] = new int[3];
		table2[0][0] = row2;
		table2[1][0] = col2;
		table2[2][0] = value_number2;
		for (int i = 0; i <= value_number2 - 1; i++)
			cin >> table2[i+3][0] >> table2[i+3][1] >> table2[i+3][2];
		print_ThreeElementTable(Matrix_addition(table1, table2));
	}
	system("pause");
	return 0;
}