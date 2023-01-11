/* ***********************************************
Author        :Nirvana
Created Time  :2020/06/05
File Name     :main.cpp
Version       :Naive Version
************************************************ */
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <stdio.h>
#include <algorithm>
#include <map>
#include <string>
#include <math.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <sys/time.h>
#include <immintrin.h>
#include <omp.h>
#define maxn 10010
using namespace std;

int k = 5;
struct node
{
    double x[256];
    string s;
}nod[maxn];
pair<double,string>p[maxn];
map<string,int>mp;

void quickSort(int left, int right, pair<double,string> arr[]);
void MergeSort(pair<double,string> arr[],int L,int R);
void oddEvenSort(pair<double,string> arr[], int n);

ifstream fin;
int input(string s)
{
    fin.open(s);
    if(!fin){
        cout<<"can not open the file "<<s<<endl;
        exit(1);
    }

    int i = 0, j = 0;
    while(!fin.eof())
    {
        string temp;
        for(j = 0; j < 256; j++)
            fin >> nod[i].x[j];
        for(j = 0; j < 10; j++)
        {
            fin >> temp;
            nod[i].s.append(temp);
        }
        i++;
    }
    return i;
}

double mul(double x,double y)
{
    return (x-y)*(x-y);
}

double dis(node a,node b)
{
    int i = 0;
    double tmp = 0;
    for(i = 0; i < 256; i++)
        tmp += mul(a.x[i], b.x[i]);

    return sqrt(tmp);
}

string knn(int n,node x)
{
    int i;
    for(i = 1; i <= n; i++)
    {
        p[i]={dis(x,nod[i]), nod[i].s};
    }

    //quickSort(0, n-1, p);

    //MergeSort(p, 0, n-1);

    oddEvenSort(p, n-1);

    mp.clear();

    for(i = 1; i <= k; i++)
        mp[p[i].second]++;

    int Max = 0;
    string ans;
    for(auto x:mp)
    {
        if(x.second > Max)
        {
            Max = x.second;
            ans = x.first;
        }
    }
    return ans;
}

void test(string s, int n)
{
    ifstream test;
    node tes;
    string label;
    int true_num = 0;
    test.open(s);
    if(!test){
        cout << "can not open the file " << s << endl;
        exit(1);
    }

    int i = 0, j = 0;
    while(!test.eof())
    {
        string temp;
        tes = {0};
        for(j = 0; j < 256; j++)
            test >> tes.x[j];
        for(j = 0; j < 10; j++)
        {
            test >> temp;
            tes.s.append(temp);
        }
        label = knn(n, tes);
        //cout << "label: " << label << " true: " << tes.s << endl;
        if(label == tes.s)
            true_num++;
        i++;
    }
    //cout << true_num << " " << i << endl;
    double accuracy = (double)true_num/i;
    cout << "the accuracy of kNN is: " << accuracy << endl;
}

int main()
{
    int n,m;
    struct timeval start, finish;
    gettimeofday(&start, NULL);
    n=input("train.txt");
    test("test.txt", n);

    gettimeofday(&finish, NULL);
    unsigned long diff;
    diff = 1000000 * (finish.tv_sec-start.tv_sec) + finish.tv_usec-start.tv_usec;
    cout << "total time: " << diff << " us" << endl;
    cout << "sec: " << finish.tv_sec-start.tv_sec << endl;
    cout << "usec: " << finish.tv_usec-start.tv_usec << endl;
    return 0;
}

void quickSort(int left, int right,pair<double,string> arr[])
{
	if(left >= right)
		return;
	int i, j;
	pair<double,string> temp, base;
	i = left, j = right;
	base = arr[left];  //set most left number as base
	while (i < j)
	{
		while (arr[j].first >= base.first && i < j)
			j--;
		while (arr[i].first <= base.first && i < j)
			i++;
		if(i < j)
		{
			temp = arr[i];
			arr[i] = arr[j];
			arr[j] = temp;
		}
	}
	arr[left] = arr[i];
	arr[i] = base;
	//recursion: left
	quickSort(left, i - 1, arr);
	//recursion: right
	quickSort(i + 1, right, arr);
}

void merge(pair<double,string> arr[],int L,int mid,int R)
{
	//pair<double,string> *help = new pair<double,string>(R-L+1);
	pair<double,string> help[R-L+1];
	int p1=L,p2=mid+1,i=0;
	while(p1<=mid && p2<=R)
	{
		help[i++] = arr[p1].first > arr[p2].first ? arr[p2++] : arr[p1++];
	}
	while(p1<=mid)
		help[i++] = arr[p1++];
	while(p2<=R)
		help[i++] = arr[p2++];

	for (int i=0;i<R-L+1;i++)
	{
		arr[L+i] = help[i];
	}
}

void sortprocess(pair<double,string> arr[],int L,int R)
{
	if (L < R)
	{
	    //  (L+R)/2
		int mid = L + ((R-L)>>2);
		sortprocess(arr, L, mid);
		sortprocess(arr, mid+1, R);
		merge(arr, L, mid, R);
	}
}

void MergeSort(pair<double,string> arr[],int L,int R)
{
	if ((sizeof(arr)-sizeof(arr[0])) < 2)
		return;
	sortprocess(arr,L,R);
}

void oddEvenSort(pair<double,string> arr[], int n)
{
    //Whether data change occurs in this iteration
    int exchFlag = 1;
    //ODD exchange / EVEN exchange
    int start = 0;
    int i;
    while(exchFlag == 1 || start == 1)
    {
        exchFlag = 0;
        for(i = start; i < n; i += 2)
        {
            if(arr[i].first > arr[i+1].first)
            {
                pair<double,string> temp = arr[i];
                arr[i] = arr[i+1];
                arr[i+1] = temp;
                exchFlag = 1;
            }
        }
        if(start == 0)
            start = 1;
        else
            start = 0;
    }
}

