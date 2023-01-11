/* ***********************************************
Author        :Nirvana
Created Time  :2020/06/24
File Name     :main.cpp
Version       :Optimize Version - 3
Describe      :
               Distance calculate part:
               Distance Sort part: parallel version of testing the data
                                   parallel version of read & test data
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
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define maxn 10010
using namespace std;

int k = 5;
int thread_num = 2;
struct node
{
    double x[256];
    string s;
}nod[maxn];
pair<double,string>p[maxn];
map<string,int>mp;
node tes[maxn];
struct param
{
    int id;
};
pthread_barrier_t barrier;
sem_t sem_parent;
sem_t sem_children;
int mid1, mid2, mid3;
int m = 480, n = 1116;
int alRead = 0;

void quickSort(int left, int right, pair<double,string> arr[]);
void* para_test(void* pa);

ifstream fin;
streampos input(string s, node data[], int beg, int readCount, streampos sp)
{
    fin.open(s);
    if(!fin){
        cout<<"can not open the file "<<s<<endl;
        exit(1);
    }

    if(sp != NULL)
        fin.seekg(sp);

    int i = 0, j = 0;
    while(i < readCount)
    {
        string temp;
        for(j = 0; j < 256; j++)
            fin >> data[i+beg].x[j];
        for(j = 0; j < 10; j++)
        {
            fin >> temp;
            data[i+beg].s.append(temp);
        }
        i++;
    }
    streampos p = fin.tellg();
    fin.close();
    return p;
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

    quickSort(0, n-1, p);

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

void test(int n, int m)
{
    string label[m];
    int true_num = 0;
    int i = 0;

    for(i = 0; i < m; i++)
    {
        label[i] = knn(n, tes[i]);
        if(label[i] == tes[i].s)
        {
            true_num++;
        }
    }

    //cout << "true_num: " << true_num << endl;
    double accuracy = (double)true_num/m;
    cout << "the accuracy of kNN is: " << accuracy << endl;
}

int main()
{
    input("train.txt", nod, 0, n, NULL);
    struct timeval start, finish;
    gettimeofday(&start, NULL);

    param paRam[thread_num];
	pthread_t *thread_handles;
    thread_handles = new pthread_t[thread_num];
    for(int i = 0; i < thread_num; i++)
    {
        paRam[i].id = i;
        int err = pthread_create(&thread_handles[i], NULL, para_test, (void*)&paRam[i]);
        if(err != 0)
            cout << "Create thread fail" << endl;
    }

    //Join
    for(int i = 0; i < thread_num; i++)
    {
        int err = pthread_join(thread_handles[i], NULL);
        if(err != 0)
            cout << "Join thread fail" << endl;
    }

    gettimeofday(&finish, NULL);
    unsigned long diff;
    diff = 1000000 * (finish.tv_sec-start.tv_sec) + finish.tv_usec-start.tv_usec;
    cout << "total time: " << diff << " us" << endl;
    cout << "sec: " << finish.tv_sec-start.tv_sec << endl;
    cout << "usec: " << finish.tv_usec-start.tv_usec << endl;
    return 0;
}

void* para_test(void* pa)
{
    param *my_paRam = (param*)pa;
    int id = my_paRam->id;
    if(id == 0)
    {
        //READ thread
        streampos sp = input("test.txt", tes, 0, 40, NULL);
        alRead += 40;
        for(int i = 1; i < 12; i++)
        {
            sp = input("test.txt", tes, i*40, 40, sp);
            alRead += 40;
        }
    }
    else if(id == 1)
    {
        //TEST thread
        string label;
        int true_num = 0;
        int i = 0;

        for(i = 0; i < m; i++)
        {
            if(i < alRead)
            {
                label = knn(n, tes[i]);
                if(label == tes[i].s)
                {
                    true_num++;
                }
            }
            else
            {
                sleep(0);
                i--;
            }

        }

        //cout << "true_num: " << true_num << endl;
        double accuracy = (double)true_num/m;
        cout << "the accuracy of kNN is: " << accuracy << endl;
    }
    pthread_exit(nullptr);
}

void quickSort(int left, int right,pair<double,string> arr[])
{
	if(left >= right)
		return;
	int i, j;
	pair<double,string> temp, base;
	i = left, j = right;
	base = arr[left];
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
	quickSort(left, i - 1, arr);
	quickSort(i + 1, right, arr);
}


