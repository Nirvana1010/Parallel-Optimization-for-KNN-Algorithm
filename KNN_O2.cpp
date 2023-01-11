/* ***********************************************
Author        :Nirvana
Created Time  :2020/06/05
File Name     :main.cpp
Version       :Optimize Version - 2
Describe      :
               Distance calculate part: SIMD, unloop, openmp
               Distance Sort part: openmp version of oddEvenSort,
                                   using pthread binding MergeSort with oddEvenSort/QuickSort
                                   binding QuickSort with SelectionSort
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

#define maxn 10010
using namespace std;

int k = 5;
int thread_num = 4;
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
    int L;
    int R;
    int id;
};
pthread_barrier_t barrier_sort;
sem_t sem_parent;
sem_t sem_children;
int mid1, mid2, mid3;
int m;
string label[maxn];
int true_num = 0;


void quickSort(int left, int right, pair<double,string> arr[]);
void MergeSort(pair<double,string> arr[],int L,int R);
void oddEvenSort(pair<double,string> arr[], int l, int r);
void* para_sort(void* pa);

int input(string s, node data[])
{
    ifstream fin;
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
            fin >> data[i].x[j];
        for(j = 0; j < 10; j++)
        {
            fin >> temp;
            data[i].s.append(temp);
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

    __m256d sum, temp;
    __m256d t1, t2;
    sum = _mm256_setzero_pd();
    temp = _mm256_setzero_pd();

    for(i = 0; i < 256; i += 16)
    {
        t1 = _mm256_loadu_pd(a.x+i);
        t2 = _mm256_loadu_pd(b.x+i);
        temp = _mm256_sub_pd(t1, t2);
        temp = _mm256_mul_pd(temp, temp);
        sum = _mm256_add_pd(sum, temp);

        t1 = _mm256_loadu_pd(a.x+i+4);
        t2 = _mm256_loadu_pd(b.x+i+4);
        temp = _mm256_sub_pd(t1, t2);
        temp = _mm256_mul_pd(temp, temp);
        sum = _mm256_add_pd(sum, temp);

        t1 = _mm256_loadu_pd(a.x+i+8);
        t2 = _mm256_loadu_pd(b.x+i+8);
        temp = _mm256_sub_pd(t1, t2);
        temp = _mm256_mul_pd(temp, temp);
        sum = _mm256_add_pd(sum, temp);

        t1 = _mm256_loadu_pd(a.x+i+12);
        t2 = _mm256_loadu_pd(b.x+i+12);
        temp = _mm256_sub_pd(t1, t2);
        temp = _mm256_mul_pd(temp, temp);
        sum = _mm256_add_pd(sum, temp);
    }

    sum = _mm256_hadd_pd(sum, sum);
    sum = _mm256_hadd_pd(sum, sum);
    tmp = sum[0];

    for(i = 0; i < 256; i++)
        tmp += mul(a.x[i], b.x[i]);
    return sqrt(tmp);
}

string knn(int n,node x)
{
    int i;
    #pragma omp parallel for num_threads(thread_num) private(i) shared(x)
    for(i = 1; i <= n; i++)
    {
        int j = 0;
        double tmp = 0;
        #pragma omp simd
        for(j = 0; j < 256; j++)
            tmp += mul(x.x[j], nod[i].x[j]);
        p[i]={sqrt(tmp), nod[i].s};
    }

    //quickSort(0, n-1, p);

    MergeSort(p, 0, n-1);

    //oddEvenSort(p, 0, n-1);

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
    int i = 0;

    int L = 0, R = n-1;
    int mid1 = L + ((R-L)/2);
	int mid2 = L + ((mid1-L)/2);
	int mid3 = mid1+1 + ((R-mid1)/2);
    param paRam[thread_num];
	pthread_t *thread_handles;
    thread_handles = new pthread_t[thread_num];
	pthread_barrier_init(&barrier_sort, NULL, thread_num);
	sem_init(&sem_parent, 0, 0);
    sem_init(&sem_children, 0, 0);

	///Create threads for parallel sort
	paRam[0].L = L;
	paRam[0].R = mid2;
	paRam[1].L = mid2+1;
	paRam[1].R = mid1;
	paRam[2].L = mid1+1;
	paRam[2].R = mid3;
	paRam[3].L = mid3+1;
	paRam[3].R = R;

    for(i = 0; i < thread_num; i++)
    {
        paRam[i].id = i;
        int err = pthread_create(&thread_handles[i], NULL, para_sort, (void*)&paRam[i]);
        if(err != 0)
            cout << "Create thread fail" << endl;
    }

    for(i = 0; i < m; i++)
    {
        label[i] = knn(n, tes[i]);

        if(label[i] == tes[i].s)
            true_num++;
    }

    //Join
    for(int j = 0; j < thread_num; j++)
    {
        cout << j << endl;
        int err = pthread_join(thread_handles[j], NULL);
        if(err != 0)
            cout << "Join thread fail" << endl;
    }

    //cout << "true_num: " << true_num << endl;
    double accuracy = (double)true_num/m;
    cout << "the accuracy of kNN is: " << accuracy << endl;
    sem_destroy(&sem_children);
    sem_destroy(&sem_parent);
}

int main()
{
    int n;
    struct timeval start, finish;
    gettimeofday(&start, NULL);
    n = input("train.txt", nod);
    m = input("test.txt", tes);

    test(n, m);

    gettimeofday(&finish, NULL);
    unsigned long diff;
    diff = 1000000 * (finish.tv_sec-start.tv_sec) + finish.tv_usec-start.tv_usec;
    cout << "test total time: " << diff << " us" << endl;
    return 0;
}

void quickSort(int left, int right,pair<double,string> arr[])
{

	if(right - left < 33)
    {
        int i, j, pos;
        pair<double,string> tmp;

        for (i = left; i < right; i++)
        {
            for (pos = i, j = i+1; j <= right; j++)
                if (arr[pos].first > arr[j].first)
                    pos = j;
            if (pos != i)
            {
                tmp = arr[i];
                arr[i] = arr[pos];
                arr[pos] = tmp;
            }
        }
        return;
    }

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

void MergeSort(pair<double,string> arr[],int L,int R)
{
    int i;

    for(i = 0; i < thread_num; i++)
        sem_post(&sem_children);

    //Waken children
    for(i = 0; i < thread_num; i++)
        sem_wait(&sem_parent);

    merge(arr, L, mid2, mid1);
    merge(arr, mid1, mid3, R);
    merge(arr, L, mid1, R);
}

void* para_sort(void* pa)
{
    param *my_paRam = (param*)pa;
    int L = my_paRam->L;
    int R = my_paRam->R;
    //printf("thread %d create\n", my_paRam->id);
    for(int i = 0; i < m; i++)
    {
        sem_wait(&sem_children);
        oddEvenSort(p, L, R);
        //quickSort(L, R, p);
        sem_post(&sem_parent);
        pthread_barrier_wait(&barrier_sort);
    }
    printf("thread %d finish\n", my_paRam->id);
}

//parallel version
void oddEvenSort(pair<double,string> arr[], int l, int r)
{

    int s, i;
    #pragma omp parallel num_threads(thread_num) shared(p, r, l) private(s, i)
    for(s = l; s <= r; s++)
    {
        if(s%2 == 1)
        {
            #pragma omp for nowait schedule(dynamic, 64)
            for (i = 1+l; i <= r-1; i = i+2)
            {
                //printf("�߳�ID%d\n",omp_get_thread_num());
                if (p[i].first > p[i+1].first)
                {
                    pair<double,string> temp = p[i];
                    p[i] = p[i+1];
                    p[i+1] = temp;
                }
            }

        }

        if(s%2 == 0)
        {
            #pragma omp for nowait schedule(dynamic, 64)
            for (i = 0+l; i <= r; i = i+2)
            {
                if (p[i].first > p[i+1].first)
                {
                    pair<double,string> temp = p[i];
                    p[i] = p[i+1];
                    p[i+1] = temp;
                }
            }
        }
    }

}
