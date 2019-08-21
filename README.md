1      常用算法

1.1      分治策略

1.1.1        基本概念

在计算机科学中，分治法是一种很重要的算法。字面上的解释是“分而治之”，就是把一个复杂的问题分成两个或更多的相同或相似的子问题，再把子问题分成更小的子问题……直到最后子问题可以简单的直接求解，原问题的解即子问题的解的合并。这个技巧是很多高效算法的基础，如排序算法(快速排序，归并排序)，傅立叶变换(快速傅立叶变换)……

 

 任何一个可以用计算机求解的问题所需的计算时间都与其规模有关。问题的规模越小，越容易直接求解，解题所需的计算时间也越少。例如，对于n个元素的排序问题，当n=1时，不需任何计算。n=2时，只要作一次比较即可排好序。n=3时只要作3次比较即可，…。而当n较大时，问题就不那么容易处理了。要想直接解决一个规模较大的问题，有时是相当困难的。

1.1.2        思想策略

  分治法的设计思想是：将一个难以直接解决的大问题，分割成一些规模较小的相同问题，以便各个击破，分而治之。

   分治策略是：对于一个规模为**n的问题，若该问题可以容易地解决（比如说规模n较小）则直接解决，否则将其分解为k个规模较小的子问题，这些子问题互相独立且与原问题形式相同，递归地解这些子问题，然后将各子问题的解合并得到原问题的解。**这种算法设计策略叫做分治法。

   如果原问题可分割成k个子问题，1<k≤n，且这些子问题都可解并可利用这些子问题的解求出原问题的解，那么这种分治法就是可行的。由分治法产生的子问题往往是原问题的较小模式，这就为使用递归技术提供了方便。在这种情况下，反复应用分治手段，可以使子问题与原问题类型一致而其规模却不断缩小，最终使子问题缩小到很容易直接求出其解。这自然导致递归过程的产生。分治与递归像一对孪生兄弟，经常同时应用在算法设计之中，并由此产生许多高效算法。

1.1.3        适用场景

 分治法所能解决的问题一般具有以下几个特征：

    1) 该问题的规模缩小到一定的程度就可以容易地解决

    2) 该问题可以分解为若干个规模较小的相同问题，即该问题具有最优子结构性质。

    3) 利用该问题分解出的子问题的解可以合并为该问题的解；

    4) 该问题所分解出的各个子问题是相互独立的，即子问题之间不包含公共的子子问题。

第一条特征是绝大多数问题都可以满足的，因为问题的计算复杂性一般是随着问题规模的增加而增加；

第二条特征是应用分治法的前提它也是大多数问题可以满足的，此特征反映了递归思想的应用；、

第三条特征是关键，能否利用分治法完全取决于问题是否具有第三条特征，如果具备了第一条和第二条特征，而不具备第三条特征，则可以考虑用贪心法或动态规划法。

第四条特征涉及到分治法的效率，如果各子问题是不独立的则分治法要做许多不必要的工作，重复地解公共的子问题，此时虽然可用分治法，但一般用动态规划法较好。

1.1.4        基本步骤

分治法在每一层递归上都有三个步骤：

    step1 分解：将原问题分解为若干个规模较小，相互独立，与原问题形式相同的子问题；

    step2 解决：若子问题规模较小而容易被解决则直接解，否则递归地解各个子问题

    step3 合并：将各个子问题的解合并为原问题的解。

它的一般的算法设计模式如下：

    Divide-and-Conquer(P)

1.     if |P|≤n0

2.     then return(ADHOC(P))

3.     将P分解为较小的子问题 P1 ,P2 ,...,Pk

4.     for i←1 to k

5.     do yi ← Divide-and-Conquer(Pi) △ 递归解决Pi

6.     T ← MERGE(y1,y2,...,yk) △ 合并子问题

7.     return(T)

1.1.5        应用实例

很多算法的思维本质上都是分治思维方式，如二分搜索、大整数乘法、合并排序、快速排序等。

实例：求**x的n次幂**

复杂度为O(lgn)的解法

    int power(int x, int n)

    {

        int result;

        if(n == 1)

            return x;

        if( n % 2 == 0)

            result = power(x, n/2) * power(x, n / 2);

        else

            result = power(x, (n+1) / 2) * power(x, (n-1) / 2);

     

        return result;

    }

1.1.6        练习题目

搜索二维矩阵II：https://leetcode-cn.com/problems/search-a-2d-matrix-ii/

求众数：https://leetcode-cn.com/problems/majority-element/

合并k个排序链表：https://leetcode-cn.com/problems/merge-k-sorted-lists/

 

1.2      排序算法

1.2.1        冒泡排序

冒泡排序是一种极其简单的排序算法，通常被用来对于程序设计入门的学生介绍算法的概念。它重复地走访过要排序的元素，依次比较相邻两个元素，如果他们的顺序错误就把他们调换过来，直到没有元素再需要交换，排序完成。这个算法的名字由来是因为越小(或越大)的元素会经由交换慢慢“浮”到数列的顶端。

　　冒泡排序算法的运作如下：

1.       比较相邻的元素，如果前一个比后一个大，就把它们两个调换位置。

2.       对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对。这步做完后，最后的元素会是最大的数。

3.       针对所有的元素重复以上的步骤，除了最后一个。

4.       持续每次对越来越少的元素重复上面的步骤，直到没有任何一对数字需要比较。

冒泡排序算法复杂度是O(n^2)

    static void bubbleSort(int[] arr) {

           int size = arr.length;

           // 外层循环，保证out之后的值 是按照从小到大顺序排列的。

           for (int out = size - 1; out > 0; out --) {

               // 内层循环，将最大的值移动到最后out位置，实现了大数冒泡

               for (int in = 0; in < out; in++) {

                   if (arr[in] > arr[in + 1]) {

                       swap(arr, in, in + 1);

                   }

               }

           }

       }

1.2.2        选择排序

选择排序也是一种简单直观的排序算法。它的工作原理是初始时在序列中找到最小（大）元素，放到序列的起始位置作为已排序序列；然后，再从剩余未排序元素中继续寻找最小（大）元素，放到已排序序列的末尾。以此类推，直到所有元素均排序完毕。

　　注意选择排序与冒泡排序的区别：冒泡排序通过依次交换相邻两个顺序不合法的元素位置，从而将当前最小（大）元素放到合适的位置；而选择排序每遍历一次都记住了当前最小（大）元素的位置，最后仅需一次交换操作即可将其放到合适的位置。

选择排序的算法复杂度与冒泡排序一样为O(n^2)，代码如下：

      static void selectSort(int[] arr) {

           int size = arr.length;

           // 外层循环，保证out之前的都是有序的

           for(int out = 0; out < size; out ++) {

               int mixIndex = out;

               // 内层循环，找到out位置最小值

               for (int in = out + 1; in < size; in++) {
    

                   if (arr[mixIndex] > arr[in]) {
    

                       mixIndex = in;
    

                   }
    

               }
    

               if (mixIndex != out) {
    

                   swap(arr, mixIndex, out);
    

               }
    

           }
    

       }
    

1.2.3        插入排序

插入排序是一种简单直观的排序算法。它的工作原理非常类似于我们抓扑克牌

　　　　　　

　　对于未排序数据(右手抓到的牌)，在已排序序列(左手已经排好序的手牌)中从后向前扫描，找到相应位置并插入。

　　插入排序在实现上，通常采用in-place排序（即只需用到O(1)的额外空间的排序），因而在从后向前扫描过程中，需要反复把已排序元素逐步向后挪位，为最新元素提供插入空间。

　　具体算法描述如下：

1.   从第一个元素开始，该元素可以认为已经被排序

2.   取出下一个元素，在已经排序的元素序列中从后向前扫描

3.   如果该元素（已排序）大于新元素，将该元素移到下一位置

4.   重复步骤3，直到找到已排序的元素小于或者等于新元素的位置

5.   将新元素插入到该位置后

6.    重复步骤2~5

插入排序的算法复杂度也是O(N^2)**，这个排序的执行时间对于随机的数据排序效率比冒泡和选择排序都少，但是对于极端数据(比如说完全倒序的数据)可能没有选择好，交换的次数过多了。如果比较操作的代价比交换操作大的话，可以采用二分查找来减少比较操作的次数**。

代码实现如下：

    static void insertSort(int[] arr) {
    

           int size = arr.length;
    

           //  out之前的是有序的，所以第0位置是有序的，然后将后面的数据插入到这个局部有序数组对应位置
    

           for(int out = 1; out < size; out++) {
    

               // 记录下out位置的值，将它替换至正确的位置
    

               int temp = arr[out];
    

               int in = out;
    

               // arr[in - 1] <= temp 表示temp就应该插入in位置，
    

               // 因为temp比in+1位置的元素小，但是又比in-1位置元素大(包括等于)
    

               while(in - 1 >= 0 && arr[in - 1] > temp) {
    

                   arr[in] = arr[in - 1];
    

                   in --;
    

               }
    

               if (in != out) arr[in] = temp;
    

           }
    

       }
    

1.2.4        希尔排序

　希尔排序，也叫递减增量排序，是插入排序的一种更高效的改进版本。希尔排序是不稳定的排序算法。

　　希尔排序是基于插入排序的以下两点性质而提出改进方法的：

l  插入排序在对几乎已经排好序的数据操作时，效率高，即可以达到线性排序的效率

l  插入排序效率每次只能将数据移动一位，效率太低。

　　希尔排序通过将比较的全部元素分为几个区域来提升插入排序的性能。这样可以让一个元素可以一次性地朝最终位置前进一大步。然后算法再取越来越小的步长进行排序，算法的最后一步就是普通的插入排序，但是到了这步，需排序的数据几乎是已排好的了（此时插入排序较快）。

 　　假设有一个很小的数据在一个已按升序排好序的数组的末端。如果用复杂度为O(n^2)的排序（冒泡排序或直接插入排序），可能会进行n次的比较和交换才能将该数据移至正确位置。而希尔排序会用较大的步长移动数据，所以小数据只需进行少数比较和交换即可到正确位置。

希尔排序是不稳定的排序算法，虽然一次插入排序是稳定的，不会改变相同元素的相对顺序，但在不同的插入排序过程中，相同的元素可能在各自的插入排序中移动，最后其稳定性就会被打乱。

希尔排序效率：没有办法从理论上分析出希尔排序的效率，对于随机数据，它的执行时间大概是O(N*( log N)^2)。

    static void shellSort(int[] arr) {
    

         int size = arr.length;
    

         int h = 1;
    

         while(h <= size / 3) {
    

             //  这里加1是保证h = (h - 1) / 3最后一个值是1，
    

             h = h * 3 + 1;
    

         }
    

         while (h > 0) {
    

             for(int out = h; out < size; out++) {
    

                 int temp = arr[out];
    

                 int in = out;
    

                 // arr[in - h] <= temp 表示temp就应该插入in位置，因为temp比in-h位置的值大，又比in+h位置的值小
    

                 while(in - h >= 0 && arr[in - h] > temp) {
    

                     arr[in] = arr[in - h];
    

                     in = in - h;
    

                 }
    

                 if (in != out) arr[in] = temp;
    

             }
    

    

             h = (h - 1) / 3;
    

         }
    

     }
    

1.2.5        归并排序

归并操作（merge），也叫归并算法，指的是将两个已经排序的序列合并成一个序列的操作。该算法是采用分治法（Divide and Conquer）的一个非常典型的应用，且各层分治递归可以同时进行。

归并排序是稳定排序，它也是一种十分高效的排序，能利用完全二叉树特性的排序一般性能都不会太差。java中Arrays.sort()采用了一种名为TimSort的排序算法，就是归并排序的优化版本。每次合并操作的平均时间复杂度为O(n)，而完全二叉树的深度为|log2n|。总的平均时间复杂度为**O(nlogn)。而且，归并排序的最好，最坏，平均时间复杂度均为O(nlogn)**。



迭代方式实现：

原理如下（假设序列共有n个元素）：

l  将序列每相邻两个数字进行归并操作，形成n/2个序列

l  若此时序列数不是1，则将上述序列再次归并n/4个序列

l  重复步骤2，直到所有元素排序完毕，即序列数为1

    int min(int x, int y) {
    

        return x < y ? x : y;
    

    }
    

    void merge_sort(int arr[], int len) {
    

        int *a = arr;
    

        int *b = (int *) malloc(len * sizeof(int));
    

        int seg, start;
    

        for (seg = 1; seg < len; seg += seg) {
    

            for (start = 0; start < len; start += seg * 2) {
    

                int low = start, mid = min(start + seg, len), high = min(start + seg * 2, len);
    

                int k = low;
    

                int start1 = low, end1 = mid;
    

                int start2 = mid, end2 = high;
    

                while (start1 < end1 && start2 < end2)
    

                    b[k++] = a[start1] < a[start2] ? a[start1++] : a[start2++];
    

                while (start1 < end1)
    

                    b[k++] = a[start1++];
    

                while (start2 < end2)
    

                    b[k++] = a[start2++];
    

            }
    

            int *temp = a;
    

            a = b;
    

            b = temp;
    

        }
    

        if (a != arr) {
    

            int i;
    

            for (i = 0; i < len; i++)
    

                b[i] = a[i];
    

            b = a;
    

        }
    

        free(b);
    

    }
    

1.2.6        快速排序

快速排序是使用分支策略的一种改进型排序算法。在平均状况下，排序**n个元素要O(nlogn)次比较。在最坏状况下则需要O(n^2)次比较**，但这种状况并不常见。事实上，快速排序通常明显比其他O(nlogn)算法更快，因为它的内部循环可以在大部分的架构上很有效率地被实现出来。

算法步骤：

1.    从序列中挑出一个元素，作为"基准"(pivot).

2.    把所有比基准值小的元素放在基准前面，所有比基准值大的元素放在基准的后面（相同的数可以到任一边），这个称为分区(partition)操作。

3.    对每个分区递归地进行步骤1~2，递归的结束条件是序列的大小是0或1，这时整体已经被排好序了。

代码实现：

    typedef struct _Range {
    

        int start, end;
    

    } Range;
    

    

    Range new_Range(int s, int e) {
    

        Range r;
    

        r.start = s;
    

        r.end = e;
    

        return r;
    

    }
    

    

    void swap(int *x, int *y) {
    

        int t = *x;
    

        *x = *y;
    

        *y = t;
    

    }
    

    

    void quick_sort(int arr[], const int len) {
    

        if (len <= 0)
    

            return; 
    

        Range r[len];
    

        int p = 0;
    

        r[p++] = new_Range(0, len - 1);
    

        while (p) {
    

            Range range = r[--p];
    

            if (range.start >= range.end)
    

                continue;
    

            int mid = arr[(range.start + range.end) / 2]; 
    

            int left = range.start, right = range.end;
    

            do {
    

                while (arr[left] < mid) ++left;   
    

                while (arr[right] > mid) --right; 
    

                if (left <= right) {
    

                    swap(&arr[left], &arr[right]);
    

                    left++;
    

                    right--; 
    

                }
    

            } while (left <= right);
    

            if (range.start < right) r[p++] = new_Range(range.start, right);
    

            if (range.end > left) r[p++] = new_Range(left, range.end);
    

        }
    

    }
    

qsort/sort**函数：**

在C库中已经实现了qsort函数，一般而言我们不需要自己再手动实现快速排序相关功能的函数。

用 法: void qsort(void *base, int nelem, int width, int (*fcmp)(const void *,const void *)); 

各参数：1 待排序数组首地址 2 数组中待排序元素数量 3 各元素的占用空间大小 4 指向函数的指针

1.2.7        堆排序

堆排序是指利用堆这种数据结构所设计的一种选择排序算法。堆是一种近似完全二叉树的结构（通常堆是通过一维数组来实现的），并满足性质：以最大堆（也叫大根堆、大顶堆）为例，其中父结点的值总是大于它的孩子节点。

堆排序的过程：

1.       由输入的无序数组构造一个最大堆，作为初始的无序区

2.       把堆顶元素（最大值）和堆尾元素互换

3.       把堆（无序区）的尺寸缩小1，并调用heapify(A, 0)从新的堆顶元素开始进行堆调整

4.       重复步骤2，直到堆的尺寸为1

因为每次插入数据效率是O(log N)，而我们需要进行n次循环，将数组中每个值插入到堆中，所以它的执行时间是O(N * log N)级。

堆排序源码示例：

    void Heapify(int A[], int i, int size)  // 从A[i]向下进行堆调整
    

    {
    

        int left_child = 2 * i + 1;         // 左孩子索引
    

        int right_child = 2 * i + 2;        // 右孩子索引
    

        int max = i;                        // 选出当前结点与其左右孩子三者之中的最大值
    

        if (left_child < size && A[left_child] > A[max])
    

            max = left_child;
    

        if (right_child < size && A[right_child] > A[max])
    

            max = right_child;
    

        if (max != i)
    

        {
    

            Swap(A, i, max);                // 把当前结点和它的最大(直接)子节点进行交换
    

            Heapify(A, max, size);          // 递归调用，继续从当前结点向下进行堆调整
    

        }
    

    }
    

    

    int BuildHeap(int A[], int n)           // 建堆，时间复杂度O(n)
    

    {
    

        int heap_size = n;
    

        for (int i = heap_size / 2 - 1; i >= 0; i--) // 从每一个非叶结点开始向下进行堆调整
    

            Heapify(A, i, heap_size);
    

        return heap_size;
    

    }
    

    

    void HeapSort(int A[], int n)
    

    {
    

        int heap_size = BuildHeap(A, n);    // 建立一个最大堆
    

        while (heap_size > 1)    　　　　　　 // 堆（无序区）元素个数大于1，未完成排序
    

        {
    

            // 将堆顶元素与堆的最后一个元素互换，并从堆中去掉最后一个元素
    

            // 此处交换操作很有可能把后面元素的稳定性打乱，所以堆排序是不稳定的排序算法
    

            Swap(A, 0, --heap_size);
    

            Heapify(A, 0, heap_size);     // 从新的堆顶元素开始向下进行堆调整，时间复杂度O(logn)
    

        }
    

    }
    

1.2.8        练习题目

按奇偶排序数组： https://leetcode-cn.com/problems/sort-array-by-parity-ii/

对链表进行插入排序：https://leetcode-cn.com/problems/insertion-sort-list/

合并区间：https://leetcode-cn.com/problems/merge-intervals/

最大数：https://leetcode-cn.com/problems/largest-number/

最接近原点的K个点：https://leetcode-cn.com/problems/k-closest-points-to-origin/

1.3      贪心算法

1.3.1        基本思路

所谓贪心算法是指，在对问题求解时，总是做出在当前看来是最好的选择。也就是说，不从整体最优上加以考虑，他所做出的仅是在某种意义上的局部最优解。

贪心算法没有固定的算法框架，算法设计的关键是贪心策略的选择。必须注意的是，贪心算法不是对所有问题都能得到整体最优解，选择的贪心策略必须具备无后效性，即某个状态以后的过程不会影响以前的状态，只与当前状态有关。所以对所采用的贪心策略一定要仔细分析其是否满足无后效性。

1.3.2        算法描述

贪心算法实现步骤：

1.     建立数学模型来描述问题。

2.     把求解的问题分成若干个子问题。

3.     对每一子问题求解，得到子问题的局部最优解。

4.     把子问题的解局部最优解合成原来解问题的一个解。

实现该算法的过程：

 从问题的某一初始解出发；

    while （能朝给定总目标前进一步）

    { 

          利用可行的决策，求出可行解的一个解元素；

    }

由所有解元素组合成问题的一个可行解；

1.3.3        应用实例

贪心算法一般而言很难正确的找到问题的最优解，但是也有一些非常漂亮的贪心算法应用实践，其中就包括最小生成树算法

设G = (V,E)是无向连通带权图，即一个网络。E中的每一条边（v,w）的权为cv。如果G的子图G’是一棵包含G的所有顶点的树，则称G’为G的生成树。生成树上各边权的总和称为生成树的耗费。在G的所有生成树中，耗费最小的生成树称为G的最小生成树。

最小生成树性质：

设G = (V,E)是连通带权图，U是V的真子集。如果(u,v)∈E,且u∈U,v∈V-U,且在所有这样的边中，(u,v)的权cu最小，那么一定存在G的一棵最小生成树，它意(u,v)为其中一条边。这个性质有时也称为MST性质。

Prim**算法：**

设G = (V,E)是连通带权图，V = {1,2,…,n}。构造G的最小生成树Prim算法的基本思想是：首先置**S = {1}，然后，只要S是V的真子集，就进行如下的贪心选择：选取满足条件i** ∈**S,j** ∈**V** – S,**且ci最小的边，将顶点j添加到S中。这个过程一直进行到S = V时为止。**在这个过程中选取到的所有边恰好构成G的一棵最小生成树。

如下带权图：



生成过程：

1 -> 3 : 1

3 -> 6 : 4

6 -> 4: 2

3 -> 2 : 5

2 -> 5 : 3

Prim**算法运行时间为O(ElgV)**

1.3.4        练习题目

柠檬水找零：https://leetcode-cn.com/problems/lemonade-change/

分发饼干：https://leetcode-cn.com/problems/assign-cookies/

1.4      动态规划

动态规划（英语：Dynamic programming，简称DP）是一种通过把原问题分解为相对简单的子问题的方式求解复杂问题的方法。常常适用于有重叠子问题[1]和最优子结构性质的问题，动态规划方法所耗时间往往远少于朴素解法。

1.4.1        基本思路

动态规划背后的基本思想非常简单。大致上，若要解一个给定问题，我们需要解其不同部分（即子问题），再根据子问题的解以得出原问题的解。通常许多子问题非常相似，为此动态规划法试图仅仅解决每个子问题一次，从而减少计算量：一旦某个给定子问题的解已经算出，则将其记忆化存储，以便下次需要同一个子问题解之时直接查表。这种做法在重复子问题的数目关于输入的规模呈指数增长时特别有用。

动态规划过程是：每次决策依赖于当前状态，又随即引起状态的转移。一个决策序列就是在变化的状态中产生出来的，这种多阶段最优化决策解决问题的过程就称为动态规划。

动态规划的思路与分治法类似，也是将求解的问题分解为若干个子问题。与分治法的差别是适合于用动态规划法求解的问题，经分解后得到的子问题往往不是互相独立的（即下一个子阶段的求解是建立在上一个子阶段的解的基础上，进行进一步的求解）。

1.4.2        算法描述

能采用动态规划求解的问题的一般要具有3个性质：

    (1) 最优化原理：如果问题的最优解所包含的子问题的解也是最优的，就称该问题具有最优子结构，即满足最优化原理。

    (2) 无后效性：即某阶段状态一旦确定，就不受这个状态以后决策的影响。也就是说，某状态以后的过程不会影响以前的状态，只与当前状态有关。

   （3）有重叠子问题：即子问题之间是不独立的，一个子问题在下一阶段决策中可能被多次使用到。（该性质并不是动态规划适用的必要条件，但是如果没有这条性质，动态规划算法同其他算法相比就不具备优势）

动态规划所处理的问题是一个多阶段决策问题，一般由初始状态开始，通过对中间阶段决策的选择，达到结束状态。这些决策形成了一个决策序列，同时确定了完成整个过程的一条活动路线(通常是求最优的活动路线)。动态规划的设计都有着一定的模式，一般要经历以下几个步骤。

    （1）分析最优解的性质，并刻画其结构特征。

    （2）递归的定义最优解。

    （3）以自底向上或自顶向下的记忆化方式（备忘录法）计算出最优值

（4）根据计算最优值时得到的信息，构造问题的最优解

动态规划的主要难点在于理论上的设计，也就是上面4个步骤的确定，一旦设计完成，实现部分就会非常简单。

     使用动态规划求解问题，最重要的就是确定动态规划三要素：

（1**）问题的阶段** 

（2**）每个阶段的状态**

（3**）从前一个阶段转化到后一个阶段之间的递推关系。**

确定了动态规划的这三要素，整个求解过程就可以用一个最优决策表来描述，最优决策表是一个二维表，其中行表示决策的阶段，列表示问题状态，表格需要填写的数据一般对应此问题的在某个阶段某个状态下的最优值（如最短路径，最长公共子序列，最大价值等），填表的过程就是根据递推关系，从1行1列开始，以行或者列优先的顺序，依次填写表格，最后根据整个表格的数据通过简单的取舍或者运算求得问题的最优解。

f(n,m)=max{f(n-1,m), f(n-1,m-w[n])+P(n,m)}

1.4.3        背包问题

问题描述：

有N件物品和一个体积为V的背包。（每种物品均只有一件）第i件物品的体积是volume[i]，价值是value[i]。求解将哪些物品装入背包可使这些物品的体积总和不超过背包体积，且价值总和最大？

解题思路：

pi代表前i件物品组合在容量为j的背包的最优解。将前i件物品放入容量为v的背包中这个子问题，若只考虑第i件物品的策略（放或不放），那么就可以转化为一个只牵扯前i-1件物品的问题。如果不放第i件物品，那么问题就转化为“前i-1件物品放入容量为v的背包中，价值为pi-1；如果放第i件物品，那么问题就转化为“前i-1件物品放入剩下的容量为v-volume[i]的背包中”，此时能获得的最大价值就是pi-1]再加上通过放入第i件物品获得的价值value[i]。

状态转移方程：

pi=MAX{pi-1]+value[i],pi-1};

伪码描述：

for i=1..N

　　    for j=V..0

　　           p[j]=MAX{p[j-volume[i]]+value[i],p[j]};

 

1.4.4        最长公共子序列

问题描述：

一个数列 S，如果分别是两个或多个已知数列的子序列，且是所有匹配此条件序列中最长的，则S称为已知序列的最长公共子序列(LCS)。

给定两个序列X、Y，求两个序列的最长公共子序列。

 

解题思路：

最长公共子序列问题存在最优子结构：这个问题可以分解成更小，更简单的“子问题”，这个子问题可以分成更多的子问题，因此整个问题就变得简单了。

最长公共子序列问题的子问题的解是可以重复使用的，也就是说，更高级别的子问题通常会重用低级子问题的解。拥有这个两个属性的问题可以使用动态规划算法来解决，这样子问题的解就可以被储存起来，而不用重复计算。这个过程需要在一个表中储存同一级别的子问题的解，因此这个解可以被更高级的子问题使用。

设有二维数组fi**表示X的i位和Y的j位之前的最长公共子序列的长度**，则有：

f1=same(1,1)

fi=max{fi-1+same(i,j),fi-1,fi}

其中，same(a,b)当X的第a位与Y的第b位完全相同时为“1”，否则为“0”。

此时，fi中最大的数便是X和Y的最长公共子序列的长度，依据该数组回溯，便可找出最长公共子序列。

该算法的空间、时间复杂度均为O(n^2)**，经过优化后，空间复杂度可为O(n)。**

 

伪码描述：

    function LCSLength(X[1..m], Y[1..n])
    

        C = array(0..m, 0..n)
    

        for i := 0..m
    

           C[i,0] = 0
    

    for j := 0..n
    

           C[0,j] = 0
    

        for i := 1..m
    

            for j := 1..n
    

                if X[i] = Y[j]
    

                    C[i,j] := C[i-1,j-1] + 1
    

                else
    

                    C[i,j] := max(C[i,j-1], C[i-1,j])
    

        return C[m,n]
    

1.4.5        练习题目

最大子序和：https://leetcode-cn.com/problems/maximum-subarray/

编辑距离：https://leetcode-cn.com/problems/edit-distance/

大礼包：https://leetcode-cn.com/problems/shopping-offers/

最长上升子序列：https://leetcode-cn.com/problems/longest-increasing-subsequence/

1.5      回溯算法

1.5.1        基本思路

回溯算法实际上一个类似枚举的搜索尝试过程，主要是在搜索尝试过程中寻找问题的解，当发现已不满足求解条件时，就“回溯”返回，尝试别的路径。

回溯法是一种选优搜索法，按选优条件向前搜索，以达到目标。但当探索到某一步时，发现原先选择并不优或达不到目标，就退回一步重新选择，这种走不通就退回再走的技术为回溯法，而满足回溯条件的某个状态的点称为“回溯点”。

许多复杂的，规模较大的问题都可以使用回溯法，有“通用解题方法”的美称。

1.5.2        算法描述

在包含问题的所有解的解空间树中，按照深度优先搜索的策略，从根结点出发深度探索解空间树。当探索到某一结点时，要先判断该结点是否包含问题的解，如果包含，就从该结点出发继续探索下去，如果该结点不包含问题的解，则逐层向其祖先结点回溯。（其实回溯法就是对隐式图的深度优先搜索算法）。

若用回溯法求问题的所有解时，要回溯到根，且根结点的所有可行的子树都要已被搜索遍才结束。而若使用回溯法求任一个解时，只要搜索到问题的一个解就可以结束。

回溯法一般解题步骤：

    （1）针对所给问题，确定问题的解空间：

         首先应明确定义问题的解空间，解空间应至少包含问题的一个（最优）解。

    （2）确定结点的扩展搜索规则

（3）以深度优先方式搜索解空间，并在搜索过程中用剪枝函数避免无效搜索。

伪码描述：

// 针对N叉树的迭代回溯方法 

void iterativeBacktrack () 

{ 

int t = 1; 

while (t > 0) { //**有路可走 

if (f(n,t) <= g(n,t)) { // 遍历结点t的所有子结点 

for (int i = f(n,t); i <= g(n,t); i ++) { 

x[t] = h(i);  // 剪枝 

if (constraint(t) && bound(t)) { 

// 找到问题的解，输出结果 

if (solution(t)) { 

output(x); 

} 

else // 未找到，向更深层次遍历 

t ++; 

} 

} 

}else { 

t--; 

} 

} 

}

1.5.3        八皇后问题

问题描述：

八皇后问题是一个以国际象棋为背景的问题：如何能够在 8×8 的国际象棋棋盘上放置八个皇后，使得任何一个皇后都无法直接吃掉其他的皇后？为了达到此目的，任两个皇后都不能处于同一条横行、纵行或斜线上。



题解：

转化规则：其实八皇后问题可以推广为更一般的n皇后摆放问题：这时棋盘的大小变为n×n，而皇后个数也变成n。当且仅当 n = 1 或 n ≥ 4 时问题有解。令一个一维数组a[n]保存所得解，其中a[i] 表示把第i个皇后放在第i行的列数（注意i的值都是从0开始计算的），下面就八皇后问题的约束条件。

（1）因为所有的皇后都不能放在同一列，因此任意两个a[0].....a[7]的值不能存在相同的两个值。

（2）所有的皇后都不能在对角线上，那么该如何检测两个皇后是否在同一个对角线上？我们将棋盘的方格成一个二维数组，如下：



假设有两个皇后被放置在（i，j）和（k，l）的位置上，明显，当且仅当|i-k|=|j-l| 时，两个皇后才在同一条对角线上。

代码描述：

    int queens(int Queens){
    

        int i, k, flag, not_finish=1, count=0;
    

        //正在处理的元素下标，表示前i-1个元素已符合要求，正在处理第i个元素
    

        int a[Queens+1];    //八皇后问题的皇后所在的行列位置，从1幵始算起，所以加1
    

        i=1;
    

        a[1]=1;  //为数组的第一个元素赋初值
    

    

        while(not_finish){  //not_finish=l:处理尚未结束
    

            while(not_finish && i<=Queens){  //处理尚未结束且还没处理到第Queens个元素            for(flag=1,k=1; flag && k<i; k++) //判断是否有多个皇后在同一行
    

                    if(a[k]==a[i]){
    

                        flag=0;
    

    }
    

    

                for (k=1; flag&&k<i; k++){  //判断是否有多个皇后在同一对角线
    

                    if( (a[i]==a[k]-(k-i)) || (a[i]==a[k]+(k-i)) ){
    

                        flag=0;
    

                    }
    

                }
    

    

                if(!flag){  //若存在矛盾不满足要求，需要重新设置第i个元素
    

                    if(a[i]==a[i-1]){  //若a[i]的值已经经过一圈追上a[i-1]的值
    

                        i--;  //退回一步，重新试探处理前一个元素
    

    

                        if(i>1 && a[i]==Queens){
    

                            a[i]=1;  //当a[i]为Queens时将a[i]的值置1
    

                        }
    

                        else {
    

                            if(i==1 && a[i]==Queens){
    

                                not_finish=0;  //当第一位的值达到Queens时结束
    

                            }
    

                            else{
    

                                a[i]++;  //将a[il的值取下一个值
    

                           }
    

                        }
    

                    }else if(a[i] == Queens){
    

                        a[i]=1;
    

                    }
    

                    else{
    

                        a[i]++;  //将a[i]的值取下一个值
    

                   }
    

                }else if(++i<=Queens){
    

                    if(a[i-1] == Queens ) {
    

                        a[i]=1;  //若前一个元素的值为Queens则a[i]=l
    

         }
    

                    else {
    

                        a[i] = a[i-1]+1;  //否则元素的值为前一个元素的下一个值
    

                   }
    

            }
    

    

            if(not_finish){
    

                ++count;    
    

                if(a[Queens-1]<Queens){
    

                    a[Queens-1]++;  //修改倒数第二位的值
    

                }
    

                else {
    

                    a[Queens-1]=1;
    

                }
    

                i = Queens -1;    //开始寻找下一个满足条件的解
    

            }
    

        }
    

        return count;
    

    }
    

1.5.4        练习题目

N皇后：https://leetcode-cn.com/problems/n-queens/

括号生成：https://leetcode-cn.com/problems/generate-parentheses/

单词搜索：https://leetcode-cn.com/problems/word-search/

解数独：https://leetcode-cn.com/problems/sudoku-solver/

2      引用

程序性能描述：

https://liam.page/2016/06/20/big-O-cheat-sheet/

https://zh.wikipedia.org/wiki/%E6%97%B6%E9%97%B4%E5%A4%8D%E6%9D%82%E5%BA%A6

线性表 ：

https://zh.wikipedia.org/wiki/%E7%BA%BF%E6%80%A7%E8%A1%A8

https://zh.wikipedia.org/wiki/%E4%BA%8C%E5%88%86%E6%90%9C%E7%B4%A2%E7%AE%97%E6%B3%95

https://www.jianshu.com/p/73f0d8f807aa           

https://www.jianshu.com/p/c47d40e9c85c

https://lotabout.me/2018/skip-list/

https://www.cnblogs.com/vamei/archive/2013/03/14/2960201.html

https://blog.csdn.net/TW_345/article/details/50054505

https://www.cnblogs.com/yangecnu/p/Introduction-Stack-and-Queue.html

https://www.cnblogs.com/vamei/archive/2013/03/14/2960201.html

https://blog.csdn.net/ljianhui/article/details/10287879

哈希表：

https://www.jianshu.com/p/4d3cb99d7580

https://blog.csdn.net/duan19920101/article/details/51579136

https://zhuanlan.zhihu.com/p/31441081

树：

https://zh.wikipedia.org/wiki/%E6%A0%91_(%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84)

http://data.biancheng.net/view/192.html

https://www.jianshu.com/p/45661b029292

https://zh.wikipedia.org/wiki/%E4%BA%8C%E5%85%83%E6%90%9C%E5%B0%8B%E6%A8%B9

https://zh.wikipedia.org/wiki/AVL%E6%A0%91

https://zh.wikipedia.org/wiki/%E5%B9%B3%E8%A1%A1%E6%A0%91

https://zh.wikipedia.org/wiki/B%2B%E6%A0%91

https://blog.csdn.net/juanqinyang/article/details/51418629

图：

https://www.jianshu.com/p/6cace353141d

https://zhuanlan.zhihu.com/p/25498681

https://blog.csdn.net/v_JULY_v/article/details/6096981

https://blog.csdn.net/Charles_ke/article/details/82497543

https://blog.csdn.net/lisonglisonglisong/article/details/45543451

 

分治：

https://www.cnblogs.com/steven_oyj/archive/2010/05/22/1741370.html

https://blog.csdn.net/baidu_35692628/article/details/78049609

https://blog.csdn.net/zwhlxl/article/details/44086105

 

排序：

https://zhuanlan.zhihu.com/p/31729381

https://www.cnblogs.com/alsf/p/6606287.html

https://www.cnblogs.com/eniac12/p/5329396.html

https://www.cnblogs.com/chengxiao/p/6194356.html

https://zh.wikipedia.org/wiki/%E5%BD%92%E5%B9%B6%E6%8E%92%E5%BA%8F

https://zh.wikipedia.org/wiki/%E5%BF%AB%E9%80%9F%E6%8E%92%E5%BA%8F

https://www.cnblogs.com/syxchina/archive/2010/07/29/2197382.html

 

贪心：

https://zh.wikipedia.org/wiki/%E8%B4%AA%E5%BF%83%E7%AE%97%E6%B3%95

https://www.cnblogs.com/steven_oyj/archive/2010/05/22/1741375.html

https://www.cnblogs.com/chinazhangjie/archive/2010/12/02/1894314.html

 

动态规划：

https://zh.wikipedia.org/wiki/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92

https://www.zhihu.com/question/23995189

https://www.cnblogs.com/steven_oyj/archive/2010/05/22/1741374.html

https://zh.wikipedia.org/wiki/%E8%83%8C%E5%8C%85%E9%97%AE%E9%A2%98

https://blog.csdn.net/liangbopirates/article/details/9750463

https://blog.csdn.net/v_july_v/article/details/6110269

https://zh.wikipedia.org/wiki/%E6%9C%80%E9%95%BF%E5%85%AC%E5%85%B1%E5%AD%90%E5%BA%8F%E5%88%97

 

回溯算法：

https://www.cnblogs.com/steven_oyj/archive/2010/05/22/1741376.html

https://zh.wikipedia.org/wiki/%E5%9B%9E%E6%BA%AF%E6%B3%95

https://xiaozhuanlan.com/topic/0289571364

https://zh.wikipedia.org/wiki/%E5%85%AB%E7%9A%87%E5%90%8E%E9%97%AE%E9%A2%98

https://blog.csdn.net/EbowTang/article/details/51570317

 
