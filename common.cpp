
#include "MS.h"

int randomN(int N){
    uniform_int_distribution<std::mt19937::result_type> dist(0,N-1);
    return dist(rng);
}

double randUniform(){
    uniform_real_distribution<double> distribution(0.0,1.0);
    return distribution(rng);
}

double sigmoid(double x){
    return 1/(1 + exp(-x));
}

int min(int x, int y){
    if(x < y) return x;
    return y;
}

vector<Pos> allPos(){
    vector<Pos> pos;
    for(int i=0; i<boardH; i++){
        for(int j=0; j<boardW; j++){
            pos.push_back(Pos(i, j));
        }
    }
    return pos;
}

vector<vector<int> > combination(int n, int k){
    assert(n != 0);
    assert(0 <= k && k <= n);
    int nums[n];
    for(int i=0; i<n; i++){
        nums[i] = -1;
    }
    for(int i=0; i<k; i++){
        nums[i] = 1;
    }
    vector<vector<int> > arrangements;
    while(true){
        vector<int> arr;
        for(int i=0; i<n; i++){
            arr.push_back(nums[i]);
        }
        arrangements.push_back(arr);
        bool foundGap = false;
        int i;
        int count = 0;
        for(i=n-1; i>=0; i--){
            if(nums[i] == -1){
                foundGap = true;
            }
            if(nums[i] == 1){
                count ++;
            }
            if(nums[i] == 1 && foundGap){
                break;
            }
        }
        int index = i;
        if(index < 0) return arrangements;
        for(int i=index; i<n; i++){
            nums[i] = -1;
        }
        for(int i=index+1; i<index+count+1; i++){
            nums[i] = 1;
        }
    }
}