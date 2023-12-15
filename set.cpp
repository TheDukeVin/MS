
#include "MS.h"

Set::Set(vector<Pos> els, int c){
    count = c;
    for(auto p : els){
        set.push_back(p);
    }
    sort(set.begin(), set.end());
    // cout << toString() << '\n';
    // cout << c << ' ' << els.size() << ' ' << ' ' << (c + els.size()) << ' ' << (c <= (int) els.size()) << '\n';
    assert((-1 <= c) && (c <= (int) els.size()));
}

string Set::toString() const{
    string s = "";
    for(auto p : set){
        s += to_string(p.x) + ',' + to_string(p.y) + ' ';
    }
    s += "Count: " + to_string(count);
    return s;
}

Set Set::getDiff(Set s){
    bool isSubset = true;
    for(auto p : s.set){
        bool found = false;
        for(auto q : set){
            if(p == q){
                found = true;
                break;
            }
        }
        if(!found){
            isSubset = false;
            break;
        }
    }
    vector<Pos> els;
    for(auto p : set){
        bool found = false;
        for(auto q : s.set){
            if(p == q){
                found = true;
                break;
            }
        }
        if(!found){
            els.push_back(p);
        }
    }
    // vector<Pos>::iterator it = set.begin();
    // for(auto p : s.set){
    //     bool found = false;
    //     while(it != set.end()){
    //         if(p == *it){
    //             found = true;
    //             break;
    //         }
    //         it ++;
    //     }
    //     if(!found){
    //         isSubset = false;
    //         break;
    //     }
    // }
    // vector<Pos> els;
    // it = s.set.begin();
    // for(auto p : set){
    //     bool found = false;
    //     while(it != s.set.end()){
    //         if(p == *it){
    //             found = true;
    //             break;
    //         }
    //         it ++;
    //     }
    //     if(!found){
    //         els.push_back(p);
    //     }
    // }
    int newCount;
    if(isSubset){
        newCount = count - s.count;
    }
    else{
        newCount = -1;
    }
    return Set(els, newCount);
}