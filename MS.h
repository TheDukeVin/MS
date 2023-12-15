
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iterator>
#include <random>
#include <vector>
#include <string>
#include <unordered_set>
#include <ctime>

// #define boardH 5
// #define boardW 6
// #define numBomb 6

// MS Easy
// #define boardH 9
// #define boardW 9
// #define numBomb 10

// MS Med
// #define boardH 16
// #define boardW 16
// #define numBomb 40

// MS Hard
#define boardH 16
#define boardW 30
#define numBomb 99

#define GUESS_RANDOM 0
#define GUESS_SET 1

#ifndef MS_h
#define MS_h

using namespace std;

const string gameOut = "MS.out";

const int BOMB = -1;
const int UNMARKED = -2;

static std::random_device dev;
static std::mt19937 rng(0);

int randomN(int N);

// all arrangements of k 1's and (n-k) (-1)'s among n elements.
vector<vector<int> > combination(int n, int k);

class Pos{
public:
    int x,y;

    Pos(){}

    Pos(int x_, int y_){
        x = x_; y = y_;
    }

    string toString(){
        return to_string(x) + ' ' + to_string(y);
    }

    bool isValid(){
        return 0 <= x && x < boardH && 0 <= y && y < boardW;
    }

    vector<Pos> proximity(){
        vector<Pos> prox;
        for(int dx=-1; dx<=1; dx++){
            for(int dy=-1; dy<=1; dy++){
                if(x+dx == -1 || x+dx == boardH || y+dy == -1 || y+dy == boardW) continue;
                prox.push_back(Pos(x+dx, y+dy));
            }
        }
        return prox;
    }

    friend bool operator < (const Pos& p, const Pos& q){
        if(p.x == q.x) return p.y < q.y;
        return p.x < q.x;
    }

    friend bool operator == (const Pos& p, const Pos& q){
        return p.x == q.x && p.y == q.y;
    }
};

vector<Pos> allPos();

// We try to keep Pos objects as abstract as possible, except in the MS implementation
// where it is used on a low level.

class MS{
private:
    // -1 = bomb
    // 0-8 = proximity read
    int ground[boardH][boardW];
public:
    // -2 = unmarked
    // -1 = marked bomb
    // 0-8 = prox mark
    int marks[boardH][boardW];
    int numEmptyMark;
    int numBombMark;
    int result;

    MS(){}
    void init();
    void setResult(int res);
    void calcProx();
    void printGround();
    string printMarks();
    int getMark(Pos p);
    void markBomb(Pos p); // returns -1 if fail
    void markEmpty(Pos p); // returns -1 if fail, 1 if success
};

class Set{
public:
    vector<Pos> set;
    int count;

    Set(vector<Pos> els, int c);
    string toString() const;

    friend bool operator == (const Set& t, const Set& s){
        return t.toString() == s.toString();
    }

    // Returns this - s
    // If s is a subset, gives bomb count of diff
    // Else, gives bomb count of -1.
    Set getDiff(Set s);
};

class SetHash{
public:
    size_t operator()(const Set& s) const {
        return hash<string>{}(s.toString());
    }
};

class Solver{
private:
    unordered_set<Set, SetHash> counts;
public:
    MS state;
    bool toggle;
    int guessCount;

    Solver(){}

    void step(int mode);
    void solve(bool print=false, int mode = GUESS_SET);

    bool bombLimit(); // returns true if clears something
    bool clearLimit();
    void compSet();
    bool clearSet();
    bool guessSet();
    bool guess();

    bool localSearch();
};

#endif