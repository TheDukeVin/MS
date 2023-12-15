
#include "MS.h"

void MS::init(){
    int index[boardH*boardW];
    for(int i=0; i<boardH*boardW; i++){
        index[i] = 0;
    }
    for(int i=0; i<numBomb; i++){
        index[i] = BOMB;
    }
    // shuffle(index, index + boardH*boardW, default_random_engine{1002});
    shuffle(index, index + boardH*boardW, default_random_engine{dev()});
    for(int i=0; i<boardH; i++){
        for(int j=0; j<boardW; j++){
            ground[i][j] = index[i*boardW + j];
        }
    }
    for(int i=0; i<boardH; i++){
        for(int j=0; j<boardW; j++){
            marks[i][j] = UNMARKED;
        }
    }
    numEmptyMark = 0;
    numBombMark = 0;
    result = 0;
}

void MS::setResult(int res){
    if(result == 0){
        result = res;
    }
}

void MS::calcProx(){
    for(int i=0; i<boardH; i++){
        for(int j=0; j<boardW; j++){
            if(ground[i][j] != 0) continue;
            int count = 0;
            for(auto neigh : Pos(i, j).proximity()){
                if(ground[neigh.x][neigh.y] == -1){
                    count ++;
                }
            }
            ground[i][j] = count;
        }
    }
}

void MS::printGround(){
    for(int i=0; i<boardH; i++){
        for(int j=0; j<boardW; j++){
            if(ground[i][j] == BOMB) cout << "X ";
            else cout << ground[i][j] << ' ';
        }
        cout << '\n';
    }
}

string MS::printMarks(){
    string s = "";
    for(int i=0; i<boardH; i++){
        for(int j=0; j<boardW; j++){
            if(marks[i][j] == UNMARKED) s += ". ";
            else if(marks[i][j] == BOMB) s += "X ";
            else s += to_string(marks[i][j]) + ' ';
        }
        s += '\n';
    }
    return s;
}

int MS::getMark(Pos p){
    return marks[p.x][p.y];
}

void MS::markBomb(Pos p){
    assert(p.isValid());
    assert(marks[p.x][p.y] == UNMARKED);
    int g = ground[p.x][p.y];
    marks[p.x][p.y] = g;
    numBombMark ++;
    assert(g == BOMB);
    // if(g != BOMB) setResult(-1);
}

void MS::markEmpty(Pos p){
    assert(p.isValid());
    assert(marks[p.x][p.y] == UNMARKED);
    if(numEmptyMark == 0){
        // Ensure all neighbors of p are not bombs
        vector<Pos> clear = p.proximity();
        clear.push_back(p);
        while(true){
            bool cleared = true;
            for(auto c : clear){
                if(ground[c.x][c.y] == BOMB){
                    cleared = false;
                    int newx = randomN(boardH);
                    int newy = randomN(boardW);
                    if(ground[newx][newy] != BOMB){
                        ground[newx][newy] = BOMB;
                        ground[c.x][c.y] = 0;
                    }
                }
            }
            if(cleared) break;
        }
        calcProx();
    }
    int g = ground[p.x][p.y];
    marks[p.x][p.y] = g;
    if(g == BOMB) setResult(-1);
    numEmptyMark ++;
    if(numEmptyMark == boardH*boardW - numBomb) setResult(1);
}