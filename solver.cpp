
#include "MS.h"

void Solver::step(int mode){
    // if(state.numEmptyMark == 0){
    //     state.markEmpty(Pos(0,0));
    // }
    // if(state.result != 0) return;
    // if(bombLimit()) return;
    // if(clearLimit()) return;
    // if(localSearch()) return;
    // compSet();
    // if(clearSet()) return;
    // guessCount ++;
    // // guess();
    // guessSet();

    if(state.numEmptyMark == 0){
        state.markEmpty(Pos(0,0));
    }
    toggle = !toggle;
    if(toggle){
        while(true){
            if(state.result != 0) return;
            if(bombLimit()) continue;
            if(clearLimit()) continue;
            if(localSearch()) continue;
            compSet();
            if(clearSet()) continue;
            break;
        }
    }
    else{
        if(mode == GUESS_RANDOM) guess(); 
        if(mode == GUESS_SET) guessSet();
    }
}

void Solver::solve(bool print, int mode){
    toggle = false;
    guessCount = 0;
    while(state.result == 0){
        if(print){
            ofstream fout(gameOut, ios::app);
            fout << state.printMarks();
            fout << '\n';
            fout.close();
        }
        step(mode);
    }
    if(print){
        ofstream fout(gameOut, ios::app);
        fout << state.result << '\n';
        fout.close();
    }
}

bool Solver::bombLimit(){
    bool marked = false;
    for(auto p : allPos()){
        if(state.getMark(p) < 0) continue;
        int count = 0;
        for(auto neigh : p.proximity()){
            if(state.getMark(neigh) < 0){
                count ++;
            }
        }
        if(count == state.getMark(p)){
            for(auto neigh : p.proximity()){
                if(state.getMark(neigh) == UNMARKED){
                    state.markBomb(neigh);
                    assert(state.result != -1);
                    marked = true;
                }
            }
        }
    }
    return marked;
}

bool Solver::clearLimit(){
    bool marked = false;
    for(auto p : allPos()){
        if(state.getMark(p) < 0) continue;
        int count = 0;
        for(auto neigh : p.proximity()){
            if(state.getMark(neigh) == BOMB){
                count ++;
            }
        }
        if(count == state.getMark(p)){
            for(auto neigh : p.proximity()){
                if(state.getMark(neigh) == UNMARKED){
                    state.markEmpty(neigh);
                    assert(state.result != -1);
                    marked = true;
                }
            }
        }
    }
    return marked;
}

void Solver::compSet(){
    counts = unordered_set<Set, SetHash>();
    // if(numBomb - state.numBombMark <= 5){
    //     vector<Pos> els;
    //     for(auto p : allPos()){
    //         if(state.getMark(p) == UNMARKED){
    //             els.push_back(p);
    //         }
    //     }
    //     counts.insert(Set(els, numBomb - state.numBombMark));
    // }
    for(auto p : allPos()){
        if(state.getMark(p) < 0) continue;
        int count = state.getMark(p);
        vector<Pos> els;
        for(auto neigh : p.proximity()){
            if(state.getMark(neigh) == UNMARKED){
                els.push_back(neigh);
            }
            if(state.getMark(neigh) == BOMB){
                count--;
            }
        }
        if(count > 0){
            Set s(els, count);
            counts.insert(s);
        }
    }
    bool done = false;
    while(!done){
        done = true;
        for(auto s : counts){
            for(auto t : counts){
                Set d = s.getDiff(t);
                // cout << d.toString() << '\n';
                if(d.count != -1 && counts.find(d) == counts.end()){
                    counts.insert(d);
                    done = false;
                }
            }
        }
    }
}

bool Solver::clearSet(){
    for(auto s : counts){
        if(s.count > 0) continue;
        for(auto p : s.set){
            if(state.getMark(p) == UNMARKED){
                state.markEmpty(p);
                return true;
            }
        }
    }
    for(auto s : counts){
        if(s.count < s.set.size()) continue;
        for(auto p : s.set){
            if(state.getMark(p) == UNMARKED){
                state.markBomb(p);
                return true;
            }
        }
    }
    for(auto s : counts){
        for(auto t : counts){
            Set d = s.getDiff(t);
            if(d.set.size() != s.count - t.count) continue;
            for(auto p : d.set){
                if(state.getMark(p) == UNMARKED){
                    state.markBomb(p);
                    return true;
                }
            }
        }
    }
    return false;
}

bool Solver::guess(){
    while(true){
        int x = randomN(boardH);
        int y = randomN(boardW);
        if(state.marks[x][y] == UNMARKED){
            state.markEmpty(Pos(x, y));
            return true;
        }
    }
}

bool Solver::guessSet(){
    double sumProb[boardH][boardW];
    double numProb[boardH][boardW];
    for(int i=0; i<boardH; i++){
        for(int j=0; j<boardW; j++){
            sumProb[i][j] = numProb[i][j] = 0;
        }
    }
    for(auto s : counts){
        for(auto p : s.set){
            sumProb[p.x][p.y] += (double) s.count / s.set.size();
            numProb[p.x][p.y] += 1;
        }
    }
    double minProb = 2;
    Pos bestAct(-1, -1);
    for(int i=0; i<boardH; i++){
        for(int j=0; j<boardW; j++){
            if(state.marks[i][j] == UNMARKED && numProb[i][j] > 0){
                double candProb = sumProb[i][j] / numProb[i][j];
                if(candProb < minProb){
                    minProb = candProb;
                    bestAct = Pos(i, j);
                }
            }
        }
    }
    if(bestAct.x == -1){
        return guess();
    }
    // cout << bestAct.x << ' ' << bestAct.y << '\n';
    state.markEmpty(bestAct);
    return true;
}

bool Solver::localSearch(){
    int contingentBomb[boardH][boardW]; // 1 = bomb. -1 = not bomb.
    for(int i=0; i<boardH; i++){
        for(int j=0; j<boardW; j++){
            contingentBomb[i][j] = 0;
        }
    }
    // cout << state.printMarks() << '\n';
    for(auto p : allPos()){
        // cout << p.toString() << '\n';
        if(state.getMark(p) < 0) continue;
        int count = state.getMark(p);
        vector<Pos> els;
        for(auto neigh : p.proximity()){
            if(state.getMark(neigh) == UNMARKED){
                els.push_back(neigh);
            }
            if(state.getMark(neigh) == BOMB){
                count--;
            }
        }
        vector<vector<int> > arrangements = combination(els.size(), count);
        bool bombWitness[els.size()];
        bool clearWitness[els.size()];
        for(int i=0; i<els.size(); i++){
            bombWitness[i] = clearWitness[i] = false;
        }
        for(auto arr : arrangements){
            // fill contingent bombs
            // cout << "Contingent bombs: ";
            for(int i=0; i<els.size(); i++){
                contingentBomb[els[i].x][els[i].y] = arr[i];
                // cout << els[i].toString() << ": " << arr[i] << ',';
            }
            // cout << '\n';
            bool valid = true;
            for(int x=-2; x<=2; x++){
                for(int y=-2; y<=2; y++){
                    Pos cond(p.x + x, p.y + y);
                    if(!cond.isValid() || state.getMark(cond) < 0) continue;
                    int bombMarkCount = 0;
                    int freeCount = 0;
                    for(auto neigh : cond.proximity()){
                        if(state.getMark(neigh) == UNMARKED && contingentBomb[neigh.x][neigh.y] == 0){
                            freeCount ++;
                        }
                        if(state.getMark(neigh) == BOMB || contingentBomb[neigh.x][neigh.y] == 1){
                            bombMarkCount ++;
                            // cout << "new bomb: " << neigh.toString() << '\n';
                            // cout << contingentBomb[neigh.x][neigh.y] << '\n';
                        }
                    }
                    // cout << "Condition: " << cond.toString() << '\n';
                    // cout << "Bomb mark: " << (bombMarkCount > state.getMark(cond)) << '\n';
                    // cout << "Total mark: " << (bombMarkCount + freeCount < state.getMark(cond)) << '\n';
                    if(bombMarkCount > state.getMark(cond) || bombMarkCount + freeCount < state.getMark(cond)){
                        valid = false;
                    }
                }
            }
            // delete contingent bombs
            for(int i=0; i<els.size(); i++){
                contingentBomb[els[i].x][els[i].y] = 0;
            }
            if(valid){
                for(int i=0; i<els.size(); i++){
                    if(arr[i] == 1){
                        bombWitness[i] = true;
                    }
                    else{
                        clearWitness[i] = true;
                    }
                }
            }
        }
        bool marked = false;
        for(int i=0; i<els.size(); i++){
            assert(bombWitness[i] || clearWitness[i]);
            if(!clearWitness[i]){
                state.markBomb(els[i]);
                assert(state.result != -1);
                marked = true;
            }
            if(!bombWitness[i]){
                state.markEmpty(els[i]);
                assert(state.result != -1);
                marked = true;
            }
        }
        if(marked) return true;
    }
    return false;
}