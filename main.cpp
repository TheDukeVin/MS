
/*
g++ -std=c++11 main.cpp MS.cpp solver.cpp common.cpp set.cpp
*/

#include "MS.h"

void manualGame(){
    MS board;
    while(board.result == 0){
        board.printMarks();
        int x,y;
        cin >> x >> y;
        board.markEmpty(Pos(x, y));
    }
    cout << "Result: " << board.result << '\n';
}

int main(){
    // vector<vector<int> > a = combination(7, 3);
    // for(auto arr : a){
    //     for(auto i : arr){
    //         cout << i << ' ';
    //     }
    //     cout << '\n';
    // }
    // cout << a.size() << '\n';

    ofstream fout(gameOut); fout.close();
    for(int i=0; i<10; i++){
        ofstream fout(gameOut, ios::app);
        fout << "########################### GAME " << i << " ###############################\n";
        fout.close();
        Solver s;
        s.state.init();
        s.solve(true, GUESS_SET);
    }

    // unsigned start_time = time(0);
    // Solver s;
    // int numGames = 1000;
    // int numWins = 0;
    // int numGuesses = 0;
    // for(int i=0; i<numGames; i++){
    //     s.state.init();
    //     s.solve(false, GUESS_RANDOM);
    //     numGuesses += s.guessCount;
    //     if(s.state.result == 1) numWins ++;
    // }
    // cout << "Win rate: " << numWins << " out of " << numGames << '\n';
    // cout << "Num guesses: " << numGuesses << '\n';
    // {
    //     numWins = 0;
    //     for(int i=0; i<numGames; i++){
    //         s.state.init();
    //         s.solve(false, GUESS_SET);
    //         if(s.state.result == 1) numWins ++;
    //     }
    //     cout << "Win rate: " << numWins << " out of " << numGames << '\n';
    // }
    // cout << "Time: " << (time(0) - start_time) << '\n';
}