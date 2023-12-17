
/*
g++ -std=c++11 -O2 -pthread main.cpp MS.cpp solver.cpp common.cpp set.cpp Qlearn.cpp environment.cpp -I "./LSTM" LSTM/node.cpp LSTM/model.cpp LSTM/PVUnit.cpp LSTM/layer.cpp LSTM/layers/lstmlayer.cpp LSTM/layers/policy.cpp LSTM/layers/conv.cpp LSTM/layers/pool.cpp LSTM/params.cpp && sbatch MS.slurm

rsync -r MS kevindu@login.rc.fas.harvard.edu:./MultiagentSnake
rsync -r kevindu@login.rc.fas.harvard.edu:./MultiagentSnake/MS .
*/

#include "DQN_MS.h"

#define numThreads 8

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

class Hyperparameters{
public:
    int numIter = 10000;
    int numRollout = 10;
    int evalPeriod = 100;
    int batchSize = 100;
    double learnRate = 0.1;
    double momentum = 0.7;
    double regRate = 0;

    Hyperparameters(int numIter_, int numRollout_, int evalPeriod_, int batchSize_, double learnRate_, double momentum_, double regRate_){
        numIter = numIter_;
        numRollout = numRollout_;
        evalPeriod = evalPeriod_;
        batchSize = batchSize_;
        learnRate = learnRate_;
        momentum = momentum_;
        regRate = regRate_;
    }
};

Qlearn* trainers[numThreads];
thread* threads[numThreads];

Hyperparameters hps[numThreads] = {
    Hyperparameters(10000, 10, 100, 100, 0.1, 0.7, 0),
    Hyperparameters(10000, 10, 100, 100, 0.05, 0.7, 0),
    Hyperparameters(10000, 10, 100, 100, 0.02, 0.7, 0),
    Hyperparameters(10000, 10, 100, 100, 0.01, 0.7, 0),
    Hyperparameters(10000, 10, 100, 100, 0.1, 0.9, 0),
    Hyperparameters(10000, 10, 100, 100, 0.05, 0.9, 0),
    Hyperparameters(10000, 10, 100, 100, 0.02, 0.9, 0),
    Hyperparameters(10000, 10, 100, 100, 0.01, 0.9, 0)
};

void runThread(int i){
    trainers[i]->fileOut = "session" + to_string(i) + ".out";
    cout << i << '\n';
    trainers[i]->train(hps[i].numIter, hps[i].numRollout, hps[i].evalPeriod, hps[i].batchSize, hps[i].learnRate, hps[i].momentum, hps[i].regRate);
}

void runAll(){
    for(int i=0; i<numThreads; i++){
        trainers[i] = new Qlearn();
        threads[i] = new thread(runThread, i);
    }
    for(int i=0; i<numThreads; i++){
        threads[i]->join();
    }
}

int main(){
    // ofstream fout(gameOut); fout.close();
    // for(int i=0; i<10; i++){
    //     ofstream fout(gameOut, ios::app);
    //     fout << "########################### GAME " << i << " ###############################\n";
    //     fout.close();
    //     Solver s;
    //     s.state.init();
    //     s.solve(true, GUESS_SET);
    // }

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
    
    runAll();
}
