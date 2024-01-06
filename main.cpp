
/*
g++ -std=c++11 -O2 -pthread main.cpp MS.cpp solver.cpp common.cpp set.cpp Qlearn.cpp environment.cpp -I "./LSTM" LSTM/node.cpp LSTM/model.cpp LSTM/PVUnit.cpp LSTM/layer.cpp LSTM/layers/lstmlayer.cpp LSTM/layers/policy.cpp LSTM/layers/conv.cpp LSTM/layers/pool.cpp LSTM/params.cpp && sbatch MS.slurm

g++ -std=c++11 -O2 -pthread main.cpp MS.cpp solver.cpp common.cpp set.cpp Qlearn.cpp PPO.cpp environment.cpp -I "./LSTM" LSTM/node.cpp LSTM/model.cpp LSTM/PVUnit.cpp LSTM/layer.cpp LSTM/layers/lstmlayer.cpp LSTM/layers/policy.cpp LSTM/layers/conv.cpp LSTM/layers/pool.cpp LSTM/params.cpp && ./a.out

-fsanitize=address ASAN_OPTIONS=detect_leaks=1 -fsanitize=undefined -fno-sanitize-recover=all -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow -fno-sanitize=null -fno-sanitize=alignment

rsync -r MS kevindu@login.rc.fas.harvard.edu:./MultiagentSnake
rsync -r kevindu@login.rc.fas.harvard.edu:./MultiagentSnake/MS .
*/

#include "DQN_MS.h"

#define numThreads 16
// #define numThreads 1

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
    double learnRate = 0.1;
    double momentum = 0.7;
    double regRate = 0;

    Hyperparameters(int numIter_, int numRollout_, int evalPeriod_, double learnRate_, double momentum_, double regRate_){
        numIter = numIter_;
        numRollout = numRollout_;
        evalPeriod = evalPeriod_;
        learnRate = learnRate_;
        momentum = momentum_;
        regRate = regRate_;
    }

    string toString(){
        string s = "";
        s += "numIter: " + to_string(numIter);
        s += " numRollout: " + to_string(numRollout);
        s += " evalPeriod: " + to_string(evalPeriod);
        s += " learnRate: " + to_string(learnRate);
        s += " momentum: " + to_string(momentum);
        s += " regRate: " + to_string(regRate);
        return s;
    }
};

Qlearn* trainers[numThreads];
thread* threads[numThreads];

Hyperparameters hps[numThreads] = {
    // Hyperparameters(10, 300, 1, 0.5, 0.7, 0)
    Hyperparameters(1000, 300, 10, 0.5, 0.7, 0),
    Hyperparameters(1000, 300, 10, 0.1, 0.7, 0),
    Hyperparameters(1000, 300, 10, 0.05, 0.7, 0),
    Hyperparameters(1000, 300, 10, 0.01, 0.7, 0),
    Hyperparameters(1000, 300, 10, 0.5, 0.9, 0),
    Hyperparameters(1000, 300, 10, 0.1, 0.9, 0),
    Hyperparameters(1000, 300, 10, 0.05, 0.9, 0),
    Hyperparameters(1000, 300, 10, 0.01, 0.9, 0),
    Hyperparameters(10000, 30, 100, 0.5, 0.7, 0),
    Hyperparameters(10000, 30, 100, 0.1, 0.7, 0),
    Hyperparameters(10000, 30, 100, 0.05, 0.7, 0),
    Hyperparameters(10000, 30, 100, 0.01, 0.7, 0),
    Hyperparameters(10000, 30, 100, 0.5, 0.9, 0),
    Hyperparameters(10000, 30, 100, 0.1, 0.9, 0),
    Hyperparameters(10000, 30, 100, 0.05, 0.9, 0),
    Hyperparameters(10000, 30, 100, 0.01, 0.9, 0)
};

void runThread(int i){
    trainers[i]->fileOut = "session" + to_string(i) + ".out";
    trainers[i]->train(hps[i].numIter, hps[i].numRollout, hps[i].evalPeriod, hps[i].learnRate, hps[i].momentum, hps[i].regRate);
}

void runAll(){
    for(int i=0; i<numThreads; i++){
        trainers[i] = new Qlearn();
        threads[i] = new thread(runThread, i);
    }
    for(int i=0; i<numThreads; i++){
        threads[i]->join();
    }
    for(int i=0; i<numThreads; i++){
        cout << hps[i].toString() << ' ' << trainers[i]->winRate << '\n';
    }
}

void testSolver(){
    unsigned start_time = time(0);
    Solver s;
    int numGames = 1000;
    int numWins = 0;
    int numGuesses = 0;
    for(int i=0; i<numGames; i++){
        s.state.init();
        s.solve(false, GUESS_RANDOM);
        numGuesses += s.guessCount;
        if(s.state.result == 1) numWins ++;
    }
    cout << "Win rate: " << numWins << " out of " << numGames << '\n';
    cout << "Num guesses: " << numGuesses << '\n';
    {
        numWins = 0;
        for(int i=0; i<numGames; i++){
            s.state.init();
            s.solve(false, GUESS_SET);
            if(s.state.result == 1) numWins ++;
        }
        cout << "Win rate: " << numWins << " out of " << numGames << '\n';
    }
    cout << "Time: " << (time(0) - start_time) << '\n';
}

int main(){

    unsigned start_time = time(0);

    // ofstream fout(gameOut); fout.close();
    // for(int i=0; i<10; i++){
    //     ofstream fout(gameOut, ios::app);
    //     fout << "########################### GAME " << i << " ###############################\n";
    //     fout.close();
    //     Solver s;
    //     s.state.init();
    //     s.solve(true, GUESS_SET);
    // }

    // testSolver();
    
    // runAll();

    PPO trainer;
    trainer.train();

    cout << "Time: " << (time(0) - start_time) << '\n';
}
