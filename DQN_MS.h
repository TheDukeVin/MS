
#include "MS.h"
#include "LSTM/lstm.h"

#define queueSize 10000
#define timeHorizon (boardH*boardW)
#define epsilon 0.01

#ifndef DQN_MS_h
#define DQN_MS_h

/*
Environment template:

class Environment{
private:

public:
    int time;
    bool endState;
    double endValue;

    Environment();
    string toString();

    vector<int> validActions();
    void makeAction(int action);

    void inputObservations(LSTM::Data* input);
};
*/

class Environment{
private:
    MS state;
    Solver simulator;

public:
    int time;
    bool endState;
    double endValue;

    Environment();
    string toString();

    vector<int> validActions();
    void makeAction(int action);

    void inputObservations(LSTM::Data* input);
};

class FeatureLabelPair{
public:
    Environment env;
    int action;
    double value;
};

class DataQueue{
public:
    FeatureLabelPair queue[queueSize];
    long index = 0;

    void enqueue(FeatureLabelPair flp);
};

class Qlearn{
public:
    LSTM::Model structure;
    DataQueue dq;

    LSTM::Model net;
    LSTM::Data* netInput;
    LSTM::Data* netOutput;

    Qlearn();
    double rollOut();

    string fileOut;

    void train(int numIter, int numRollout, int evalPeriod, int batchSize, double learnRate, double momentum, double regRate);
};

#endif