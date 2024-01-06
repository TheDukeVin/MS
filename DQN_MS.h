
#include "MS.h"
#include "LSTM/lstm.h"

#define queueSize 10000
#define timeHorizon (boardH*boardW)
#define epsilon 0.01

#define numActions (boardH*boardW)

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

void computeSoftmaxPolicy(double* logits, int size, vector<int> validActions, double* policy); // -1 means invalid action.
int sampleDist(double* dist, int N); // -1 represents an invalid value.

class FeatureLabelPair{
public:
    Environment env;
    int action;
    double value;
};

// class DataQueue{
// public:
//     FeatureLabelPair queue[queueSize];
//     long index = 0;

//     void enqueue(FeatureLabelPair flp);
// };

class Qlearn{
public:
    LSTM::Model structure;
    // DataQueue dq;

    // vector<LSTM::Model> net;
    // vector<LSTM::Data*> netInput;
    // vector<LSTM::Data*> netOutput;

    LSTM::Model net;
    LSTM::Data* netInput;
    LSTM::Data* netOutput;

    Qlearn();
    // void initializeNet(int index);

    vector<FeatureLabelPair> rollout(); // returns feature-label pairs for each state

    string fileOut;

    double winRate; // train() edits this variable
    void train(int numIter, int numRollout, int evalPeriod, double learnRate, double momentum, double regRate);
};

class PPOStateInstance{
public:
    Environment env;
    int action;
    double prevActionProb;
    double value;
};

class PPO{
public:
    LSTM::Model policyStructure;
    LSTM::Model valueStructure;

    const static int datasetSize = 1000;
    const int numBatches = 300;
    const int batchSize = 30;

    LSTM::Model policyNet;
    LSTM::Data* policyInput;
    LSTM::Data* policyOutput;

    LSTM::Model valueNet;
    LSTM::Data* valueInput;
    LSTM::Data* valueOutput;

    vector<PPOStateInstance> dataset[datasetSize];

    PPO();

    // Actor-critic training cadence
    vector<PPOStateInstance> rollout();
    void generateDataset();
    void updateNet();

    void train();
};

#endif