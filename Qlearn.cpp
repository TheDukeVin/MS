
#include "DQN_MS.h"

void DataQueue::enqueue(FeatureLabelPair flp){
    queue[index % queueSize] = flp;
    index ++;
}

Qlearn::Qlearn(){
    structure = LSTM::Model(LSTM::Shape(boardH, boardW, 10));
    structure.addConv(LSTM::Shape(16, 30, 20), 3, 3);
    structure.addPool(LSTM::Shape(8, 15, 20));
    structure.addConv(LSTM::Shape(8, 14, 40), 3, 3);
    structure.addPool(LSTM::Shape(4, 7, 40));
    structure.addDense(800);
    structure.addOutput(boardH*boardW);
    structure.randomize(0.1);

    netInput = new LSTM::Data(boardH * boardW * 10);
    netOutput = new LSTM::Data(boardH * boardW);

    net = LSTM::Model(structure, NULL, netInput, netOutput);
}

double Qlearn::rollOut(){
    Environment env;
    net.copyParams(&structure);
    vector<FeatureLabelPair> pairs;
    for(int t=0; t<timeHorizon; t++){
        vector<int> validActions = env.validActions();
        int action;
        if(randUniform() < epsilon){
            action = validActions[randomN(validActions.size())];
        }
        else{
            env.inputObservations(netInput);
            net.forwardPass();
            double maxLogit = -1e+10;
            for(auto a : validActions){
                if(netOutput->data[a] > maxLogit){
                    maxLogit = netOutput->data[a];
                    action = a;
                }
            }
        }

        FeatureLabelPair flp;
        flp.env = env;
        flp.action = action;
        pairs.push_back(flp);

        env.makeAction(action);
        if(env.endState) break;
    }
    for(int i=0; i<pairs.size(); i++){
        pairs[i].value = env.endValue;
        dq.enqueue(pairs[i]);
    }
    return env.endValue;
}

void Qlearn::train(int numIter, int numRollout, int evalPeriod, int batchSize, double learnRate, double momentum, double regRate){
    double scoreSum = 0;
    double totalScore = 0;

    // int numIter = 10000;
    // int numRollout = 10;
    // int evalPeriod = 100;
    // int batchSize = 100;
    // double learnRate = 0.1;
    // double momentum = 0.7;
    // double regRate = 0;

    {
        ofstream fout(fileOut);
        fout.close();
    }
    

    long start_time = time(0);
    for(int it=0; it<numIter; it++){
        if(it > 0 && it % evalPeriod == 0){
            {
                ofstream fout(fileOut, ios::app);
                fout << "Iteration: " << it << " Score: " << (scoreSum / (evalPeriod * numRollout)) << " Time stamp: " << (time(0) - start_time) << '\n';
                fout.close();
            }
            scoreSum = 0;
        }
        //enqueue new rollouts
        for(int i=0; i<numRollout; i++){
            double score = rollOut();
            scoreSum += score;
            totalScore += score;
        }
        structure.resetGradient();
        for(int i=0; i<batchSize; i++){
            net.resetGradient();
            int trainInstance = randomN(min(dq.index, queueSize));
            FeatureLabelPair flp = dq.queue[trainInstance];
            flp.env.inputObservations(netInput);
            net.forwardPass();
            for(int j=0; j<netOutput->size; j++){
                netOutput->gradient[j] = 0;
            }
            double actionLogit = netOutput->data[flp.action];
            double actionQ = LSTM::sigmoid(actionLogit);
            netOutput->gradient[flp.action] = actionQ - flp.value;
            net.backwardPass();
            structure.accumulateGradient(&net);
        }
        structure.updateParams(learnRate / batchSize, momentum, regRate);
    }
    ofstream fout(fileOut, ios::app);
    {
        fout << "Total score: " << (totalScore / (numIter * numRollout)) << '\n';
        fout.close();
    }
    
}