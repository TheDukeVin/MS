
#include "DQN_MS.h"

// void DataQueue::enqueue(FeatureLabelPair flp){
//     queue[index % queueSize] = flp;
//     index ++;
// }

Qlearn::Qlearn(){
    structure = LSTM::Model(LSTM::Shape(boardH, boardW, 10));
    structure.addConv(LSTM::Shape(16, 30, 20), 3, 3);
    structure.addPool(LSTM::Shape(8, 15, 20));
    structure.addConv(LSTM::Shape(8, 14, 40), 3, 3);
    structure.addPool(LSTM::Shape(4, 7, 40));
    structure.addDense(800);
    structure.addOutput(boardH*boardW);
    structure.randomize(0.1);
    structure.resetGradient();

    // structure = LSTM::Model(LSTM::Shape(boardH, boardW, 10));
    // structure.addConv(LSTM::Shape(16, 30, 30), 3, 3);
    // structure.addPool(LSTM::Shape(8, 15, 30));
    // structure.addConv(LSTM::Shape(8, 14, 50), 3, 3);
    // structure.addPool(LSTM::Shape(4, 7, 50));
    // structure.addDense(1000);
    // structure.addOutput(boardH*boardW);
    // structure.randomize(0.1);
    // structure.resetGradient();

    netInput = new LSTM::Data(structure.inputSize);
    netOutput = new LSTM::Data(structure.outputSize);
    net = LSTM::Model(structure, NULL, netInput, netOutput);
}

// void Qlearn::initializeNet(int index){
//     assert(index <= net.size());
//     if(index == net.size()){
//         netInput.push_back(new LSTM::Data(boardH * boardW * 10));
//         netOutput.push_back(new LSTM::Data(boardH * boardW));

//         net.push_back(LSTM::Model(structure, NULL, netInput[index], netOutput[index]));
//     }
// }

vector<FeatureLabelPair> Qlearn::rollout(){
    Environment env;
    vector<FeatureLabelPair> pairs;
    int t;
    net.copyParams(&structure);
    for(t=0; t<timeHorizon; t++){
        // initializeNet(t);
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
    }
    return pairs;
}

void Qlearn::train(int numIter, int numRollout, int evalPeriod, double learnRate, double momentum, double regRate){
    double scoreSum = 0;
    double totalScore = 0;

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
        // generate new rollouts and train on them
        for(int i=0; i<numRollout; i++){
            // generate rollout
            vector<FeatureLabelPair> data = rollout();

            // update evaluation variables
            double score = data[0].value;
            scoreSum += score;
            totalScore += score;
            int rolloutLength = data.size();

            // train network
            for(int t=0; t<rolloutLength; t++){
                FeatureLabelPair flp = data[t];
                net.resetGradient();
                flp.env.inputObservations(netInput);
                net.forwardPass();
                for(int j=0; j<netOutput->size; j++){
                    netOutput->gradient[j] = 0;
                }
                double actionLogit = netOutput->data[flp.action];
                double actionQ = LSTM::sigmoid(actionLogit);

                // normalize gradient based on rollout length.
                netOutput->gradient[flp.action] = (actionQ - flp.value) / rolloutLength;
                net.backwardPass();
                structure.accumulateGradient(&net);
            }
        }
        structure.updateParams(learnRate / numRollout, momentum, regRate);
    }
    winRate = totalScore / (numIter * numRollout);
    ofstream fout(fileOut, ios::app);
    {
        fout << "Total score: " << winRate << '\n';
        fout.close();
    }
}