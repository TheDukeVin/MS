
#include "DQN_MS.h"

PPO::PPO(){
    policyStructure = LSTM::Model(LSTM::Shape(boardH, boardW, 10));
    policyStructure.addConv(LSTM::Shape(16, 30, 20), 3, 3);
    policyStructure.addPool(LSTM::Shape(8, 15, 20));
    policyStructure.addConv(LSTM::Shape(8, 14, 40), 3, 3);
    policyStructure.addPool(LSTM::Shape(4, 7, 40));
    policyStructure.addDense(800);
    policyStructure.addOutput(boardH*boardW);
    policyStructure.randomize(0.1);
    policyStructure.resetGradient();

    valueStructure = LSTM::Model(LSTM::Shape(boardH, boardW, 10));
    valueStructure.addConv(LSTM::Shape(16, 30, 20), 3, 3);
    valueStructure.addPool(LSTM::Shape(8, 15, 20));
    valueStructure.addConv(LSTM::Shape(8, 14, 40), 3, 3);
    valueStructure.addPool(LSTM::Shape(4, 7, 40));
    valueStructure.addDense(400);
    valueStructure.addOutput(1);
    valueStructure.randomize(0.1);
    valueStructure.resetGradient();

    policyInput = new LSTM::Data(policyStructure.inputSize);
    policyOutput = new LSTM::Data(policyStructure.outputSize);
    policyNet = LSTM::Model(policyStructure, NULL, policyInput, policyOutput);

    valueInput = new LSTM::Data(valueStructure.inputSize);
    valueOutput = new LSTM::Data(valueStructure.outputSize);
    valueNet = LSTM::Model(valueStructure, NULL, valueInput, valueOutput);
}

vector<PPOStateInstance> PPO::rollout(){
    vector<PPOStateInstance> states;
    Environment env;
    double policy[numActions];
    for(int t=0; t<timeHorizon; t++){
        env.inputObservations(policyInput);
        policyNet.forwardPass();
        computeSoftmaxPolicy(policyOutput->data, numActions, env.validActions(), policy);
        int action = sampleDist(policy, numActions);

        PPOStateInstance instance;
        instance.env = env;
        instance.action = action;
        instance.prevActionProb = policy[action];
        states.push_back(instance);

        env.makeAction(action);
        if(env.endState) break;
    }
    for(int i=0; i<states.size(); i++){
        states[i].value = env.endValue;
    }
    return states;
}

void PPO::generateDataset(){
    for(int i=0; i<datasetSize; i++){
        dataset[i] = rollout();
    }
}

void PPO::updateNet(){
    double clip = 0.2;
    double entropyConstant = 0.01;
    for(int batchID=0; batchID<numBatches; batchID ++){
        for(int it=0; it<batchSize; it++){
            policyNet.resetGradient();
            valueNet.resetGradient();
            int gameIndex = randomN(datasetSize);
            int sampleIndex = randomN(dataset[gameIndex].size());
            PPOStateInstance instance = dataset[gameIndex][sampleIndex];
            instance.env.inputObservations(policyInput);
            policyNet.forwardPass();
            instance.env.inputObservations(valueInput);
            valueNet.forwardPass();

            // Compute value gradient
            valueOutput->gradient[0] = instance.value - valueOutput->data[0];

            // Compute policy gradient

            vector<int> validActions = instance.env.validActions();

            double policy[numActions];
            computeSoftmaxPolicy(policyOutput->data, numActions, validActions, policy);

            for(int i=0; i<numActions; i++){
                policyOutput->gradient[i] = 0;
            }

            // Add entropy bonus

            double entropy = 0;
            for(auto a : validActions){
                entropy += policy[a] * log(policy[a]);
            }
            for(auto a : validActions){
                policyOutput->gradient[a] = policy[a] * (log(policy[a]) - entropy) * entropyConstant;
            }
            
            // Add PPO loss

            double policyRatio = policy[instance.action] / instance.prevActionProb;
            double advantage = instance.value - valueOutput->data[0];

            if((advantage > 0 && policyRatio < 1 + clip) || (advantage < 0 && policyRatio > 1 - clip)){
                for(auto a : validActions){
                    policyOutput->gradient[a] += (policy[a] - (a == instance.action)) * policyRatio * advantage;
                }
            }

            policyNet.backwardPass();
            valueNet.backwardPass();
            policyStructure.accumulateGradient(&policyNet);
            valueStructure.accumulateGradient(&valueNet);
        }

        policyStructure.updateParams(0.01 / batchSize, 0.9, 0.0001);
        valueStructure.updateParams(0.01 / batchSize, 0.9, 0.0001);
    }
}

void PPO::train(){
    for(int it=0; it<20; it++){
        generateDataset();
        // compute score
        double sumScore = 0;
        for(int i=0; i<datasetSize; i++){
            sumScore += dataset[i][0].value;
        }
        cout << "Dataset score: " << (sumScore / datasetSize) << '\n';
        updateNet();
    }
}