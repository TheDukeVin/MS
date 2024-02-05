
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

double PPO::generateDataset(){
    cout << "Generating dataset...\n";
    dataset = vector<PPOStateInstance>();
    int numGames = 0;
    double sumScore = 0;
    while(dataset.size() < datasetSize){
        vector<PPOStateInstance> new_states = rollout();
        numGames ++;
        sumScore += new_states[0].value;
        for(auto state : new_states){
            dataset.push_back(state);
        }
    }
    return sumScore / numGames;
    // for(int i=0; i<datasetSize; i++){
    //     dataset[i] = rollout();
    // }
}

double clip(double val, double leftLim, double rightLim){
    if(val < leftLim) return leftLim;
    if(val > rightLim) return rightLim;
    return val;
}

void PPO::updateValueNet(string outFile){
    double valueLoss = 0;

    ofstream valueOut(outFile);
    for(int batchID=0; batchID<numBatches; batchID ++){
        for(int it=0; it<batchSize; it++){
            valueNet.resetGradient();

            // int gameIndex = randomN(datasetSize);
            // int sampleIndex = randomN(dataset[gameIndex].size());
            // PPOStateInstance instance = dataset[gameIndex][sampleIndex];
            int sampleIndex = randomN(datasetSize);
            PPOStateInstance instance = dataset[sampleIndex];

            instance.env.inputObservations(valueInput);
            valueNet.forwardPass();
            double networkValue = LSTM::sigmoid(valueOutput->data[0]);

            // Compute value gradient
            valueOutput->gradient[0] = networkValue - instance.value;
            valueLoss += pow(networkValue - instance.value, 2);

            valueNet.backwardPass();
            valueStructure.accumulateGradient(&valueNet);
        }
        if(batchID % 10 == 9){
            if(batchID > 10){
                valueOut << ',';
            }
            valueOut << valueLoss;
            valueLoss = 0;
        }
        

        valueStructure.updateParams(0.01 / batchSize, 0.9, 0.0001);
        valueNet.copyParams(&valueStructure);
    }
    valueOut << '\n';
    // ofstream fout ("MS.out");
    // for(int i=0; i<30; i++){
    //     PPOStateInstance instance = dataset[i];
    //     fout << instance.env.toString();
    //     instance.env.inputObservations(valueInput);
    //     valueNet.forwardPass();
    //     double networkValue = LSTM::sigmoid(valueOutput->data[0]);
    //     fout << "Network Value: " << networkValue << " End Value: " << instance.value << '\n';
    // }
}

void PPO::updatePolicyNet(string outFile){
    double clipRange = 0.2;
    double entropyConstant = 0.01;
    double policyLoss = 0;

    ofstream policyOut(outFile);
    for(int batchID=0; batchID<numBatches; batchID ++){
        for(int it=0; it<batchSize; it++){
            policyNet.resetGradient();

            // int gameIndex = randomN(datasetSize);
            // int sampleIndex = randomN(dataset[gameIndex].size());
            // PPOStateInstance instance = dataset[gameIndex][sampleIndex];
            int sampleIndex = randomN(datasetSize);
            PPOStateInstance instance = dataset[sampleIndex];

            instance.env.inputObservations(policyInput);
            policyNet.forwardPass();
            instance.env.inputObservations(valueInput);
            valueNet.forwardPass();
            double networkValue = LSTM::sigmoid(valueOutput->data[0]);

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
            policyLoss += entropy * entropyConstant;
            for(auto a : validActions){
                policyOutput->gradient[a] = policy[a] * (log(policy[a]) - entropy) * entropyConstant;
            }
            
            // Add PPO loss

            double policyRatio = policy[instance.action] / instance.prevActionProb;
            double advantage = instance.value - networkValue;

            if((advantage > 0 && policyRatio < 1 + clipRange) || (advantage < 0 && policyRatio > 1 - clipRange)){
                for(auto a : validActions){
                    policyOutput->gradient[a] += (policy[a] - (a == instance.action)) * policyRatio * advantage;
                }
            }

            policyLoss -= min(policyRatio * advantage, clip(policyRatio, 1-clipRange, 1+clipRange) * advantage);

            policyNet.backwardPass();
            policyStructure.accumulateGradient(&policyNet);
        }
        if(batchID % 10 == 9){
            if(batchID > 10){
                policyOut << ',';
            }
            policyOut << policyLoss;
            policyLoss = 0;
        }

        policyStructure.updateParams(0.01 / batchSize, 0.9, 0.0001);
        policyNet.copyParams(&policyStructure);
    }
    policyOut << '\n';
}

void PPO::train(){
    ofstream fout("control.out");
    fout << "Running PPO training...\n";
    fout.close();
    unsigned start_time = time(0);
    for(int it=0; it<10; it++){
        double score = generateDataset();
        {
            ofstream fout("control.out", ios::app);
            fout << "Dataset score: " << score << '\n';
            fout << "Updating value net:\n";
            fout.close();
        }
        updateValueNet("value" + to_string(it) + ".out");
        {
            ofstream fout("control.out", ios::app);
            fout << "Updating policy net:\n";
            fout.close();
        }
        updatePolicyNet("policy" + to_string(it) + ".out");
        {
            ofstream fout("control.out", ios::app);
            fout << "Time stamp: " << (time(0) - start_time) << '\n';
            fout.close();
        }
    }
}