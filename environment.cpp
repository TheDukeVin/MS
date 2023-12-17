
#include "DQN_MS.h"

Environment::Environment(){
    time = 0;
    endState = false;
    endValue = -1;
    state.init();
}

string Environment::toString(){
    return "Time: " + to_string(time) + " endValue: " + to_string(endValue) + "\n" + state.printMarks();
}

vector<int> Environment::validActions(){
    vector<int> actions;
    for(auto p : allPos()){
        if(state.getMark(p) == UNMARKED){
            actions.push_back(p.toIndex());
        }
    }
    return actions;
}

void Environment::makeAction(int action){
    state.markEmpty(Pos(action));
    simulator.state = state;
    while(true){
        if(simulator.state.result != 0) break;
        if(simulator.bombLimit()) continue;
        if(simulator.clearLimit()) continue;
        if(simulator.localSearch()) continue;
        simulator.compSet();
        if(simulator.clearSet()) continue;
        break;
    }
    state = simulator.state;
    assert(state.printMarks() == simulator.state.printMarks());
    if(state.result != 0){
        endState = true;
        endValue = state.result == 1;
    }
}

void Environment::inputObservations(LSTM::Data* input){
    for(int i=0; i<input->size; i++){
        input->data[i] = 0;
    }
    for(auto p : allPos()){
        if(state.getMark(p) == UNMARKED) continue;
        int tokenID;
        if(state.getMark(p) == BOMB) tokenID = 9;
        else tokenID = state.getMark(p);
        input->data[tokenID*boardH*boardW + p.toIndex()] = 1;
    }
}