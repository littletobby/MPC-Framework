#include <iostream>

#include "SocketManager.h"
#include "Player.h"
#include "IOManager.h"
#include "BPGraph.h"

int node_type;
SocketManager::SMMLF tel;
int globalRound;
int main(int argc, char** argv) {
    Player::init();
    IOManager::init();
    if (argc < 2) {
        DBGprint("Please enter node type:\n");
        scanf("%d", &node_type);
    }
    else {
        node_type = argv[1][0] - '0';
    }
    DBGprint("node type: %d\n", node_type);
    tel.init();

    BPGraph::LR *bp = new BPGraph::LR(&IOManager::train_data, &IOManager::train_label, &IOManager::test_data, &IOManager::test_label);
    bp->graph();
    bp->train();
    return 0;
}