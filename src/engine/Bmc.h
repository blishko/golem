/*
 * Copyright (c) 2020-2022, Martin Blicha <martin.blicha@gmail.com>
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef GOLEM_BMC_H
#define GOLEM_BMC_H

#include "Engine.h"
#include "TransitionSystem.h"

class BMC : public Engine {
    Logic & logic;
//    Options const & options;
    int verbosity = 0;
public:

    BMC(Logic & logic, Options const & options) : logic(logic) {
        verbosity = std::stoi(options.getOrDefault(Options::VERBOSE, "0"));
    }

    virtual VerificationResult solve(ChcDirectedHyperGraph const & graph) override {
        if (graph.isNormalGraph()) {
            auto normalGraph = graph.toNormalGraph();
            return solve(*normalGraph);
        }
        return VerificationResult(VerificationAnswer::UNKNOWN);
    }

    VerificationResult solve(ChcDirectedGraph const & graph);

private:
    VerificationResult solveTransitionSystem(ChcDirectedGraph const & graph);
    TransitionSystemVerificationResult solveTransitionSystemInternal(TransitionSystem const & system);
};


#endif //GOLEM_BMC_H
