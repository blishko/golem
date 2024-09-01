/*
 * Copyright (c) 2023, Martin Blicha <martin.blicha@gmail.com>
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef GOLEM_NESTED_LOOP_TRANSFORMATION_H
#define GOLEM_NESTED_LOOP_TRANSFORMATION_H

#include "TransitionSystem.h"
#include "Witnesses.h"
#include "graph/ChcGraph.h"

/**
 * This class implements a transformation from a general linear CHC system to a linear CHC system without
 * nested loops.
 * The implementation is influenced by the algorithm used in SingleLoopTransformation.cc
 *
 * The idea behind the transformation is the following.
 * First we parse CHC DAG, detecting loops.
 * We introduce a separate boolean variable for each location inside of the loop (predicate, vertex in the CHC graph).
 * For each predicate and each argument of the predicate we add a new state variable of the transition system.
 * Then, for each edge (clause), we create a new disjunct of the transition relation.
 * This fragment captures the changes to the variables as defined by the edge constraint.
 * The fragment enforces that the current source location is variable is true and next target variable is true.
 * All other location variables are false (or alternatively keep their previous value).
 * The edge's constraint is translated to be over corresponding state variables (state vars from source, next state
 * vars from target), all other variables are unchanged.
 *
 * Backtranslating witness from transition system is currently based on the following reasoning:
 * To backtranslate proof, we have to identify which disjunct of the transition relation was used to make each step
 * in the error trace of the transition system. From that we can deduce which edge needs to be taken
 * (and with what values) in the CHC graph.
 *
 * For the model, we compute restriction of the TS invariant to each location by substituting appropriate constant
 * values for location variables. This may still leave some undesired variables in the invariant. We currently make
 * best effort to eliminate these variables by simplifying the formula.
 */
class NestedLoopTransformation {


public:

    // Helper types
    struct VarPosition {
        SymRef vertex;
        uint32_t pos;

        inline bool operator==(VarPosition other) const { return vertex == other.vertex and pos == other.pos; }
    };
    struct VarPositionHasher {
        std::size_t operator()(VarPosition pos) const {
            std::hash<std::uint32_t> hasher;
            return hasher(pos.vertex.x) ^ hasher(pos.pos);
        }
    };


    using LocationVarMap = std::unordered_map<SymRef, PTRef, SymRefHash>;
    using PositionVarMap = std::unordered_map<VarPosition, PTRef, VarPositionHasher>;


    class WitnessBackTranslator {
        ChcDirectedGraph const & initialGraph;
        ChcDirectedGraph const & graph;
        LocationVarMap locationVarMap;
        PositionVarMap positionVarMap;

    public:
        WitnessBackTranslator(ChcDirectedGraph const & initialGraph, ChcDirectedGraph const & graph,
                              LocationVarMap && locationVarMap, PositionVarMap && positionVarMap)
            : graph(graph), initialGraph(initialGraph), locationVarMap(std::move(locationVarMap)),
              positionVarMap(std::move(positionVarMap)) {}

        VerificationResult translate(TransitionSystemVerificationResult result);

    private:
        template<typename T> using ErrorOr = std::variant<NoWitness, T>;

        ErrorOr<InvalidityWitness> translateErrorPath(std::size_t unrolling);

        ErrorOr<ValidityWitness> translateInvariant(PTRef inductiveInvariant);

        std::unordered_set<PTRef, PTRefHash> getVarsForVertex(SymRef vertex) const;
    };


    std::vector<EId> detectLoop(ChcDirectedGraph const & graph);
    void simplifyLoop(ChcDirectedGraph & graph, std::vector<EId> loop);

    void transform(ChcDirectedGraph & graph);

};

#endif // GOLEM_NESTED_LOOP_TRANSFORMATION_H
