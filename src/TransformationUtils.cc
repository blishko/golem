/*
 * Copyright (c) 2021-2023, Martin Blicha <martin.blicha@gmail.com>
 *
 * SPDX-License-Identifier: MIT
 */

#include "TransformationUtils.h"
#include "QuantifierElimination.h"
#include <unordered_set>

bool isTransitionSystem(ChcDirectedGraph const & graph) {
    auto graphRepresentation = AdjacencyListsGraphRepresentation::from(graph);
    auto reversePostorder = reversePostOrder(graph, graphRepresentation);
    // TS has 3 vertices: Init, Body, Bad
    if (reversePostorder.size() != 3) { return false; }
    // TS has 3 edges: From Init to Body, self-loop on Body, from Body to Bad
    auto beg = reversePostorder[0];
    auto loop = reversePostorder[1];
    auto end = reversePostorder[2];
    auto const & begOutEdges = graphRepresentation.getOutgoingEdgesFor(beg);
    if (begOutEdges.size() != 1 || graph.getTarget(begOutEdges[0]) != loop) { return false; }
    auto const & loopOutEdges = graphRepresentation.getOutgoingEdgesFor(loop);
    if (loopOutEdges.size() != 2) { return false; }
    bool selfLoop = std::find_if(loopOutEdges.begin(), loopOutEdges.end(), [&graph, loop](EId eid) {
                        return graph.getTarget(eid) == loop;
                    }) != loopOutEdges.end();
    if (not selfLoop) { return false; }
    bool edgeToEnd = std::find_if(loopOutEdges.begin(), loopOutEdges.end(),
                                  [&graph, end](EId eid) { return graph.getTarget(eid) == end; }) != loopOutEdges.end();
    if (not edgeToEnd) { return false; }
    if (not graphRepresentation.getOutgoingEdgesFor(end).empty()) { return false; }
    return true;
}

std::unique_ptr<TransitionSystem> toTransitionSystem(ChcDirectedGraph const & graph) {
    Logic & logic = graph.getLogic();
    auto adjacencyRepresentation = AdjacencyListsGraphRepresentation::from(graph);
    auto vertices = reversePostOrder(graph, adjacencyRepresentation);
    assert(vertices.size() == 3);
    auto loopNode = vertices[1];
    auto edges = graph.getEdges();
    EId loopEdge = getSelfLoopFor(loopNode, graph, adjacencyRepresentation).value();
    auto edgeVars = getVariablesFromEdge(logic, graph, loopEdge);
    // Now we can continue building the transition system
    auto systemType = systemTypeFrom(edgeVars.stateVars, edgeVars.auxiliaryVars, logic);
    auto stateVars = systemType->getStateVars();
    auto nextStateVars = systemType->getNextStateVars();
    assert(stateVars.size() == edgeVars.stateVars.size());
    assert(nextStateVars.size() == edgeVars.nextStateVars.size());
    PTRef init = PTRef_Undef;
    PTRef transitionRelation = PTRef_Undef;
    PTRef bad = PTRef_Undef;
    graph.forEachEdge([&](DirectedEdge const & edge) {
        auto source = edge.from;
        auto target = edge.to;
        bool isInit = source == vertices[0] && target == vertices[1];
        bool isLoop = source == vertices[1] && target == vertices[1];
        bool isEnd = source == vertices[1] && target == vertices[2];
        assert(isInit || isLoop || isEnd);
        PTRef fla = edge.fla.fla;
        TermUtils utils(logic);
        if (isInit) {
            std::unordered_map<PTRef, PTRef, PTRefHash> subMap;
            std::transform(edgeVars.nextStateVars.begin(), edgeVars.nextStateVars.end(), stateVars.begin(),
                           std::inserter(subMap, subMap.end()),
                           [](PTRef key, PTRef value) { return std::make_pair(key, value); });
            init = utils.varSubstitute(fla, subMap);
            init = QuantifierElimination(logic).keepOnly(init, stateVars);
            //            std::cout << logic.printTerm(init) << std::endl;
        }
        if (isLoop) {
            transitionRelation = transitionFormulaInSystemType(*systemType, edgeVars, fla, logic);
            //            std::cout << logic.printTerm(transitionRelation) << std::endl;
        }
        if (isEnd) {
            std::unordered_map<PTRef, PTRef, PTRefHash> subMap;
            std::transform(edgeVars.stateVars.begin(), edgeVars.stateVars.end(), stateVars.begin(),
                           std::inserter(subMap, subMap.end()),
                           [](PTRef key, PTRef value) { return std::make_pair(key, value); });
            bad = utils.varSubstitute(fla, subMap);
            bad = QuantifierElimination(logic).keepOnly(bad, stateVars);
            //            std::cout << logic.printTerm(bad) << std::endl;
        }
    });
    assert(init != PTRef_Undef && transitionRelation != PTRef_Undef && bad != PTRef_Undef);
    auto ts = std::unique_ptr<TransitionSystem>(
        new TransitionSystem(logic, std::move(systemType), init, transitionRelation, bad));
    return ts;
}

bool strongConnection(std::unordered_set<int> & visitedVertices, std::unordered_set<int> & verticesOnStack,
                      AdjacencyListsGraphRepresentation const & graphRepresentation, ChcDirectedGraph const & graph,
                      SymRef node) {
    visitedVertices.insert(node.x);
    verticesOnStack.insert(node.x);
    auto const & outEdges = graphRepresentation.getOutgoingEdgesFor(node);
    if (size(outEdges) <= 1) {
        verticesOnStack.erase(node.x);
        return false;
    }

    for (EId eid : outEdges) {
        if (graph.getTarget(eid) != node) {
            auto nextVertex = graph.getTarget(eid);
            if (visitedVertices.find(nextVertex.x) == visitedVertices.end()) {
                bool loopFound =
                    strongConnection(visitedVertices, verticesOnStack, graphRepresentation, graph, nextVertex);
                if (loopFound) { return true; }
            } else if (verticesOnStack.find(nextVertex.x) != verticesOnStack.end()) {
                return true;
            }
        }
    }

    verticesOnStack.erase(node.x);

    return false;
}

bool TarjanLoopDetection(ChcDirectedGraph const & graph) {
    auto graphRepresentation = AdjacencyListsGraphRepresentation::from(graph);
    auto vertices = reversePostOrder(graph, graphRepresentation);
    std::unordered_set<int> visitedVertices;
    std::unordered_set<int> verticesOnStack;
    for (uint i = 1; i < vertices.size() - 1; i++) {
        if (visitedVertices.find(vertices[i].x) == visitedVertices.end()) {
            bool loop_detected =
                strongConnection(visitedVertices, verticesOnStack, graphRepresentation, graph, vertices[i]);
            if (loop_detected) { return true; }
        }
    }
    return false;
}

bool isTransitionSystemDAG(ChcDirectedGraph const & graph) {
    if (graph.getVertices().size() < 3) { return false; }
    auto graphRepresentation = AdjacencyListsGraphRepresentation::from(graph);
    auto vertices = reversePostOrder(graph, graphRepresentation);
    assert(graph.getEntry() == vertices[0]);
    bool hasLoop = TarjanLoopDetection(graph);
    if (hasLoop) { return false; }
    for (unsigned i = 1; i < vertices.size() - 1; ++i) {
        auto current = vertices[i];
        auto const & outEdges = graphRepresentation.getOutgoingEdgesFor(current);
        bool hasSelfLoop = false;
        bool hasEdgeToNext = false;
        for (EId eid : outEdges) {
            hasSelfLoop |= graph.getTarget(eid) == current;
            hasEdgeToNext |= graph.getTarget(eid) != current;
        }
        if (not(hasSelfLoop and hasEdgeToNext)) { return false; }
    }
    // There should be no edge from entry to exit
    auto const & entryEdges = graphRepresentation.getOutgoingEdgesFor(graph.getEntry());
    bool canReachExit = std::any_of(entryEdges.begin(), entryEdges.end(),
                                    [&](EId eid) { return graph.getTarget(eid) == graph.getExit(); });
    return not canReachExit;
}

EdgeVariables getVariablesFromEdge(Logic & logic, ChcDirectedGraph const & graph, EId eid) {
    EdgeVariables res;
    TermUtils utils(logic);
    PTRef sourcePred = graph.getStateVersion(graph.getSource(eid));
    PTRef targetPred = graph.getNextStateVersion(graph.getTarget(eid));
    res.stateVars = utils.predicateArgsInOrder(sourcePred);
    res.nextStateVars = utils.predicateArgsInOrder(targetPred);
    PTRef edgeLabel = graph.getEdgeLabel(eid);
    auto allVars = TermUtils(logic).getVars(edgeLabel);
    for (PTRef var : allVars) {
        if (std::find(res.stateVars.begin(), res.stateVars.end(), var) == res.stateVars.end() and
            std::find(res.nextStateVars.begin(), res.nextStateVars.end(), var) == res.nextStateVars.end()) {
            res.auxiliaryVars.push_back(var);
        }
    }
    return res;
}

std::unique_ptr<SystemType> systemTypeFrom(vec<PTRef> const & stateVars, vec<PTRef> const & auxiliaryVars,
                                           Logic & logic) {
    std::vector<SRef> stateVarTypes;
    std::transform(stateVars.begin(), stateVars.end(), std::back_inserter(stateVarTypes),
                   [&logic](PTRef var) { return logic.getSortRef(var); });
    std::vector<SRef> auxVarTypes;
    std::transform(auxiliaryVars.begin(), auxiliaryVars.end(), std::back_inserter(auxVarTypes),
                   [&logic](PTRef var) { return logic.getSortRef(var); });
    return std::make_unique<SystemType>(std::move(stateVarTypes), std::move(auxVarTypes), logic);
}

PTRef transitionFormulaInSystemType(SystemType const & systemType, EdgeVariables const & edgeVars, PTRef edgeLabel,
                                    Logic & logic) {
    std::unordered_map<PTRef, PTRef, PTRefHash> subMap;
    std::transform(edgeVars.stateVars.begin(), edgeVars.stateVars.end(), systemType.getStateVars().begin(),
                   std::inserter(subMap, subMap.end()),
                   [](PTRef key, PTRef value) { return std::make_pair(key, value); });

    std::transform(edgeVars.nextStateVars.begin(), edgeVars.nextStateVars.end(), systemType.getNextStateVars().begin(),
                   std::inserter(subMap, subMap.end()),
                   [](PTRef key, PTRef value) { return std::make_pair(key, value); });

    std::transform(edgeVars.auxiliaryVars.begin(), edgeVars.auxiliaryVars.end(), systemType.getAuxiliaryVars().begin(),
                   std::inserter(subMap, subMap.end()),
                   [](PTRef key, PTRef value) { return std::make_pair(key, value); });
    return TermUtils(logic).varSubstitute(edgeLabel, subMap);
}

bool isTrivial(ChcDirectedGraph const & graph) {
    auto vertices = graph.getVertices();
    assert(not vertices.empty()); // NOTE: We always put Entry in the vertices, even if there are no edges
    if (vertices.size() == 1) {
        assert(vertices[0] == graph.getEntry());
        return true;
    }
    if (vertices.size() != 2) { return false; }
    // We have two vertices, they should be Entry and Exit
    return (vertices[0] == graph.getEntry() or vertices[0] == graph.getExit()) and
           (vertices[1] == graph.getEntry() or vertices[1] == graph.getExit());
}
