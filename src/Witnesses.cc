/*
 * Copyright (c) 2020-2022, Martin Blicha <martin.blicha@gmail.com>
 *
 * SPDX-License-Identifier: MIT
 */

#include "Witnesses.h"

#include "TransformationUtils.h"

SystemInvalidityWitness graphToSystemInvalidityWitness(InvalidityWitness const & witness, ChcDirectedGraph & graph) {
    using Derivation = SystemInvalidityWitness::Derivation;
    Logic & logic = graph.getLogic();
    Derivation derivation;
    using DerivationStep = Derivation::DerivationStep;
    auto const & path = witness.getErrorPath();
    if (path.isEmpty()) { return SystemInvalidityWitness(); }
    auto edgeIds = path.getEdges();
    struct UPHelper{
        int counter = 0;
        ChcDirectedGraph const & graph;
        TimeMachine timeMachine;

        PTRef operator()(SymRef vertex) {
            PTRef term = graph.getStateVersion(vertex);
            return timeMachine.sendFlaThroughTime(term, counter++);
        }

        UPHelper(ChcDirectedGraph const & graph, Logic & logic) : graph(graph), timeMachine(logic) {}
    };
    std::vector<SymRef> vertices;
    std::transform(edgeIds.begin(), edgeIds.end(), std::back_inserter(vertices),
                   [&graph](EId eid) { return graph.getSource(eid); });
    vertices.push_back(graph.getTarget(edgeIds.back()));
    std::vector<PTRef> vertexPredicates;
    std::transform(vertices.begin(), vertices.end(), std::back_inserter(vertexPredicates), UPHelper(graph, logic));
    std::size_t stepCounter = 0;
    for (std::size_t i = 0; i < edgeIds.size(); ++i) {
        std::vector<UninterpretedPredicate> uninterpretedPart;
        if (vertexPredicates[i] != logic.getTerm_true()) { uninterpretedPart.push_back(UninterpretedPredicate{.predicate = vertexPredicates[i]}); }
        PTRef edgeFla = graph.getEdgeLabel(edgeIds[i]);
        PTRef interpretedPart = TimeMachine(logic).sendFlaThroughTime(edgeFla, i);
        DerivationStep step;
        step.index = stepCounter++;
        step.type = DerivationStep::StepType::INPUT;
        step.clause = ChClause{ChcHead{UninterpretedPredicate{vertexPredicates[i + 1]}},
                               ChcBody{{interpretedPart}, std::move(uninterpretedPart)}
        };
        derivation.addDerivationStep(std::move(step));
    }
    auto hasOnlyInterpretedBody = [&logic](ChClause const & clause) {
        auto const & uninterpreted = clause.body.uninterpretedPart;
        return uninterpreted.empty() || (uninterpreted.size() == 1 && uninterpreted[0].predicate == logic.getTerm_true());
    };
    assert(hasOnlyInterpretedBody(derivation[0].clause));
    if (not hasOnlyInterpretedBody(derivation[0].clause)) {
        throw std::logic_error("Unexpected error in building the invalidity witness");
    }
    std::size_t previousDerivation = 0;
    for (std::size_t i = 1; i < edgeIds.size(); ++i) {
        DerivationStep step;
        step.index = stepCounter++;
        step.type = decltype(step.type)::DERIVED;
        step.nucleus = i;
        step.satellites = {previousDerivation};
        DerivationStep const & satellite = derivation[previousDerivation];
        DerivationStep const & nucleus = derivation[step.nucleus];
        (void)nucleus;
        assert(hasOnlyInterpretedBody(satellite.clause));
        assert(nucleus.clause.body.uninterpretedPart.size() == 1);
        assert(satellite.clause.head.predicate == nucleus.clause.body.uninterpretedPart[0]);
        step.clause.head = derivation[i].clause.head;
        step.clause.body = ChcBody{{logic.mkAnd(satellite.clause.body.interpretedPart.fla, derivation[i].clause.body.interpretedPart.fla)},
            {satellite.clause.body.uninterpretedPart}
        };
        previousDerivation = step.index;
        derivation.addDerivationStep(std::move(step));
    }
    auto const & derivedClause = derivation.last().clause;
    SystemInvalidityWitness sysWitness;
    // build the model
    if (derivedClause.head.predicate.predicate != logic.getTerm_false() || not(hasOnlyInterpretedBody(derivedClause))) {
        // error in derivation, no valid witness
        throw std::logic_error("Error in computing the invalidity witness");
    }
    PTRef antecedent = derivedClause.body.interpretedPart.fla;
    {
        SMTConfig config;
        MainSolver solver(logic, config, "CEX computer");
        solver.insertFormula(antecedent);
        auto res = solver.check();
        if (res != s_True) { throw std::logic_error("Error in computing the invalidity witness"); }
        auto model = solver.getModel();
        assert(model->evaluate(antecedent) == logic.getTerm_true());
        sysWitness.setModel(WitnessModel(std::move(model)));
    }
    sysWitness.setDerivation(std::move(derivation));
    return sysWitness;
}

void SystemInvalidityWitness::print(std::ostream & out, Logic & logic) const {
    auto derivationSize = derivation.size();
    ChcPrinter printer(logic, out);
    for (std::size_t i = 0; i < derivationSize; ++i) {
        auto const & step = derivation[i];
        out << i << ".\t";
        printer.print(step.clause);
        if (step.type == Derivation::DerivationStep::StepType::DERIVED) {
            out << "From steps: " << step.nucleus;
            for (auto index : step.satellites) {
                out << ' ' << index;
            }
            out << '\n';
        }
        out << '\n';
    }
    out << "\nWith model:\n";
    auto vars = TermUtils(logic).getVars(derivation.last().clause.body.interpretedPart.fla);
    for (auto var : vars) {
        out << logic.printTerm(var) << " := " << logic.printTerm(model.evaluate(var)) << '\n';
    }
    out << std::endl;
}

void ValidityWitness::print(std::ostream & out, Logic & logic) const {
    for (auto && [predicate, definition] : interpretations) {
        out << "  (define-fun " << logic.getSymName(predicate) << " (";
        const auto & args = TermUtils(logic).predicateArgsInOrder(predicate);
        for (std::size_t i = 0; i < args.size(); ++i) {
            auto sortString = logic.printSort(logic.getSortRef(args[i]));
            out << "(" << logic.protectName(logic.getSymRef(args[i])) << " " << sortString << ")" << (i == args.size()-1 ? "" : " ");
        }
        assert(logic.getSortRef(predicate) == logic.getSort_bool());
        out << ")" << " " << logic.printSort(logic.getSortRef(predicate)) << "\n";
        out << "    " << logic.printTerm(definition) << ")\n";
    }
}

InvalidityWitness InvalidityWitness::fromTransitionSystem(const ChcDirectedGraph & graph, std::size_t unrollings) {
    if (not isTransitionSystem(graph)) { return {}; }
    auto adjacencyList = AdjacencyListsGraphRepresentation::from(graph);
    auto vertices = graph.getVertices();
    assert(vertices.size() == 3);
    auto loopingVertex = *std::find_if(vertices.begin(), vertices.end(), [&](SymRef sym) {
        return sym != graph.getEntry() and sym != graph.getExit();
    });
    auto loopingVertexIncoming = adjacencyList.getIncomingEdgesFor(loopingVertex);
    auto it = std::find_if(loopingVertexIncoming.begin(), loopingVertexIncoming.end(), [&graph](EId eid) {
        return graph.getSource(eid) == graph.getTarget(eid);
    });
    assert(it != loopingVertexIncoming.end());
    EId loopingEdge = *it;
    EId startEdge = adjacencyList.getOutgoingEdgesFor(graph.getEntry())[0];
    EId finalEdge = adjacencyList.getIncomingEdgesFor(graph.getExit())[0];
    InvalidityWitness witness;
    InvalidityWitness::ErrorPath path;
    std::vector<EId> pathEdges(unrollings, loopingEdge);
    pathEdges.push_back(finalEdge);
    pathEdges.insert(pathEdges.begin(), startEdge);
    path.setPath(std::move(pathEdges));
    witness.setErrorPath(std::move(path));
    return witness;
}

ValidityWitness
ValidityWitness::fromTransitionSystem(Logic & logic, ChcDirectedGraph const & graph,
                                      TransitionSystem const & transitionSystem, PTRef inductiveInvariant) {
    auto vertices = graph.getVertices();
    assert(vertices.size() == 3);
    auto vertex = vertices[2];
    assert(vertex != graph.getEntry() and vertex != graph.getExit());
    TermUtils utils(logic);
    TimeMachine timeMachine(logic);
    TermUtils::substitutions_map subs;
    auto graphVars = utils.predicateArgsInOrder(graph.getStateVersion(vertex));
    auto systemVars = transitionSystem.getStateVars();
    vec<PTRef> unversionedVars;
    assert(graphVars.size() == systemVars.size());
    for (std::size_t i = 0; i < graphVars.size(); ++i) {
        unversionedVars.push(timeMachine.getUnversioned(graphVars[i]));
        subs.insert({systemVars[i], unversionedVars.last()});
    }
    PTRef graphInvariant = utils.varSubstitute(inductiveInvariant, subs);
//    std::cout << "Graph invariant: " << logic.printTerm(graphInvariant) << std::endl;
    PTRef unversionedPredicate = logic.mkUninterpFun(vertex, std::move(unversionedVars));
    ValidityWitness::definitions_t definitions{{unversionedPredicate, graphInvariant}};
    return ValidityWitness(std::move(definitions));
}