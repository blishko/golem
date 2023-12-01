/*
 * Copyright (c) 2023, Matias Barandiaran <matias.barandiaran03@gmail.com>
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef GOLEM_PROOFSTEPS_H
#define GOLEM_PROOFSTEPS_H
#include "Term.h"
#include "Witnesses.h"
#include "graph/ChcGraph.h"
#include "utils/SmtSolver.h"
#include <memory>
#include <utility>

class Step {
public:
    enum stepType { ASSUME, STEP, ANCHOR };

private:
    std::size_t stepId;
    stepType type;
    std::vector<std::shared_ptr<Term>> clause;
    std::string rule;
    std::vector<std::size_t> premises;
    std::vector<std::pair<std::string, std::string>> args;

public:
    Step(std::size_t stepId, stepType type, std::vector<std::shared_ptr<Term>> clause, std::string rule,
         std::vector<std::size_t> premises)
        : stepId(stepId), type(type), clause(std::move(clause)), rule(std::move(rule)), premises(std::move(premises)) {}
    Step(std::size_t stepId, stepType type, std::vector<std::shared_ptr<Term>> clause, std::string rule,
         std::vector<std::pair<std::string, std::string>> args)
        : stepId(stepId), type(type), clause(std::move(clause)), rule(std::move(rule)), args(std::move(args)) {}
    Step(std::size_t stepId, stepType type, std::vector<std::shared_ptr<Term>> clause, std::string rule)
        : stepId(stepId), type(type), clause(std::move(clause)), rule(std::move(rule)) {}
    Step(std::size_t stepId, stepType type, std::vector<std::shared_ptr<Term>> clause)
        : stepId(stepId), type(type), clause(std::move(clause)), rule(" ") {}
    Step(std::size_t stepId, stepType type, std::string rule, std::vector<std::size_t> premises)
        : stepId(stepId), type(type), rule(std::move(rule)), premises(std::move(premises)) {}
    std::string printStepAlethe() const;
    std::string printStepIntermediate() const;
};

class Observer {
public:
    virtual void update(Step const & step) = 0;
};

class AlethePrintObserver : public Observer {
    std::ostream & out;

public:
    explicit AlethePrintObserver(std::ostream & out) : out(out) {}
    void update(Step const & step) override { out << step.printStepAlethe(); }
};

class [[maybe_unused]] CountingObserver : public Observer {
    std::size_t count = 0;

public:
    CountingObserver() = default;
    ~CountingObserver() { std::cout << count << std::endl; }
    void update(Step const &) override { ++count; }
};

class IntermediatePrintObserver : public Observer {
    std::ostream & out;

public:
    explicit IntermediatePrintObserver(std::ostream & out) : out(out) {}
    void update(Step const & step) override { out << step.printStepIntermediate(); }
};

class StepHandler {

    InvalidityWitness::Derivation derivation;
    std::vector<std::shared_ptr<Term>> originalAssertions;
    Normalizer::Equalities const & normalizingEqualities;
    Logic & logic;
    ChcDirectedHyperGraph originalGraph;

    std::vector<Observer *> observers;

    std::size_t currentStep = 0;
    std::size_t trueRuleStep = 0;

    std::shared_ptr<Term> implicationRHS;
    std::shared_ptr<Term> implicationLHS;
    std::shared_ptr<Term> currTerm;
    std::vector<std::size_t> modusPonensSteps; // Modus Ponens Steps to derive the next node

    // Visitors
    OperateLetTermVisitor operateLetTermVisitor;
    LetLocatorVisitor letLocatorVisitor;
    RemoveUnusedVisitor removeUnusedVisitor;

public:
    StepHandler(InvalidityWitness::Derivation derivation, std::vector<std::shared_ptr<Term>> originalAssertions,
                Normalizer::Equalities const & normalizingEqualities, Logic & logic,
                ChcDirectedHyperGraph originalGraph)
        : derivation(std::move(derivation)), originalAssertions(std::move(originalAssertions)),
          normalizingEqualities(normalizingEqualities), logic(logic), originalGraph(std::move(originalGraph)) {}

    std::vector<std::pair<std::string, std::string>> getInstPairs(std::size_t it,
                                                                  vec<Normalizer::Equality> const & stepNormEq);
    static std::vector<std::shared_ptr<Term>> packClause(std::shared_ptr<Term> const & term);
    static std::vector<std::shared_ptr<Term>> packClause(std::shared_ptr<Term> const & term1,
                                                         std::shared_ptr<Term> const & term2);
    void buildAletheProof();
    void buildIntermediateProof();

    void instantiationSteps(std::size_t i);
    void assumptionSteps();
    void directSimplification(std::vector<std::size_t> requiredMP, std::size_t implicationStep,
                              std::shared_ptr<Term> const & lastClause, std::shared_ptr<Term> const & renamedImpLHS);
    void conjunctionSimplification(std::vector<std::size_t> requiredMP, std::shared_ptr<Term> const & finalClause,
                                   std::size_t implicationStep, std::shared_ptr<Term> const & renamedImpLHS);

    void stepReusage(std::shared_ptr<Term> const & term);

    void registerObserver(Observer * observer) { observers.push_back(observer); }

    void deRegisterObserver(Observer * observer) {
        for (std::size_t i = 0; i < observers.size(); i++) {
            if (observer == observers[i]) { observers.erase(observers.begin() + int(i)); }
        }
    }

    void notifyObservers(Step const & step) {
        for (Observer * observer : observers) { // notify all observers
            observer->update(step);
        }
    }
};

#endif // GOLEM_PROOFSTEPS_H
