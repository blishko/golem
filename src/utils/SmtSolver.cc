/*
 * Copyright (c) 2023-2024, Martin Blicha <martin.blicha@gmail.com>
 *
 * SPDX-License-Identifier: MIT
 */

#include "SmtSolver.h"

SMTSolver::SMTSolver(Logic & logic, WitnessProduction setup) {
    bool produceModel = setup == WitnessProduction::ONLY_MODEL || setup == WitnessProduction::MODEL_AND_INTERPOLANTS;
    bool produceInterpolants =
        setup == WitnessProduction::ONLY_INTERPOLANTS || setup == WitnessProduction::MODEL_AND_INTERPOLANTS;
    const char * msg = "ok";
    this->config.setOption(SMTConfig::o_produce_models, SMTOption(produceModel), msg);
    this->config.setOption(SMTConfig::o_produce_inter, SMTOption(produceInterpolants), msg);
    solver = std::make_unique<MainSolver>(logic, config, "");
}

void SMTSolver::resetSolver() {
    solver = std::make_unique<MainSolver>(solver->getLogic(), config, "");
}

void SMTSolver::assertProp(PTRef prop) {
    solver->insertFormula(prop);
}

SMTSolver::Answer SMTSolver::check() {
    auto res = solver->check();
    if (res == s_True) { return Answer::SAT; }
    if (res == s_False) { return Answer::UNSAT; }
    if (res == s_Error) { return Answer::ERROR; }
    return Answer::UNKNOWN;
}

void SMTSolver::push() {
    solver->push();
}

void SMTSolver::pop() {
    solver->pop();
}
