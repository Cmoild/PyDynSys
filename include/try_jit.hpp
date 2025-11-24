#pragma once
#include <string>
#include <llvm/Support/Error.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>

#include <backend/cpu/integrator_cpu.hpp>
#include <memory>

int try_jit(std::string& code, integratedFunc& func, std::unique_ptr<llvm::orc::LLJIT>& jit,
            const std::string name = "lorenz");
