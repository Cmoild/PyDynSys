#include <iostream>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <memory>
#include <try_jit.hpp>

#include "jit.hpp"

llvm::Expected<std::unique_ptr<llvm::Module>> compileToIR(std::string& codeString,
                                                          llvm::LLVMContext& context) {
    clang::CompilerInstance compiler;
    compiler.createFileManager();
    compiler.createDiagnostics(compiler.getVirtualFileSystem());
    auto& invocation = compiler.getInvocation();

    const char* args[] = {"jit.c"};
    clang::CompilerInvocation::CreateFromArgs(invocation, args, compiler.getDiagnostics());

    compiler.getPreprocessorOpts().addRemappedFile(
        "jit.c", llvm::MemoryBuffer::getMemBuffer(codeString).release());

    // TODO: Add optimization level

    clang::EmitLLVMOnlyAction irAction(&context);
    if (!compiler.ExecuteAction(irAction)) {
        return llvm::make_error<llvm::StringError>("Failed to create IR",
                                                   llvm::inconvertibleErrorCode());
    }

    return irAction.takeModule();
}

int try_jit(std::string& code, integratedFunc& func, std::unique_ptr<llvm::orc::LLJIT>& jit) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    llvm::LLVMContext context;

    auto irModule = compileToIR(code, context);
    if (!irModule) {
        llvm::Error err = llvm::handleErrors(irModule.takeError());
        return 1;
    }

    auto jitOrErr = llvm::orc::LLJITBuilder().create();
    if (!jitOrErr) {
        llvm::Error err = llvm::handleErrors(jitOrErr.takeError());
        return 1;
    }
    jit = std::move(jitOrErr.get());

    jit->getMainJITDylib().addGenerator(
        llvm::cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            jit->getDataLayout().getGlobalPrefix())));

    llvm::orc::ThreadSafeModule tsm(std::move(irModule.get()),
                                    std::make_unique<llvm::LLVMContext>());
    if (auto err = jit->addIRModule(std::move(tsm))) {
        err = llvm::handleErrors(std::move(err));
        return 1;
    }

    auto sym = jit->lookup("lorenz");
    if (!sym) {
        llvm::Error err = llvm::handleErrors(sym.takeError());
        return 1;
    }

    func = (integratedFunc)sym.get().getValue();

    return 0;
}
