# setup.py
import os
import sys
import subprocess
import glob
import shutil
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as build_py_orig

PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_NAME = "dynsys"
EXT_BASENAME = "_dynsys"


class CMakeBuildAndStub(build_py_orig):
    def run(self):
        build_type = os.environ.get("BUILD", "").lower()
        if build_type in ("debug", "d"):
            cfg = "Debug"
        else:
            cfg = "Release"

        cmake_generator = os.environ.get("CMAKE_GENERATOR")
        vcpkg_toolchain = os.environ.get("VCPKG_TOOLCHAIN_FILE")
        llvm_dir = os.environ.get("LLVM_DIR")

        build_dir = PROJECT_ROOT / "build"
        build_dir.mkdir(exist_ok=True)

        cmake_cmd = ["cmake", "-S", str(PROJECT_ROOT), "-B", str(build_dir)]
        cmake_cmd.append(f"-DCMAKE_BUILD_TYPE={cfg}")

        cmake_cmd += [
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            "-DCLUSTERING_USE_AVX2=ON",
        ]

        if vcpkg_toolchain:
            cmake_cmd.append(f"-DCMAKE_TOOLCHAIN_FILE={vcpkg_toolchain}")
        if llvm_dir:
            cmake_cmd.append(f"-DLLVM_DIR={llvm_dir}")

        if cmake_generator:
            cmake_cmd.extend(["-G", cmake_generator])

        print("[setup.py] cmake configure:", " ".join(cmake_cmd))
        try:
            subprocess.check_call(cmake_cmd, cwd=str(PROJECT_ROOT))
        except Exception as e:
            raise RuntimeError("cmake configure failed") from e

        cmake_build_cmd = [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            cfg,
            "--parallel",
        ]
        jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL") or os.environ.get("JOBS")
        if jobs:
            cmake_build_cmd.extend([f"-j{jobs}"])
        print("[setup.py] cmake build:", " ".join(cmake_build_cmd))
        try:
            subprocess.check_call(cmake_build_cmd, cwd=str(PROJECT_ROOT))
        except Exception as e:
            raise RuntimeError("cmake build failed") from e

        patterns = [
            str(PROJECT_ROOT / PACKAGE_NAME / f"{EXT_BASENAME}.*"),
            str(build_dir / "**" / f"{EXT_BASENAME}.*"),
            str(build_dir / "**" / f"{PACKAGE_NAME}" / f"{EXT_BASENAME}.*"),
        ]
        matches = []
        for pat in patterns:
            matches.extend(glob.glob(pat, recursive=True))
        matches.extend(
            glob.glob(str(build_dir / "**" / f"*{EXT_BASENAME}*.*"), recursive=True)
        )

        if not matches:
            raise RuntimeError("Error: _dynsys compiled module not found")
        matches = sorted(set(matches))
        built_path = Path(matches[-1]).resolve()
        print(f"[setup.py] Found built extension: {built_path}")

        target_pkg_dir = Path(self.build_lib) / PACKAGE_NAME
        target_pkg_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_pkg_dir / built_path.name
        print(f"[setup.py] Copying {built_path} -> {target_file}")
        shutil.copy2(str(built_path), str(target_file))

        try:
            print("[setup.py] Running stubgen for dynsys._dynsys")
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "mypy.stubgen",
                    "-m",
                    f"{PACKAGE_NAME}.{EXT_BASENAME}",
                    "-o",
                    str(target_pkg_dir),
                ]
            )
        except subprocess.CalledProcessError:
            print(
                "[setup.py] Warning: stubgen failed (mypy.stubgen). Continuing without .pyi files."
            )
        except Exception as e:
            print("[setup.py] Warning: stubgen not available or failed:", e)

        for pyi in (PROJECT_ROOT / PACKAGE_NAME).glob("*.pyi"):
            shutil.copy2(str(pyi), str(target_pkg_dir / pyi.name))

        super().run()


setup(
    name="dynsys",
    version="0.1.0",
    description="Dynamics simulation library (C++ core + pybind11)",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    package_data={"dynsys": ["_dynsys.*", "*.pyi"]},
    install_requires=["numpy>=2.3.4", "lark>=1.3.0"],
    python_requires=">=3.13",
    cmdclass={"build_py": CMakeBuildAndStub},
    zip_safe=False,
)
