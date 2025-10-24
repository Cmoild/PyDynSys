
.PHONY: build install clean
build:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCLUSTERING_USE_AVX2=ON
	cmake --build build -j4
	stubgen -m dynsys._dynsys -o .

debug:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCLUSTERING_USE_AVX2=OFF
	cmake --build build -j12
	stubgen -m dynsys._dynsys -o .

install:
	python -m pip install -e .

rebuild: clean build install

clean:
	rm -rf build dist *.egg-info
