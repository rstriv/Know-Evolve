GNN_HOME=../graphnn_base

include $(GNN_HOME)/make_common

lib_dir = $(GNN_HOME)/build/lib
gnn_lib = $(lib_dir)/libgnn.a

include_dirs = $(CUDA_HOME)/include $(MKL_ROOT)/include $(GNN_HOME)/include/matrix $(GNN_HOME)/include/graphnn ./include ${EIGEN_ROOT}

CXXFLAGS += $(addprefix -I,$(include_dirs)) -Wno-unused-local-typedef

cpp_files = $(shell $(FIND) src/lib -name "*.cpp" -printf "%P\n")
cxx_obj_files = $(subst .cpp,.o,$(cpp_files))

objs = $(addprefix build/lib/,$(cxx_obj_files))
DEPS = $(objs:.o=.d)

target = build/main
target_dep = $(addsuffix .d,$(target))

.PRECIOUS: build/lib/%.o

all: $(target)

build/%: src/%.cpp $(gnn_lib) $(objs)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -MMD -o $@ $(filter %.cpp %.o, $^) -L$(lib_dir) -lgnn $(LDFLAGS)

DEPS += $(target_dep)

build/lib/%.o: src/lib/%.cpp
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -MMD -c -o $@ $(filter %.cpp, $^)

clean:
	rm -rf build

-include $(DEPS)
