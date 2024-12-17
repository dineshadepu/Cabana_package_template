# Define the build directory containing binaries
BUILD_DIR := /home/dineshadepu/life/softwares/Cabana_package_template/build

# Find all binaries in BUILD_DIR with "test" in the name (Ubuntu compatible)
TEST_BINARIES := $(shell find $(BUILD_DIR) -type f -executable -name "*test*")

# Target to list all test binaries
.PHONY: list_tests
list_tests:
	@echo "Test binaries found:"
	@for test in $(TEST_BINARIES); do echo $$test; done

# Target to run all test binaries
.PHONY: run_tests
run_tests:
	@echo "Running all test binaries..."
	@for test in $(TEST_BINARIES); do \
	    echo "Running $$test"; \
	    $$test; \
	done
