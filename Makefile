CXX ?= g++
AR ?= ar
BUILD_DIR ?= build/native
EXE_SUFFIX ?=
TARGET_NAME ?= forevervalidator
TARGET := $(BUILD_DIR)/$(TARGET_NAME)$(EXE_SUFFIX)
OBJ_DIR := $(BUILD_DIR)/obj

SOURCE_DATE_EPOCH ?= 0
export SOURCE_DATE_EPOCH
export LC_ALL := C
export TZ := UTC

COMMON_WARNINGS := -Wall -Wextra -Werror -Wno-deprecated-declarations -Wno-error=maybe-uninitialized
COMMON_CXXFLAGS ?= -O2 -ffp-contract=off -MMD -MP -std=c++17 \
	$(COMMON_WARNINGS) -frandom-seed=forevervalidator \
	-ffile-prefix-map=$(CURDIR)=.
CPPFLAGS ?=
PRIVATE_CXXFLAGS := $(COMMON_CXXFLAGS) $(CPPFLAGS) -Iinclude -Isrc
PUBLIC_CXXFLAGS := $(COMMON_CXXFLAGS) $(CPPFLAGS) -Iinclude
ARFLAGS := rcsD
LDLIBS ?= -lcrypto -lz -lm

CORE_SOURCES := $(shell find \
	src/engine \
	src/format \
	src/simulation \
	src/validation/api \
	src/validation/planning \
	src/validation/evaluation \
	-type f -name '*.cpp' -print | sort)
JSON_SOURCES := $(shell find \
	src/validation/serialization -type f -name '*.cpp' -print | sort)
NATIVE_SOURCES := $(shell find \
	src/platform/native -type f -name '*.cpp' -print | sort)
CLI_SOURCES := $(shell find \
	src/app/cli -type f -name '*.cpp' -print | sort)

objects_for = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(1))
CORE_OBJECTS := $(call objects_for,$(CORE_SOURCES))
JSON_OBJECTS := $(call objects_for,$(JSON_SOURCES))
NATIVE_OBJECTS := $(call objects_for,$(NATIVE_SOURCES))
CLI_OBJECTS := $(call objects_for,$(CLI_SOURCES))

CORE_LIB := $(BUILD_DIR)/libforevervalidator_core.a
JSON_LIB := $(BUILD_DIR)/libforevervalidator_json.a
NATIVE_LIB := $(BUILD_DIR)/libforevervalidator_native.a
LIBRARIES := $(CORE_LIB) $(JSON_LIB) $(NATIVE_LIB)

WINDOWS_TRIPLE ?= x86_64-w64-mingw32
WINDOWS_CXX ?= $(WINDOWS_TRIPLE)-g++
WINDOWS_AR ?= $(WINDOWS_TRIPLE)-ar
WINDOWS_OBJDUMP ?= $(WINDOWS_TRIPLE)-objdump
WINDOWS_DEPS_DIR ?= build/windows-deps
WINDOWS_PREFIX := $(abspath $(WINDOWS_DEPS_DIR)/prefix)
WINDOWS_DEPS_STAMP := $(WINDOWS_DEPS_DIR)/.installed
WINDOWS_CRYPTO_LIB := $(WINDOWS_PREFIX)/lib64/libcrypto.a
WINDOWS_ZLIB_LIB := $(WINDOWS_PREFIX)/lib/libz.a
WINDOWS_BUILD_DIR ?= build/windows
WINDOWS_OPENSSL_VERSION := 3.5.7
WINDOWS_OPENSSL_SHA256 := a8c0d28a529ca480f9f36cf5792e2cd21984552a3c8e4aa11a24aa31aeac98e8
WINDOWS_ZLIB_VERSION := 1.3.2
WINDOWS_ZLIB_SHA256 := bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16
WINDOWS_OPENSSL_ARCHIVE := $(WINDOWS_DEPS_DIR)/openssl-$(WINDOWS_OPENSSL_VERSION).tar.gz
WINDOWS_ZLIB_ARCHIVE := $(WINDOWS_DEPS_DIR)/zlib-$(WINDOWS_ZLIB_VERSION).tar.gz
WINDOWS_CPPFLAGS := -I$(WINDOWS_PREFIX)/include
WINDOWS_LDLIBS := -L$(WINDOWS_PREFIX)/lib64 -L$(WINDOWS_PREFIX)/lib \
	-lcrypto -lz -lws2_32 -lgdi32 -lcrypt32 -static \
	-Wl,--no-insert-timestamp
.PHONY: all libraries cli-link clean windows windows-deps \
	verify-windows-imports reproducible-linux reproducible-windows \
	reproducible

all: libraries cli-link

libraries: $(LIBRARIES)

cli-link: $(TARGET)

$(TARGET): $(CLI_OBJECTS) $(NATIVE_LIB) $(JSON_LIB) $(CORE_LIB) | $(BUILD_DIR)
	$(CXX) -o $@ $(CLI_OBJECTS) $(NATIVE_LIB) $(JSON_LIB) $(CORE_LIB) $(LDLIBS)

$(CORE_LIB): $(CORE_OBJECTS) | $(BUILD_DIR)
	rm -f $@
	$(AR) $(ARFLAGS) $@ $^

$(JSON_LIB): $(JSON_OBJECTS) | $(BUILD_DIR)
	rm -f $@
	$(AR) $(ARFLAGS) $@ $^

$(NATIVE_LIB): $(NATIVE_OBJECTS) | $(BUILD_DIR)
	rm -f $@
	$(AR) $(ARFLAGS) $@ $^

$(OBJ_DIR)/src/app/cli/%.o: src/app/cli/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(PUBLIC_CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(PRIVATE_CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@

windows-deps:
	@set -eu; \
	if test ! -f "$(WINDOWS_CRYPTO_LIB)" || \
	   test ! -f "$(WINDOWS_ZLIB_LIB)"; then \
		rm -f "$(WINDOWS_DEPS_STAMP)"; \
		$(MAKE) "$(WINDOWS_DEPS_STAMP)"; \
	fi; \
	test -f "$(WINDOWS_CRYPTO_LIB)"; \
	test -f "$(WINDOWS_ZLIB_LIB)"

$(WINDOWS_DEPS_STAMP):
	@set -eu; \
	command -v curl >/dev/null; \
	command -v perl >/dev/null; \
	command -v $(WINDOWS_CXX) >/dev/null; \
	mkdir -p "$(WINDOWS_DEPS_DIR)"; \
	if test ! -f "$(WINDOWS_OPENSSL_ARCHIVE)"; then \
		curl --fail --location --retry 3 \
			-o "$(WINDOWS_OPENSSL_ARCHIVE)" \
			"https://www.openssl.org/source/openssl-$(WINDOWS_OPENSSL_VERSION).tar.gz"; \
	fi; \
	echo "$(WINDOWS_OPENSSL_SHA256)  $(WINDOWS_OPENSSL_ARCHIVE)" | sha256sum -c -; \
	if test ! -f "$(WINDOWS_ZLIB_ARCHIVE)"; then \
		curl --fail --location --retry 3 \
			-o "$(WINDOWS_ZLIB_ARCHIVE)" \
			"https://zlib.net/zlib-$(WINDOWS_ZLIB_VERSION).tar.gz"; \
	fi; \
	echo "$(WINDOWS_ZLIB_SHA256)  $(WINDOWS_ZLIB_ARCHIVE)" | sha256sum -c -; \
	rm -rf "$(WINDOWS_DEPS_DIR)/openssl-$(WINDOWS_OPENSSL_VERSION)" \
		"$(WINDOWS_DEPS_DIR)/zlib-$(WINDOWS_ZLIB_VERSION)" \
		"$(WINDOWS_PREFIX)"; \
	tar -xf "$(WINDOWS_OPENSSL_ARCHIVE)" -C "$(WINDOWS_DEPS_DIR)"; \
	tar -xf "$(WINDOWS_ZLIB_ARCHIVE)" -C "$(WINDOWS_DEPS_DIR)"; \
	cd "$(WINDOWS_DEPS_DIR)/openssl-$(WINDOWS_OPENSSL_VERSION)"; \
	CROSS_COMPILE=$(WINDOWS_TRIPLE)- ./Configure mingw64 \
		no-shared no-tests no-apps no-docs \
		--prefix=/ --openssldir=/ssl; \
	$(MAKE) -j$${JOBS:-2} build_sw; \
	$(MAKE) DESTDIR="$(WINDOWS_PREFIX)" install_sw; \
	cd "$(CURDIR)/$(WINDOWS_DEPS_DIR)/zlib-$(WINDOWS_ZLIB_VERSION)"; \
	$(MAKE) -f win32/Makefile.gcc PREFIX=$(WINDOWS_TRIPLE)- -j$${JOBS:-2}; \
	$(MAKE) -f win32/Makefile.gcc PREFIX=$(WINDOWS_TRIPLE)- \
		BINARY_PATH="$(WINDOWS_PREFIX)/bin" \
		INCLUDE_PATH="$(WINDOWS_PREFIX)/include" \
		LIBRARY_PATH="$(WINDOWS_PREFIX)/lib" install; \
	touch "$(CURDIR)/$@"

windows: windows-deps
	$(MAKE) BUILD_DIR="$(WINDOWS_BUILD_DIR)" EXE_SUFFIX=.exe \
		CXX="$(WINDOWS_CXX)" AR="$(WINDOWS_AR)" \
		CPPFLAGS="$(WINDOWS_CPPFLAGS)" LDLIBS="$(WINDOWS_LDLIBS)" all
	$(MAKE) WINDOWS_BUILD_DIR="$(WINDOWS_BUILD_DIR)" verify-windows-imports

verify-windows-imports:
	@set -eu; \
	target="$(WINDOWS_BUILD_DIR)/$(TARGET_NAME).exe"; \
	test -f "$$target"; \
	imports=`$(WINDOWS_OBJDUMP) -p "$$target" | \
		sed -n 's/^[[:space:]]*DLL Name:[[:space:]]*//p'`; \
	if printf '%s\n' "$$imports" | \
		grep -Eiq '^(libgcc|libstdc\+\+|libwinpthread).*\.dll$$'; then \
		printf '%s\n' "$$imports" >&2; \
		echo "non-system MinGW runtime dependency detected" >&2; \
		exit 1; \
	fi

reproducible-linux:
	rm -rf build/repro-linux-a build/repro-linux-b
	$(MAKE) BUILD_DIR=build/repro-linux-a all
	$(MAKE) BUILD_DIR=build/repro-linux-b all
	cmp build/repro-linux-a/$(TARGET_NAME) build/repro-linux-b/$(TARGET_NAME)
	cmp build/repro-linux-a/libforevervalidator_core.a build/repro-linux-b/libforevervalidator_core.a
	cmp build/repro-linux-a/libforevervalidator_json.a build/repro-linux-b/libforevervalidator_json.a
	cmp build/repro-linux-a/libforevervalidator_native.a build/repro-linux-b/libforevervalidator_native.a

reproducible-windows: windows-deps
	rm -rf build/repro-windows-a build/repro-windows-b
	$(MAKE) WINDOWS_BUILD_DIR=build/repro-windows-a windows
	$(MAKE) WINDOWS_BUILD_DIR=build/repro-windows-b windows
	cmp build/repro-windows-a/$(TARGET_NAME).exe build/repro-windows-b/$(TARGET_NAME).exe
	cmp build/repro-windows-a/libforevervalidator_core.a build/repro-windows-b/libforevervalidator_core.a
	cmp build/repro-windows-a/libforevervalidator_json.a build/repro-windows-b/libforevervalidator_json.a
	cmp build/repro-windows-a/libforevervalidator_native.a build/repro-windows-b/libforevervalidator_native.a

reproducible: reproducible-linux reproducible-windows

clean:
	rm -rf build

ALL_OBJECTS := $(CORE_OBJECTS) $(JSON_OBJECTS) $(NATIVE_OBJECTS) $(CLI_OBJECTS)
-include $(ALL_OBJECTS:.o=.d)
