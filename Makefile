# Makefile for Edge Agent Project
# 目录结构：
#   project/
#     Makefile
#     include/   (agent.h, tools.h, memory.h)
#     Src/       (agent.c, tools.c, memory.c)
#     example/   (main.c)
#     libagent.a (生成)
#     agent_app  (生成)

CC       = gcc
AR       = ar
RM       = rm -f

# ---------- llama.cpp 路径 (请按你的实际位置修改) ----------
LLAMA_DIR    = ../llama.cpp
LLAMA_BUILD  = $(LLAMA_DIR)/build

# ---------- 包含路径 ----------
INCLUDES = -I./Inc \
           -I$(LLAMA_DIR)/include \
           -I$(LLAMA_DIR)/ggml/include \
           -I$(LLAMA_BUILD)/src

# ---------- 编译选项 ----------
CFLAGS   = -O2 -Wall $(INCLUDES) -fopenmp

# ---------- 源文件与目标文件 ----------
AGENT_SRCS = $(wildcard Src/*.c)
AGENT_OBJS = $(patsubst Src/%.c, Src/%.o, $(AGENT_SRCS))
MAIN_SRC   = example/main.c
MAIN_OBJ   = example/main.o

# ---------- 静态库与可执行文件 ----------
LIBAGENT   = libagent.a
TARGET     = agent_app

# ---------- llama.cpp 静态库列表 ----------
LLAMA_LIBS = $(LLAMA_BUILD)/src/libllama.a \
             $(LLAMA_BUILD)/common/libllama-common.a \
             $(LLAMA_BUILD)/common/libllama-common-base.a \
             $(LLAMA_BUILD)/ggml/src/libggml.a \
             $(LLAMA_BUILD)/ggml/src/libggml-base.a \
             $(LLAMA_BUILD)/ggml/src/libggml-cpu.a

# ---------- 最终链接标志 ----------
LDFLAGS  = $(LLAMA_LIBS) \
           -lpthread -lm -ldl -lstdc++ -lstdc++fs \
           -fopenmp

# ---------- 规则 ----------
all: $(LIBAGENT) $(TARGET)

# 编译 Src 下的 .c 文件
Src/%.o: Src/%.c
	$(CC) -c $(CFLAGS) $< -o $@

# 编译 example/main.c
$(MAIN_OBJ): $(MAIN_SRC)
	$(CC) -c $(CFLAGS) $< -o $@

# 生成静态库
$(LIBAGENT): $(AGENT_OBJS)
	$(AR) rcs $@ $^

# 链接可执行文件
$(TARGET): $(MAIN_OBJ) $(LIBAGENT)
	$(CC) -o $@ $(MAIN_OBJ) -L. -lagent $(LDFLAGS)
# 测试目标
test: $(LIBAGENT) example/test_agent.o
	$(CC) -o test_agent example/test_agent.o -L. -lagent $(LDFLAGS)

example/test_agent.o: example/test_agent.c
	$(CC) -c $(CFLAGS) $< -o $@
# 清理
clean:
	$(RM) $(AGENT_OBJS) $(MAIN_OBJ) $(LIBAGENT) $(TARGET)

.PHONY: all clean