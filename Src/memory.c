#include "memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void memory_init(memory_t *mem, int max_msgs) {
    if (max_msgs <= 0) max_msgs = MEMORY_DEFAULT_MAX_MESSAGES;
    mem->max_messages = max_msgs;
    // 预分配较小的容量，按需增长
    mem->capacity = (max_msgs < MEMORY_DEFAULT_MAX_MESSAGES) ? max_msgs : MEMORY_DEFAULT_MAX_MESSAGES;
    mem->messages = (llama_chat_message*)malloc(mem->capacity * sizeof(llama_chat_message));
    mem->count = 0;
}

void memory_free(memory_t *mem) {
    if (mem->messages) {
        for (int i = 0; i < mem->count; i++) {
            free((void*)mem->messages[i].content);
            // role 通常为字面量，但为安全也释放
            free((void*)mem->messages[i].role);
        }
        free(mem->messages);
    }
    mem->messages = NULL;
    mem->count = 0;
    mem->capacity = 0;
}

void memory_clear(memory_t *mem) {
    for (int i = 0; i < mem->count; i++) {
        free((void*)mem->messages[i].content);
        free((void*)mem->messages[i].role);
    }
    mem->count = 0;
}

// 动态扩容
static bool memory_ensure_capacity(memory_t *mem, int needed) {
    if (needed <= mem->capacity) return true;
    int new_cap = mem->capacity * 2;
    if (new_cap < needed) new_cap = needed;
    llama_chat_message *new_msgs = (llama_chat_message*)realloc(mem->messages, new_cap * sizeof(llama_chat_message));
    if (!new_msgs) return false;
    mem->messages = new_msgs;
    mem->capacity = new_cap;
    return true;
}

bool memory_add(memory_t *mem, const char *role, const char *content) {
    // 自动裁剪（如果达到最大消息数）
    if (mem->count >= mem->max_messages) {
        // 保留首条（developer）和最近的消息
        memory_trim_last(mem, mem->max_messages / 2 > 2 ? mem->max_messages / 2 : 2);
    }
    if (!memory_ensure_capacity(mem, mem->count + 1)) return false;

    // 复制 role 和 content
    char *role_copy = strdup(role);
    char *content_copy = strdup(content);
    if (!role_copy || !content_copy) {
        free(role_copy);
        free(content_copy);
        return false;
    }

    mem->messages[mem->count].role = role_copy;
    mem->messages[mem->count].content = content_copy;
    mem->count++;
    return true;
}

int memory_get_count(const memory_t *mem) {
    return mem->count;
}

const llama_chat_message* memory_get_messages(const memory_t *mem) {
    return mem->messages;
}

void memory_trim_last(memory_t *mem, int keep) {
    if (keep >= mem->count) return;
    int first_keep = 0; // 始终保留第一条（developer）
    if (mem->count > 0 && strcmp(mem->messages[0].role, "developer") == 0) {
        first_keep = 1;
    }
    int remove_count = mem->count - keep;
    if (remove_count <= 0) return;

    // 保留前 first_keep 条和最后 keep - first_keep 条
    int keep_end = keep - first_keep;   // 尾部保留数
    if (keep_end < 0) keep_end = 0;

    // 计算要删除的消息索引范围：[first_keep, mem->count - keep_end)
    int del_start = first_keep;
    int del_end = mem->count - keep_end;
    if (del_end <= del_start) return;

    // 释放要删除的消息
    for (int i = del_start; i < del_end; i++) {
        free((void*)mem->messages[i].content);
        free((void*)mem->messages[i].role);
    }
    // 移动尾部保留部分到前面
    if (del_end < mem->count) {
        memmove(&mem->messages[del_start], &mem->messages[del_end],
                (mem->count - del_end) * sizeof(llama_chat_message));
    }
    mem->count = del_start + (mem->count - del_end);
}

// ---- 简单的 JSONL 保存/加载 ----

// 保存为 JSONL 文件
bool memory_save(const memory_t *mem, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) return false;
    for (int i = 0; i < mem->count; i++) {
        fprintf(fp, "{\"role\":\"%s\",\"content\":\"%s\"}\n",
                mem->messages[i].role, mem->messages[i].content);
    }
    fclose(fp);
    return true;
}

// 从 JSONL 文件加载（简单解析，假设没有转义双引号）
bool memory_load(memory_t *mem, const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return false;
    char line[16384];
    while (fgets(line, sizeof(line), fp)) {
        // 简单解析 {"role":"...","content":"..."}
        char role[256] = {0}, content[8192] = {0};
        char *p = line;
        // 跳过空白
        while (*p == ' ' || *p == '\t') p++;
        if (*p != '{') continue;
        p++;
        // 查找 "role":
        char *role_key = strstr(p, "\"role\":");
        if (!role_key) continue;
        role_key += 7;  // 跳过 "role":
        while (*role_key == ' ' || *role_key == '\"') role_key++;
        // 读取直到遇到双引号
        int ri = 0;
        while (*role_key && *role_key != '"' && ri < (int)sizeof(role)-1) {
            role[ri++] = *role_key++;
        }
        role[ri] = '\0';

        // 查找 "content":
        char *content_key = strstr(p, "\"content\":");
        if (!content_key) continue;
        content_key += 11; // 跳过 "content":
        while (*content_key == ' ' || *content_key == '\"') content_key++;
        int ci = 0;
        while (*content_key && *content_key != '"' && ci < (int)sizeof(content)-1) {
            content[ci++] = *content_key++;
        }
        content[ci] = '\0';

        memory_add(mem, role, content);
    }
    fclose(fp);
    return true;
}