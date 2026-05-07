#ifndef MEMORY_H
#define MEMORY_H
#ifdef __cplusplus
extern "C" {
#endif
#include "llama.h"
#include <stdbool.h>

#define MEMORY_DEFAULT_MAX_MESSAGES 100

typedef struct {
    llama_chat_message *messages;
    int count;
    int capacity;
    int max_messages;
} memory_t;

void memory_init(memory_t *mem, int max_msgs);
void memory_free(memory_t *mem);
void memory_clear(memory_t *mem);
bool memory_add(memory_t *mem, const char *role, const char *content);
int memory_get_count(const memory_t *mem);
const llama_chat_message* memory_get_messages(const memory_t *mem);
void memory_trim_last(memory_t *mem, int keep);

bool memory_save(const memory_t *mem, const char *filename);
bool memory_load(memory_t *mem, const char *filename);
#ifdef __cplusplus
}
#endif
#endif // MEMORY_H