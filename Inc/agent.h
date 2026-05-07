#ifndef AGENT_H
#define AGENT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

typedef struct Agent Agent;
/**
 * 从配置文件创建 Agent
 * @param config_file   配置文件的路径
 * @return 成功返回 Agent 指针，失败返回 NULL
 */
Agent* agent_create_from_config(const char *config_file);
Agent* agent_create(const char *model_path,
                    int n_ctx,
                    int n_threads,
                    float temperature,
                    float top_p,
                    bool verbose);

void agent_destroy(Agent *agent);

const char* agent_chat(Agent *agent, const char *user_input);

void agent_reset_history(Agent *agent);

bool agent_save_history(const Agent *agent, const char *filename);

bool agent_load_history(Agent *agent, const char *filename);

#ifdef __cplusplus
}
#endif

#endif // AGENT_H