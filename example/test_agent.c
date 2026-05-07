#include "agent.h"
#include <stdio.h>
#include <string.h>

int main() {
    // 从配置文件初始化 Agent
    Agent *agent = agent_create_from_config("agent.conf");
    if (!agent) {
        fprintf(stderr, "❌ Failed to create agent\n");
        return 1;
    }
    printf("✅ Agent created successfully\n");

    // 发送一条简单消息（不涉及工具调用，纯聊天）
    const char *reply = agent_chat(agent, "Hello, who are you?");
    printf("Reply: %s\n", reply);

    // 清理
    agent_destroy(agent);
    printf("✅ Agent destroyed\n");
    return 0;
}