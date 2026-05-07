#include "agent.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    Agent *agent = agent_create(argv[1], 4096, 4, 0.0f, 0.9f, false);
    if (!agent) {
        fprintf(stderr, "Failed to create agent\n");
        return 1;
    }
    printf("Agent ready. Type /exit to quit.\n");
    char line[4096];
    while (1) {
        printf("> "); fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;
        line[strcspn(line, "\n")] = 0;
        if (strcmp(line, "/exit") == 0) break;
        const char *reply = agent_chat(agent, line);
        printf("%s\n\n", reply);
    }
    agent_destroy(agent);
    return 0;
}