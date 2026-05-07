#include "agent.h"
#include "llama.h"
#include "tools.h"
#include "memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PROMPT_LEN  16384
#define MAX_GEN_TOKENS  2048

struct Agent {
    struct llama_model *model;
    struct llama_context *ctx;
    struct llama_sampler *smpl;
    memory_t memory;
    bool verbose;

    llama_token tok_end_function_call;
    llama_token tok_end_of_turn;
    llama_token tok_eos;
};

static llama_token find_special_token(const struct llama_vocab *vocab, const char *text) {
    llama_token tokens[4];
    int n = llama_tokenize(vocab, text, (int32_t)strlen(text), tokens, 4, false, true);
    if (n == 1) {
        char piece[256];
        if (llama_token_to_piece(vocab, tokens[0], piece, sizeof(piece), 0, true) > 0)
            return tokens[0];
    }
    return LLAMA_TOKEN_NULL;
}

static int build_prompt(struct llama_model *model, const struct llama_vocab *vocab,
                        const struct llama_chat_message *msgs, int n_msgs,
                        bool add_assistant, llama_token *tokens, int max_tokens) {
    const char *tmpl = llama_model_chat_template(model, NULL);
    if (!tmpl) return -1;
    char buf[MAX_PROMPT_LEN];
    int res = llama_chat_apply_template(tmpl, msgs, n_msgs, add_assistant, buf, sizeof(buf));
    if (res < 0 || (size_t)res > sizeof(buf)) return -1;
    return llama_tokenize(vocab, buf, res, tokens, max_tokens, true, true);
}

static bool prefill(struct llama_context *ctx, llama_token *tokens, int n, int *logits_idx) {
    if (n <= 0) { if (logits_idx) *logits_idx = 0; return true; }
    int32_t nb = (int32_t)llama_n_batch(ctx), nc = (int32_t)llama_n_ctx(ctx);
    llama_memory_t mem = llama_get_memory(ctx);
    int32_t used = llama_memory_seq_pos_max(mem, 0) + 1;
    if (used + n > nc) return false;

    int final_idx = 0;
    for (int i = 0; i < n; i += nb) {
        int bs = (i + nb > n) ? (n - i) : nb;
        llama_batch batch = llama_batch_init(bs, 0, 1);
        for (int j = 0; j < bs; j++) {
            batch.token[j] = tokens[i+j];
            batch.pos[j] = used + i + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = (i+j == n-1);
        }
        batch.n_tokens = bs;
        if (llama_decode(ctx, batch) != 0) { llama_batch_free(batch); return false; }
        if (i+bs >= n) final_idx = bs-1;
        llama_batch_free(batch);
    }
    if (logits_idx) *logits_idx = final_idx;
    return true;
}

static int generate_until_stop(struct Agent *agent, llama_token *out, int max_out,
                               llama_token *stop_tok, int start_idx) {
    struct llama_context *ctx = agent->ctx;
    struct llama_sampler *smpl = agent->smpl;
    const struct llama_vocab *vocab = llama_model_get_vocab(llama_get_model(ctx));
    int n_gen = 0;
    *stop_tok = LLAMA_TOKEN_NULL;
    int idx = (start_idx >= 0) ? start_idx : 0;

    while (n_gen < max_out) {
        int32_t nc = (int32_t)llama_n_ctx(ctx);
        llama_memory_t mem = llama_get_memory(ctx);
        int32_t used = llama_memory_seq_pos_max(mem, 0) + 1;
        if (used + 1 > nc) break;

        llama_token tok = llama_sampler_sample(smpl, ctx, idx);
        llama_sampler_accept(smpl, tok);
        if (tok == agent->tok_end_function_call ||
            tok == agent->tok_end_of_turn ||
            tok == agent->tok_eos) {
            out[n_gen++] = tok;
            *stop_tok = tok;
            break;
        }
        out[n_gen++] = tok;

        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.token[0] = tok;
        batch.pos[0] = used;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        batch.n_tokens = 1;
        if (llama_decode(ctx, batch) != 0) { llama_batch_free(batch); break; }
        llama_batch_free(batch);
        idx = 0;
    }
    return n_gen;
}

static char* tokens_to_str(const struct llama_vocab *vocab, const llama_token *toks, int n) {
    static char buf[8192];
    int off = 0;
    for (int i = 0; i < n && off < (int)sizeof(buf)-1; i++) {
        char p[256];
        int l = llama_token_to_piece(vocab, toks[i], p, sizeof(p), 0, false);
        if (l < 0) continue;
        for (int j = 0; j < l && off < (int)sizeof(buf)-1; j++) buf[off++] = p[j];
    }
    buf[off] = '\0';
    return buf;
}

Agent* agent_create(const char *model_path, int n_ctx, int n_threads,
                    float temperature, float top_p, bool verbose) {
    if (n_ctx <= 0) n_ctx = 4096;
    if (n_threads <= 0) n_threads = 4;

    Agent *agent = (Agent*)calloc(1, sizeof(Agent));
    if (!agent) return NULL;
    agent->verbose = verbose;

    tools_init();
    ggml_backend_load_all();

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    struct llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { free(agent); return NULL; }
    agent->model = model;

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    agent->tok_end_function_call = find_special_token(vocab, "<end_function_call>");
    agent->tok_end_of_turn       = find_special_token(vocab, "<end_of_turn>");
    agent->tok_eos               = llama_vocab_eos(vocab);

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)n_ctx;
    cp.n_batch = 256;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); free(agent); return NULL; }
    agent->ctx = ctx;

    struct llama_sampler *smpl;
    if (temperature < 0.001f) {
        smpl = llama_sampler_init_greedy();
    } else {
        smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, 1.1f, 0.0f, 0.0f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }
    agent->smpl = smpl;

    memory_init(&agent->memory, 98);

    char decl_buf[2048];
    tools_build_declaration(decl_buf, sizeof(decl_buf));
    char dev_msg[MAX_PROMPT_LEN];
    snprintf(dev_msg, sizeof(dev_msg),
        "You are a model that can do function calling with the following functions\n"
        "%s\n\n"
        "When you need to call a function, output exactly:\n"
        "<start_function_call>call:function_name{arg1:value1, arg2:value2}<end_function_call>\n"
        "After receiving a function response, respond directly to the user in plain text.",
        decl_buf);
    memory_add(&agent->memory, "developer", dev_msg);

    return agent;
}

void agent_destroy(Agent *agent) {
    if (!agent) return;
    memory_free(&agent->memory);
    if (agent->smpl) llama_sampler_free(agent->smpl);
    if (agent->ctx) llama_free(agent->ctx);
    if (agent->model) llama_model_free(agent->model);
    free(agent);
}

const char* agent_chat(Agent *agent, const char *user_input) {
    if (!agent || !user_input) return "";
    memory_t *mem = &agent->memory;
    memory_add(mem, "user", user_input);

    const struct llama_vocab *vocab = llama_model_get_vocab(agent->model);
    int max_iter = 5;
    char last_call[512] = "";
    int repeat = 0;

    const llama_chat_message *msgs = memory_get_messages(mem);
    int n_msgs = memory_get_count(mem);
    llama_chat_message work[32];
    int wcnt = (n_msgs < 32) ? n_msgs : 32;
    memcpy(work, msgs, wcnt * sizeof(llama_chat_message));

    while (max_iter--) {
        if (wcnt + 2 >= 32) break;
        llama_token prompt_tokens[MAX_PROMPT_LEN];
        int np = build_prompt(agent->model, vocab, work, wcnt, true, prompt_tokens, MAX_PROMPT_LEN);
        if (np < 0) return "";

        llama_memory_clear(llama_get_memory(agent->ctx), true);
        llama_sampler_reset(agent->smpl);
        int logits_idx;
        if (!prefill(agent->ctx, prompt_tokens, np, &logits_idx)) return "";

        llama_token out_toks[MAX_GEN_TOKENS];
        llama_token stop_tok;
        int ng = generate_until_stop(agent, out_toks, MAX_GEN_TOKENS, &stop_tok, logits_idx);
        if (ng == 0) return "";
        char *text = tokens_to_str(vocab, out_toks, ng);
        if (agent->verbose) fprintf(stderr, "[agent] gen: %s\n", text);

        if (stop_tok == agent->tok_end_function_call) {
            tool_call_t call;
            if (!tools_parse_call(text, &call)) return text;

            char cur[512];
            snprintf(cur, sizeof(cur), "%s(", call.function_name);
            for (int i = 0; i < call.n_params; i++)
                snprintf(cur + strlen(cur), sizeof(cur) - strlen(cur), "%s:%s,", call.params[i].key, call.params[i].value);
            if (strcmp(cur, last_call) == 0) {
                if (++repeat >= 2) {
                    if (agent->verbose) fprintf(stderr, "[agent] stopped: same call repeated\n");
                    break;
                }
            } else {
                strncpy(last_call, cur, sizeof(last_call)-1);
                repeat = 1;
            }

            char result[8192];
            tool_status_t status = tools_execute(&call, result, sizeof(result));
            if (status == TOOL_OK && agent->verbose) {
                fprintf(stderr, "  [call %s succeeded]\n", call.function_name);
            } else if (status != TOOL_OK && agent->verbose) {
                fprintf(stderr, "  [call error: %s]\n", result);
            }

            work[wcnt++] = (llama_chat_message){"assistant", text};
            work[wcnt++] = (llama_chat_message){"user", result};
            continue;
        } else {
            char *eot = strstr(text, "<end_of_turn>");
            if (eot) *eot = '\0';
            tools_strip_escape(text);
            memory_add(mem, "assistant", text);
            return text;
        }
    }
    return "I'm sorry, I couldn't process that.";
}

void agent_reset_history(Agent *agent) {
    if (!agent) return;
    memory_t *mem = &agent->memory;
    if (mem->count > 1) {
        memory_trim_last(mem, 1); // keep developer message
    }
}

bool agent_save_history(const Agent *agent, const char *filename) {
    if (!agent || !filename) return false;
    return memory_save(&agent->memory, filename);
}

bool agent_load_history(Agent *agent, const char *filename) {
    if (!agent || !filename) return false;
    return memory_load(&agent->memory, filename);
}
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* 去除字符串首尾空白 */
static char* trim(char *s) {
    while (isspace(*s)) s++;
    if (*s == '\0') return s;
    char *end = s + strlen(s) - 1;
    while (end > s && isspace(*end)) end--;
    *(end + 1) = '\0';
    return s;
}

/* 简单的键值对解析，跳过注释 */
static bool parse_config_line(char *line, char *key, size_t key_size, char *value, size_t value_size) {
    char *p = line;
    // 跳过空白
    while (*p && isspace(*p)) p++;
    // 跳过空行和注释
    if (*p == '\0' || *p == '#') return false;

    char *eq = strchr(p, '=');
    if (!eq) return false;

    // 提取 key
    size_t klen = eq - p;
    if (klen >= key_size) klen = key_size - 1;
    strncpy(key, p, klen);
    key[klen] = '\0';
    trim(key);

    // 提取 value
    p = eq + 1;
    while (*p && isspace(*p)) p++;
    char *val_end = p + strlen(p) - 1;
    while (val_end > p && isspace(*val_end)) val_end--;
    size_t vlen = (val_end - p) + 1;
    if (vlen >= value_size) vlen = value_size - 1;
    strncpy(value, p, vlen);
    value[vlen] = '\0';
    trim(value);
    return true;
}

Agent* agent_create_from_config(const char *config_file) {
    FILE *fp = fopen(config_file, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open config file: %s\n", config_file);
        return NULL;
    }

    char model_path[512] = "";
    int n_ctx = 4096;
    int n_threads = 4;
    float temperature = 0.0f;
    float top_p = 0.9f;
    bool verbose = false;

    char line[512];
    char key[128], value[256];
    while (fgets(line, sizeof(line), fp)) {
        if (!parse_config_line(line, key, sizeof(key), value, sizeof(value)))
            continue;
        if (strcmp(key, "model_path") == 0) {
            strncpy(model_path, value, sizeof(model_path)-1);
        } else if (strcmp(key, "n_ctx") == 0) {
            n_ctx = atoi(value);
        } else if (strcmp(key, "n_threads") == 0) {
            n_threads = atoi(value);
        } else if (strcmp(key, "temperature") == 0) {
            temperature = (float)atof(value);
        } else if (strcmp(key, "top_p") == 0) {
            top_p = (float)atof(value);
        } else if (strcmp(key, "verbose") == 0) {
            verbose = (strcmp(value, "true") == 0 || strcmp(value, "1") == 0);
        }
    }
    fclose(fp);

    if (model_path[0] == '\0') {
        fprintf(stderr, "model_path missing in config\n");
        return NULL;
    }

    return agent_create(model_path, n_ctx, n_threads, temperature, top_p, verbose);
}