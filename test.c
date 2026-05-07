//整合记忆和工具，构建一个简单的 Agent 循环，支持多轮对话和函数调用
#include "llama.h"
#include "tools.h"
#include "memory.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define MAX_PROMPT_LEN  16384
#define MAX_OUTPUT_LEN  8192
#define MAX_GEN_TOKENS  2048

#define DEFAULT_N_CTX   4096
#define DEFAULT_TEMP    0.0f
#define DEFAULT_TOP_P   0.9f
#define DEFAULT_THREADS 4

typedef struct {
    llama_token start_function_declaration;
    llama_token end_function_declaration;
    llama_token start_function_call;
    llama_token end_function_call;
    llama_token start_function_response;
    llama_token end_function_response;
    llama_token end_of_turn;
    llama_token bos;
    llama_token eos;
} special_tokens_t;

static special_tokens_t g_stok;
static bool g_verbose = false;

/* ---------- 特殊 token ---------- */
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

static void init_special_tokens(const struct llama_vocab *vocab) {
    g_stok.start_function_declaration = find_special_token(vocab, "<start_function_declaration>");
    g_stok.end_function_declaration   = find_special_token(vocab, "<end_function_declaration>");
    g_stok.start_function_call       = find_special_token(vocab, "<start_function_call>");
    g_stok.end_function_call         = find_special_token(vocab, "<end_function_call>");
    g_stok.start_function_response   = find_special_token(vocab, "<start_function_response>");
    g_stok.end_function_response     = find_special_token(vocab, "<end_function_response>");
    g_stok.end_of_turn               = find_special_token(vocab, "<end_of_turn>");
    g_stok.bos                       = llama_vocab_bos(vocab);
    g_stok.eos                       = llama_vocab_eos(vocab);
}

/* ---------- chat template ---------- */
static int build_prompt(struct llama_model *model, const struct llama_vocab *vocab,
                        const llama_chat_message *msgs, int n_msgs,
                        bool add_assistant, llama_token *tokens, int max_tokens) {
    const char *tmpl = llama_model_chat_template(model, NULL);
    if (!tmpl) return -1;
    char buf[MAX_PROMPT_LEN];
    int res = llama_chat_apply_template(tmpl, msgs, n_msgs, add_assistant, buf, sizeof(buf));
    if (res < 0 || (size_t)res > sizeof(buf)) return -1;
    return llama_tokenize(vocab, buf, res, tokens, max_tokens, true, true);
}

/* ---------- 分批 prefill ---------- */
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

/* ---------- 生成直到停止 token ---------- */
static int gen_stop(struct llama_context *ctx, struct llama_sampler *smpl,
                    llama_token *out, int max_out, llama_token *stop_tok, int start_idx) {
    const struct llama_vocab *vocab = llama_model_get_vocab(llama_get_model(ctx));
    int n = 0; *stop_tok = LLAMA_TOKEN_NULL;
    int idx = (start_idx >= 0) ? start_idx : 0;

    while (n < max_out) {
        int32_t nc = (int32_t)llama_n_ctx(ctx);
        llama_memory_t mem = llama_get_memory(ctx);
        int32_t used = llama_memory_seq_pos_max(mem, 0) + 1;
        if (used+1 > nc) break;

        llama_token tok = llama_sampler_sample(smpl, ctx, idx);
        llama_sampler_accept(smpl, tok);
        if (tok == g_stok.end_function_call || tok == g_stok.end_of_turn ||
            llama_vocab_is_eog(vocab, tok)) {
            out[n++] = tok; *stop_tok = tok; break;
        }
        out[n++] = tok;

        llama_batch batch = llama_batch_init(1,0,1);
        batch.token[0]=tok; batch.pos[0]=used; batch.n_seq_id[0]=1; batch.seq_id[0][0]=0;
        batch.logits[0]=true; batch.n_tokens=1;
        if (llama_decode(ctx, batch) != 0) { llama_batch_free(batch); break; }
        llama_batch_free(batch);
        idx = 0;
    }
    return n;
}

/* ---------- token 转字符串 ---------- */
static char* tokens_to_string(const struct llama_vocab *vocab, const llama_token *toks, int n) {
    static char buf[MAX_OUTPUT_LEN];
    int off=0;
    for (int i=0; i<n && off<(int)sizeof(buf)-1; i++) {
        char p[256]; int l = llama_token_to_piece(vocab, toks[i], p, sizeof(p), 0, false);
        if (l<0) continue;
        for (int j=0; j<l && off<(int)sizeof(buf)-1; j++) buf[off++]=p[j];
    }
    buf[off]=0; return buf;
}

/* ========== Agent 循环（使用 memory_t） ========== */
static int run_agent_with_history(struct llama_context *ctx, struct llama_model *model,
                                  struct llama_sampler *smpl,
                                  memory_t *mem) {
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int max_iter = 5;
    char last_call[512] = ""; int repeat = 0;

    // 从记忆获取消息
    const llama_chat_message *msgs = memory_get_messages(mem);
    int n_msgs = memory_get_count(mem);
    // 工作副本（最多 32 条，避免过长 prompt）
    llama_chat_message work[32];
    int wcnt = (n_msgs < 32) ? n_msgs : 32;
    memcpy(work, msgs, wcnt * sizeof(llama_chat_message));

    while (max_iter--) {
        if (wcnt + 2 >= 32) break;
        llama_token prompt_tokens[MAX_PROMPT_LEN];
        int np = build_prompt(model, vocab, work, wcnt, true, prompt_tokens, MAX_PROMPT_LEN);
        if (np < 0) return -1;

        llama_memory_clear(llama_get_memory(ctx), true);
        llama_sampler_reset(smpl);
        int logits_idx;
        if (!prefill(ctx, prompt_tokens, np, &logits_idx)) return -1;

        llama_token out_toks[MAX_GEN_TOKENS]; llama_token stop_tok;
        int ng = gen_stop(ctx, smpl, out_toks, MAX_GEN_TOKENS, &stop_tok, logits_idx);
        if (ng == 0) break;
        char *text = tokens_to_string(vocab, out_toks, ng);
        if (g_verbose) fprintf(stderr, "[agent] gen: %s\n", text);

        if (stop_tok == g_stok.end_function_call) {
            tool_call_t call;
            if (!tools_parse_call(text, &call)) { printf("%s\n", text); break; }

            // 检测重复调用
            char cur[512];
            snprintf(cur, sizeof(cur), "%s(", call.function_name);
            for (int i = 0; i < call.n_params; i++)
                snprintf(cur + strlen(cur), sizeof(cur) - strlen(cur), "%s:%s,", call.params[i].key, call.params[i].value);
            if (strcmp(cur, last_call) == 0) {
                if (++repeat >= 2) { printf("  [stopped: same call repeated]\n"); break; }
            } else {
                strncpy(last_call, cur, sizeof(last_call)-1); repeat = 1;
            }

            char result[MAX_OUTPUT_LEN];
            tool_status_t status = tools_execute(&call, result, sizeof(result));
            if (status == TOOL_OK) {
                printf("  [call %s(", call.function_name);
                for (int i = 0; i < call.n_params; i++)
                    printf("%s:\"%s\"%s", call.params[i].key, call.params[i].value, i<call.n_params-1?", ":"");
                printf(") = %s]\n", result);
            } else {
                printf("  [error: %s]\n", result);
            }

            // 只将临时消息加入工作副本，不污染持久记忆
            work[wcnt++] = (llama_chat_message){"assistant", text};
            work[wcnt++] = (llama_chat_message){"user", result};
            continue;
        } else {
            char *eot = strstr(text, "<end_of_turn>");
            if (eot) *eot = 0;
            printf("%s\n", text);

            // 将助手最终回复追加到记忆
            memory_add(mem, "assistant", text);
            break;
        }
    }
    return 0;
}

/* ---------- 主函数 ---------- */
static void log_cb(enum ggml_log_level lv, const char *t, void *ud) {
    (void)ud;
    if (lv >= GGML_LOG_LEVEL_ERROR) fprintf(stderr, "%s", t);
    else if (g_verbose && lv >= GGML_LOG_LEVEL_WARN) fprintf(stderr, "%s", t);
}

int main(int argc, char **argv) {
    char model_path[512] = ""; int n_ctx = DEFAULT_N_CTX, n_threads = DEFAULT_THREADS;
    float temp = DEFAULT_TEMP, top_p = DEFAULT_TOP_P; bool inter = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i+1 < argc) strncpy(model_path, argv[++i], sizeof(model_path)-1);
        else if (!strcmp(argv[i], "-c") && i+1 < argc) n_ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i+1 < argc) temp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-p") && i+1 < argc) top_p = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-i")) inter = true;
        else if (!strcmp(argv[i], "--verbose")) g_verbose = true;
        else if (!strcmp(argv[i], "-h")) { printf("Usage: %s -m model.gguf -i ...\n", argv[0]); return 0; }
    }
    if (!model_path[0]) { fprintf(stderr, "need -m\n"); return 1; }

    tools_init();
    llama_log_set(log_cb, NULL);
    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    struct llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) return 1;
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    init_special_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)n_ctx; cp.n_batch = 256;
    cp.n_threads = n_threads; cp.n_threads_batch = n_threads;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return 1; }

    struct llama_sampler *smpl;
    if (temp < 0.001f) smpl = llama_sampler_init_greedy();
    else {
        smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, 1.1f, 0.0f,0.0f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    if (inter) {
        printf("Edge Agent (modular, memory, real write_file)\nTools: write_file\n\n");

        // 初始化记忆
        memory_t mem;
        memory_init(&mem, 98);  // 留出空间给 system 消息和动态增加

        // 构建并添加 developer 消息
        char decl[2048]; tools_build_declaration(decl, sizeof(decl));
        char dev_msg[MAX_PROMPT_LEN];
        snprintf(dev_msg, sizeof(dev_msg),
            "You are a model that can do function calling with the following functions\n"
            "%s\n\n"
            "When you need to call a function, output exactly:\n"
            "<start_function_call>call:function_name{arg1:value1, arg2:value2}<end_function_call>\n"
            "After receiving a function response, respond directly to the user in plain text.",
            decl);
        memory_add(&mem, "developer", dev_msg);

        char input[MAX_PROMPT_LEN];
        while (1) {
            printf("> "); fflush(stdout);
            if (!fgets(input, sizeof(input), stdin)) break;
            if (input[0] == '\n') continue;
            input[strcspn(input, "\n")] = 0;

            // 用户输入加入记忆
            memory_add(&mem, "user", input);

            if (run_agent_with_history(ctx, model, smpl, &mem) != 0)
                printf("[agent] error\n");
            printf("\n");
        }

        // 可选：退出前保存对话历史
        // memory_save(&mem, "history.jsonl");
        memory_free(&mem);
    } else {
        fprintf(stderr, "Use -i for interactive mode\n");
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}