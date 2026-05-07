// tools.c
#include "tools.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static tool_def_t g_tools[MAX_TOOLS];
static int g_n_tools = 0;

void tools_strip_escape(char *s) {
    char *r = s, *w = s;
    while (*r) {
        if (strncmp(r, "<escape>", 8) == 0)
            r += 8;
        else if (strncmp(r, "</escape>", 9) == 0)
            r += 9;
        else
            *w++ = *r++;
    }
    *w = '\0';
}

// ---------- 内置工具：写文件 ----------
static void do_write_file(const tool_call_t *call, char *result, int result_size) {
    char fn[MAX_ARG_LEN] = {0}, cnt[MAX_ARG_LEN] = {0};
    for (int i = 0; i < call->n_params; i++) {
        if (strcmp(call->params[i].key, "filename") == 0) {
            strncpy(fn, call->params[i].value, sizeof(fn)-1);
            tools_strip_escape(fn);
        }
        if (strcmp(call->params[i].key, "content") == 0) {
            strncpy(cnt, call->params[i].value, sizeof(cnt)-1);
            tools_strip_escape(cnt);
        }
    }
    if (!fn[0]) {
        snprintf(result, result_size, "Error: empty filename");
        return;
    }
    FILE *fp = fopen(fn, "w");
    if (!fp) {
        snprintf(result, result_size, "Error: cannot open '%s'", fn);
        return;
    }
    if (fputs(cnt, fp) < 0) {
        snprintf(result, result_size, "Error: write failed");
        fclose(fp);
        return;
    }
    fclose(fp);
    snprintf(result, result_size, "File '%s' written successfully", fn);
}
//---------- 工具系统接口实现 ----------
void tools_init(void) {
    g_n_tools = 0;
    g_tools[g_n_tools++] = (tool_def_t){
        .name = "write_file",
        .description = "Write content to a file",
        .signature = {{"filename", "example.txt"}, {"content", "Hello, world!"}},
        .n_signature = 2,
        .executor = do_write_file
    };
}

void tools_build_declaration(char *buf, int buf_size) {
    int off = 0;
    off += snprintf(buf + off, buf_size - off, "<start_function_declaration>");
    for (int i = 0; i < g_n_tools; i++) {
        off += snprintf(buf + off, buf_size - off,
            "declaration:%s{description:%s,parameters:{type:OBJECT,properties:{",
            g_tools[i].name, g_tools[i].description);
        for (int j = 0; j < g_tools[i].n_signature; j++) {
            off += snprintf(buf + off, buf_size - off,
                "%s%s:{type:STRING}", (j > 0 ? "," : ""), g_tools[i].signature[j].key);
        }
        off += snprintf(buf + off, buf_size - off, "}}}");
    }
    off += snprintf(buf + off, buf_size - off, "<end_function_declaration>");
}

bool tools_parse_call(const char *text, tool_call_t *call) {
    const char *s = strstr(text, "<start_function_call>");
    if (!s) return false;
    s += strlen("<start_function_call>");
    const char *e = strstr(s, "<end_function_call>");
    if (!e) return false;
    while (s < e && (*s == ' ' || *s == '\n')) s++;
    if (strncmp(s, "call:", 5) != 0) return false;
    s += 5;
    const char *br = strchr(s, '{');
    if (!br || br >= e) return false;
    int nl = br - s;
    if (nl <= 0 || nl >= (int)sizeof(call->function_name)) return false;
    memcpy(call->function_name, s, nl);
    call->function_name[nl] = '\0';

    const char *ps = br, *pe = e;
    int dep = 0;
    while (ps < pe) {
        if (*ps == '{') dep++;
        else if (*ps == '}') {
            dep--;
            if (!dep) { pe = ps; break; }
        }
        ps++;
    }
    if (pe <= br) {
        call->n_params = 0;
        return true;
    }

    const char *p = br + 1;
    call->n_params = 0;
    while (p < pe && call->n_params < MAX_PARAMS) {
        while (p < pe && (*p == ' ' || *p == ',' || *p == '\n')) p++;
        if (p >= pe) break;
        const char *col = p;
        while (col < pe && *col != ':') col++;
        if (col >= pe) break;
        int kl = col - p;
        if (kl <= 0 || kl >= MAX_ARG_LEN) break;
        memcpy(call->params[call->n_params].key, p, kl);
        call->params[call->n_params].key[kl] = '\0';

        const char *vs = col + 1;
        while (vs < pe && *vs == ' ') vs++;
        if (strncmp(vs, "<escape>", 8) == 0) vs += 8;
        char q = 0;
        if (*vs == '"' || *vs == '\'') { q = *vs; vs++; }
        const char *ve = vs;
        if (q) {
            while (ve < pe && *ve != q) ve++;
            int vl = ve - vs;
            if (vl >= MAX_ARG_LEN) vl = MAX_ARG_LEN - 1;
            if (vl > 0) memcpy(call->params[call->n_params].value, vs, vl);
            else call->params[call->n_params].value[0] = '\0';
            call->params[call->n_params].value[vl] = '\0';
            p = ve + 1;
        } else {
            while (ve < pe && *ve != ',' && *ve != '}' && *ve != '"') ve++;
            int vl = ve - vs;
            if (vl >= MAX_ARG_LEN) vl = MAX_ARG_LEN - 1;
            memcpy(call->params[call->n_params].value, vs, vl);
            call->params[call->n_params].value[vl] = '\0';
            p = ve;
        }
        tools_strip_escape(call->params[call->n_params].value);
        call->n_params++;
    }
    return true;
}

tool_status_t tools_execute(const tool_call_t *call, char *result, int result_size) {
    for (int i = 0; i < g_n_tools; i++) {
        if (strcmp(call->function_name, g_tools[i].name) == 0) {
            // 检查必要参数是否存在
            for (int j = 0; j < g_tools[i].n_signature; j++) {
                bool found = false;
                for (int k = 0; k < call->n_params; k++) {
                    if (strcmp(call->params[k].key, g_tools[i].signature[j].key) == 0) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    snprintf(result, result_size, "Missing parameter: %s", g_tools[i].signature[j].key);
                    return TOOL_MISSING_PARAM;
                }
            }
            g_tools[i].executor(call, result, result_size);
            return TOOL_OK;
        }
    }
    snprintf(result, result_size, "Unknown function '%s'", call->function_name);
    return TOOL_UNKNOWN_FUNC;
}
