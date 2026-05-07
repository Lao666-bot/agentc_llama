// tools.h
#ifndef TOOLS_H
#define TOOLS_H
#ifdef __cplusplus
extern "C" {
#endif 
#include <stdbool.h>

#define MAX_TOOLS       32
#define MAX_PARAMS      16
#define MAX_ARG_LEN     256

// 工具执行状态码
typedef enum {
    TOOL_OK,
    TOOL_UNKNOWN_FUNC,
    TOOL_MISSING_PARAM,
    TOOL_EXEC_ERROR
} tool_status_t;

// 单个参数
typedef struct {
    char key[MAX_ARG_LEN];
    char value[MAX_ARG_LEN];
} tool_param_t;

// 一次完整的工具调用
typedef struct {
    char function_name[MAX_ARG_LEN];
    tool_param_t params[MAX_PARAMS];
    int n_params;
} tool_call_t;

// 工具注册描述符（内部使用）
typedef struct {
    char name[MAX_ARG_LEN];
    char description[MAX_ARG_LEN];
    tool_param_t signature[MAX_PARAMS];  // 参数名和默认值
    int n_signature;
    void (*executor)(const tool_call_t *call, char *result, int result_size);
} tool_def_t;

// 初始化内置工具
void tools_init(void);

// 生成 Unsloth 格式的工具声明字符串（供 system prompt 使用）
void tools_build_declaration(char *buf, int buf_size);

// 解析模型输出的工具调用字符串，填充 call 结构体
bool tools_parse_call(const char *text, tool_call_t *call);

// 执行工具调用，返回状态码，并通过 result 返回结果字符串
tool_status_t tools_execute(const tool_call_t *call, char *result, int result_size);

// 清除字符串中的 <escape> 和 </escape> 标记
void tools_strip_escape(char *s);
#ifdef __cplusplus
}
#endif // __cplusplus
#endif // TOOLS_H