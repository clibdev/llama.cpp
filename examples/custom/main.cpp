#include <common.h>

int32_t llama_decode_tokens(struct llama_context *ctx, llama_token *tokens, int32_t n_tokens,
                            int32_t n_batch, int32_t n_ctx, bool keep_bos, int32_t &n_past);

int main() {
    const char *weights = "llama-2-7b-chat.Q4_K_M.gguf";

    llama_context_params cparams = llama_context_default_params();
    cparams.seed = 1709058531; // time(nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 100;

    llama_sampling_params sparams;

    llama_backend_init();

    llama_model *model = llama_load_model_from_file(weights, mparams);
    llama_context *ctx = llama_new_context_with_model(model, cparams);
    llama_sampling_context *ctxs = llama_sampling_init(sparams);

    const auto n_ctx = (int32_t) llama_n_ctx(ctx);
    const auto n_batch = (int32_t) cparams.n_batch;
    llama_token tokens[4096];
    bool add_bos = llama_should_add_bos_token(model);
    bool keep_bos = add_bos;
    bool first_run = true;

    int32_t n_past = 0;
    while (true) {
        std::string prompt;
        getline(std::cin, prompt);
        if (prompt == "END") {
            break;
        }

        const auto length = (int32_t) prompt.length();
        int32_t n_tokens = llama_tokenize(model, prompt.data(), length, tokens, length + add_bos, add_bos, first_run);

        llama_sampling_reset(ctxs);
        for (int32_t i = 0; i < n_tokens && n_tokens < n_batch; ++i) {
            llama_sampling_accept(ctxs, ctx, *(tokens + i), false);
        }

        if (llama_decode_tokens(ctx, tokens, n_tokens, n_batch, n_ctx, keep_bos, n_past)) {
            return 1; // Failed
        }

        llama_token token;
        do {
            token = llama_sampling_sample(ctxs, ctx, nullptr);
            llama_sampling_accept(ctxs, ctx, token, true);

            std::cout << llama_token_to_piece(ctx, token) << std::flush;

            if (llama_decode_tokens(ctx, &token, 1, n_batch, n_ctx, keep_bos, n_past)) {
                return 1; // Failed
            }
        } while (token != llama_token_eos(model));

        add_bos = false;
        first_run = false;
        std::cout << "\n\n";
    }

    llama_free(ctx);
    llama_free_model(model);

    llama_sampling_free(ctxs);
    llama_backend_free();

    return 0;
}

int32_t llama_decode_tokens(struct llama_context *ctx, llama_token *tokens, const int32_t n_tokens,
                            const int32_t n_batch, const int32_t n_ctx, bool keep_bos, int32_t &n_past) {
    if (n_past + n_tokens > n_ctx) {
        const int32_t n_discard = (n_past - keep_bos) / 2;

        llama_kv_cache_seq_rm(ctx, 0, keep_bos, keep_bos + n_discard);
        llama_kv_cache_seq_add(ctx, 0, keep_bos + n_discard, n_past, -n_discard);

        n_past -= n_discard;
    }

    for (int32_t i = 0; i < n_tokens; i += n_batch) {
        const int32_t n_eval = n_tokens - i;

        if (llama_decode(ctx, llama_batch_get_one(tokens + i, n_eval, n_past, 0))) {
            return 1; // Failed
        }
        n_past += n_eval;
    }

    return 0;
}
