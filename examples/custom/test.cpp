#include <common.h>

void context_shifting(struct llama_context *ctx, uint32_t n_tokens, uint32_t n_ctx, int32_t &n_past);

int main() {
    const char *weights = "llama-2-7b-chat.Q4_K_M.gguf";

    llama_context_params cparams = llama_context_default_params();
    cparams.seed = 1707244793; // time(nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 100;

    llama_sampling_params sparams;

    llama_backend_init();

    llama_model *model = llama_load_model_from_file(weights, mparams);
    llama_context *ctx = llama_new_context_with_model(model, cparams);
    llama_sampling_context *ctxs = llama_sampling_init(sparams);

    const auto n_ctx = (int32_t) llama_n_ctx(ctx);
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);

    llama_token tokens[4096];
    bool add_bos = llama_should_add_bos_token(model);
    bool first_run = true;

    int32_t n_past = 0;
    while (true) {
        std::string prompt; // = "USER: What is CUDA?ASSISTANT: ";
        getline(std::cin, prompt);
        if (prompt == "END") {
            break;
        }

        const auto length = (int32_t) prompt.length();
        int32_t n_tokens = llama_tokenize(model, prompt.data(), length, tokens, length + add_bos, add_bos, first_run);

        llama_sampling_reset(ctxs);
        llama_batch_clear(batch);
        for (int32_t i = 0; i < n_tokens; ++i) {
            llama_batch_add(batch, *(tokens + i), n_past++, {0}, false);
        }

        batch.logits[n_past - 1] = true;

        context_shifting(ctx, n_tokens, n_ctx, n_past);
        if (llama_decode(ctx, batch)) {
            return 1; // Failed
        }

        llama_token token = llama_sampling_sample(ctxs, ctx, nullptr, n_past - 1);
        llama_sampling_accept(ctxs, ctx, token, false);

        do {
            std::cout << llama_token_to_piece(ctx, token) << std::flush;

            llama_batch_clear(batch);
            llama_batch_add(batch, token, n_past++, {0}, true);

            context_shifting(ctx, 1, n_ctx, n_past);
            if (llama_decode(ctx, batch)) {
                return 1; // Failed
            }

            token = llama_sampling_sample(ctxs, ctx, nullptr);
            llama_sampling_accept(ctxs, ctx, token, true);
        } while (token != llama_token_eos(model));

        add_bos = false;
        first_run = false;
        std::cout << "\n\n";
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_sampling_free(ctxs);
    llama_batch_free(batch);
    llama_backend_free();

    return 0;
}

void context_shifting(struct llama_context *ctx, const uint32_t n_tokens, const uint32_t n_ctx, int32_t &n_past) {
    if (n_past + n_tokens > n_ctx) {
        const int32_t n_discard = (n_past - 1) / 2;

        llama_kv_cache_seq_rm(ctx, 0, 1, n_discard + 1);
        llama_kv_cache_seq_add(ctx, 0, 1 + n_discard, n_past, -n_discard);

        n_past -= n_discard;
    }
}
