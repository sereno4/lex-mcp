import os, gradio as gr
from huggingface_hub import HfApi, InferenceClient
from datasets import load_dataset

HF_TOKEN = os.getenv("HF_TOKEN") or ""
_api = HfApi(token=HF_TOKEN or None)

def search_models(query, language, limit):
    limit = int(limit)
    try:
        # busca sem filtro de idioma para garantir resultados
        results = list(_api.list_models(
            search=query,
            sort="downloads",
            limit=limit * 4,
        ))
        # filtra por idioma manualmente se informado
        if language:
            filtered = [m for m in results if language in (m.tags or [])]
            results = filtered if filtered else results
        results.sort(key=lambda m: m.downloads or 0, reverse=True)
        results = results[:limit]
        if not results:
            return "Nenhum modelo encontrado."
        output = ""
        for m in results:
            tags = ", ".join(m.tags or [])
            output += f"**{m.modelId}**\n- Task: {m.pipeline_tag}\n- Downloads: {(m.downloads or 0):,}\n- Tags: {tags}\n- URL: https://huggingface.co/{m.modelId}\n\n"
        return output
    except Exception as e:
        return f"Erro: {str(e)}"

def analyze_text(text, task):
    if not HF_TOKEN:
        return "HF_TOKEN nao configurado. Defina a variavel de ambiente antes de rodar."
    try:
        models = {
            "summarization": "facebook/bart-large-cnn",
            "text-classification": "nlpaueb/legal-bert-base-uncased",
        }
        model_id = models.get(task, "facebook/bart-large-cnn")
        client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)
        if task == "summarization":
            result = client.summarization(text[:512], model=model_id)
            return str(result.summary_text) if hasattr(result, "summary_text") else str(result)
        else:
            result = client.text_classification(text[:512], model=model_id)
            return "\n".join(f"{r.label}: {r.score:.4f}" for r in result)
    except Exception as e:
        return f"Erro: {str(e)}"

def explore_dataset(dataset_id, n_samples):
    n_samples = int(n_samples)
    try:
        ds = load_dataset(
            dataset_id,
            split=f"train[:{n_samples}]",
            token=HF_TOKEN or None,
            trust_remote_code=False,
        )
        output = f"**Dataset:** {dataset_id}\n\n**Colunas:** {', '.join(ds.column_names)}\n\n"
        for i, sample in enumerate(ds.to_list()):
            output += f"--- Amostra {i+1} ---\n"
            for key, val in sample.items():
                if isinstance(val, str) and len(val) > 300:
                    val = val[:300] + "..."
                output += f"{key}: {val}\n"
            output += "\n"
        return output
    except Exception as e:
        return f"Erro: {str(e)}"

with gr.Blocks(title="LEX - Assistente Juridico") as demo:
    gr.Markdown("# ⚖️ LEX - Assistente Juridico MCP")
    gr.Markdown("Powered by Hugging Face Hub + FastMCP")
    if not HF_TOKEN:
        gr.Markdown("⚠️ **HF_TOKEN nao configurado.** Busca funciona sem token, mas analise de texto requer autenticacao.")

    with gr.Tab("🔍 Buscar Modelos"):
        with gr.Row():
            with gr.Column(scale=1):
                model_query = gr.Textbox(label="Query", value="legal", placeholder="Ex: legal, juridico, NLP...")
                model_lang  = gr.Textbox(label="Idioma (opcional)", value="", placeholder="Ex: pt, en")
                model_limit = gr.Slider(1, 20, value=5, step=1, label="Limite")
                model_btn   = gr.Button("🔎 Buscar", variant="primary")
            with gr.Column(scale=2):
                model_output = gr.Textbox(label="Resultados", lines=15, interactive=False)
        model_btn.click(fn=search_models, inputs=[model_query, model_lang, model_limit], outputs=model_output)

    with gr.Tab("📝 Analisar Texto"):
        with gr.Row():
            with gr.Column(scale=1):
                text_input  = gr.Textbox(label="Texto Juridico", lines=5, placeholder="Cole aqui o texto para analisar...")
                task_type   = gr.Dropdown(choices=["summarization", "text-classification"], value="summarization", label="Tarefa")
                analyze_btn = gr.Button("▶️ Analisar", variant="primary")
            with gr.Column(scale=2):
                analyze_output = gr.Textbox(label="Resultado", lines=10, interactive=False)
        analyze_btn.click(fn=analyze_text, inputs=[text_input, task_type], outputs=analyze_output)

    with gr.Tab("📊 Explorar Dataset"):
        with gr.Row():
            with gr.Column(scale=1):
                dataset_input = gr.Textbox(label="Dataset ID", value="joelniklaus/brazilian_court_decisions")
                sample_count  = gr.Slider(1, 10, value=3, step=1, label="Amostras")
                dataset_btn   = gr.Button("🔍 Explorar", variant="primary")
            with gr.Column(scale=2):
                dataset_output = gr.Textbox(label="Dataset Info", lines=15, interactive=False)
        dataset_btn.click(fn=explore_dataset, inputs=[dataset_input, sample_count], outputs=dataset_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
