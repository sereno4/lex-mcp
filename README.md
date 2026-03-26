# ⚖️ LEX - Assistente Jurídico MCP

> Assistente jurídico com busca de modelos, análise de textos e exploração de datasets via Hugging Face Hub + Gradio

## Funcionalidades

- 🔍 Buscar modelos jurídicos no HuggingFace por query e idioma
- 📝 Analisar textos jurídicos (sumarização e classificação)
- 📊 Explorar datasets jurídicos brasileiros

## Como rodar localmente

### 1. Clone o repositório
```bash
git clone https://github.com/sereno4/lex-mcp.git
cd lex-mcp
```

### 2. Instale as dependências
```bash
pip install gradio huggingface_hub datasets httpx
```

### 3. Configure o token
```powershell
# Windows PowerShell
$env:HF_TOKEN = "hf_SEU_TOKEN_AQUI"
```
> Gere seu token em https://huggingface.co/settings/tokens
> Permissão necessária: Inference > Make calls to serverless Inference Providers

### 4. Rode
```bash
py app.py
```

Acesse em: http://localhost:7860

## Tecnologias

- [Gradio](https://gradio.app) — interface web
- [Hugging Face Hub](https://huggingface.co) — modelos e datasets
- [Datasets](https://huggingface.co/docs/datasets) — exploração de dados

## Autor

[@sereno4](https://github.com/sereno4)
