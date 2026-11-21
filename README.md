# Inference Check

CLI open source para detectar regressões em APIs de IA compatíveis com Chat Completions. Roda fora do caminho das requisições da aplicação e funciona com o gateway que o time já usa.

| Teste | O que precisa passar |
| --- | --- |
| `structured_output` | O JSON respeita o schema solicitado e contém os valores esperados. |
| `tool_call` | A resposta chama a função esperada com os argumentos corretos. |
| `streaming_tool_call` | O stream SSE termina corretamente e os fragmentos formam uma tool call válida. |

## Demo em um minuto

Requer Python 3.10 ou superior. A demo usa um servidor HTTP local e dispensa chaves e dependências externas.

```sh
python3 demo.py
```

A primeira execução passa nos três testes. Na segunda, o mesmo endpoint perde um fragmento em 4 de 10 streams, com HTTP 200 em todas as chamadas. A CLI retorna código 1 e a demo valida esse resultado. Os relatórios ficam em `demo-reports/`.

```text
  streaming_tool_call: 6/10 passaram · HTTP 200 em 10/10
    invalid_tool_arguments_json: 4
    baseline: 10/10 (-40.0 p.p.)
```

## Teste sua rota

Defina `OPENAI_API_KEY` ou `OPENROUTER_API_KEY` no gerenciador de secrets do CI ou no ambiente local. Escolha um modelo para o qual a rota anuncie suporte a JSON Schema estrito e tool calls. Cada execução faz três chamadas por amostra, e o provedor pode cobrar por elas.

**OpenAI direto**

```sh
python3 inference_check.py check \
  --name openai-direto --url https://api.openai.com/v1/chat/completions \
  --model gpt-4o-mini --key-env OPENAI_API_KEY \
  --samples 10 --output baseline-openai.json
```

**OpenRouter**

```sh
python3 inference_check.py check \
  --name openrouter --url https://openrouter.ai/api/v1/chat/completions \
  --model openai/gpt-4o-mini --key-env OPENROUTER_API_KEY \
  --samples 10 --output baseline-openrouter.json
```

Adicione `--openrouter-provider SLUG` para solicitar um upstream específico sem fallback entre provedores. A comparação usa a rota solicitada. Sem fixação, as chamadas podem cair em upstreams diferentes.

### Comparar no CI

Repita o comando da rota com `--baseline baseline-openai.json --output atual-openai.json`, usando a mesma configuração registrada no baseline. Guarde o relatório de uma execução saudável como referência e preserve o atual mesmo se o passo falhar.

| Código | Significado |
| --- | --- |
| 0 | Todas as amostras passaram. |
| 1 | Ao menos uma amostra falhou. Consulte as contagens e motivos no relatório. |
| 2 | Erro nos argumentos, na configuração, no baseline ou ao salvar o relatório. |

O relatório JSON inclui horário UTC, versão da suíte, configuração da rota, contagem de passes e de HTTP 200, e motivos de falha. O baseline acrescenta a contagem anterior e o terminal mostra a diferença em pontos percentuais. A CLI usa prompts fixos e deixa chaves e respostas brutas fora do relatório. Falhas de transporte, status HTTP e falhas de contrato aparecem com motivos distintos.

A suíte mede esses três contratos no endpoint configurado e na amostra executada. Os resultados documentam o comportamento observado. Identidade do modelo, cobrança, retenção e localização do processamento exigem evidências próprias. As duas configurações de provedores acima são exemplos. A demo local verifica a execução e a detecção de regressão sem acessar APIs pagas.

## Contribuição

Para propor um novo teste, inclua um caso sintético, o critério de passagem e um exemplo de falha reproduzível. Execute `python3 demo.py` antes de enviar a mudança. O workflow do repositório roda essa demo em Python 3.10 e 3.12.

Licença [Apache-2.0](LICENSE).
