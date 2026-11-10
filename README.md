# RAG com Gemini, LangChain e PostgreSQL + pgVector

Aplicação de **ingestão de PDF, busca semântica e perguntas e respostas via terminal (CLI)**, desenvolvida para o desafio de Ingestão e Busca do MBA em Inteligência Artificial.

O projeto implementa um fluxo de **Retrieval-Augmented Generation (RAG)**: extrai o texto de um PDF, divide-o em fragmentos, gera embeddings com o Google Gemini, armazena os vetores no PostgreSQL com pgVector e recupera trechos relevantes para fundamentar as respostas da IA. As respostas devem utilizar somente as informações presentes no documento; caso o contexto não contenha a informação necessária, o assistente deve informar que não pode responder.

> **Escopo:** a aplicação é executada localmente no terminal. Ela utiliza a API Google Gemini, sujeita à disponibilidade dos modelos, às cotas e às condições de cobrança da conta utilizada.

## Sumário

- [Visão geral e arquitetura](#visão-geral-e-arquitetura)
- [Requisitos obrigatórios do desafio](#requisitos-obrigatórios-do-desafio)
- [Pré-requisitos](#pré-requisitos)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Instalação e configuração](#instalação-e-configuração)
- [Execução](#execução)
- [Exemplos de uso e resultados](#exemplos-de-uso-e-resultados)
- [Validação no PostgreSQL](#validação-no-postgresql)
- [Testes de aceitação](#testes-de-aceitação)
- [Troubleshooting](#troubleshooting)
- [Observações técnicas e segurança](#observações-técnicas-e-segurança)

## Visão geral e arquitetura

```text
                           document.pdf
                                |
                                v
                      src/ingest.py
                  PyPDFLoader (34 páginas)
                                |
                                v
                  RecursiveCharacterTextSplitter
                 chunk_size=1000 / overlap=150
                                |
                                v
             Gemini Embeddings (768 dimensões)
                                |
                                v
                  PostgreSQL + pgVector
                      coleção pdf_chunks
                                ^
                                |
                   busca vetorial (k=10)
                                ^
                                |
         pergunta --> src/search.py --> prompt + Gemini Chat
                                              |
                                              v
                  src/chat.py <--- resposta fundamentada
```

**Componentes:**

| Componente | Responsabilidade |
| --- | --- |
| `src/ingest.py` | Extrair texto, dividir em chunks, gerar embeddings e persistir no banco. |
| `src/search.py` | Vetorizar perguntas, recuperar os 10 chunks mais próximos e solicitar uma resposta à LLM. |
| `src/chat.py` | Disponibilizar um chat interativo via terminal. |
| PostgreSQL + pgVector | Armazenar o texto, os metadados e os vetores. |
| Google Gemini | Gerar embeddings e respostas a partir do contexto recuperado. |

## Requisitos obrigatórios do desafio

| Requisito | Implementação |
| --- | --- |
| Linguagem Python | Scripts da pasta `src/`. |
| Framework LangChain | Integrações de PDF, divisão de texto, embeddings, busca e LLM. |
| PostgreSQL com extensão pgVector | Ambiente definido em `docker-compose.yml`. |
| Docker e Docker Compose | Inicialização do banco e da extensão vetorial. |
| Leitura de PDF | `PyPDFLoader`. |
| Fragmentos de 1.000 caracteres com sobreposição de 150 | `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)`. |
| Vetorização e persistência dos fragmentos | Gemini Embeddings + `PGVector`. |
| Perguntas pela CLI | `src/chat.py`. |
| Recuperação de 10 resultados relevantes | `similarity_search_with_score(..., k=10)`. |
| Respostas fundamentadas somente no PDF | `PROMPT_TEMPLATE` em `src/search.py`. |
| Resposta padrão para perguntas sem informação suficiente | `Não tenho informações necessárias para responder sua pergunta.` |

O sistema utiliza uma busca **semântica**, não uma consulta SQL exata por nome de empresa. A presença de informações relevantes nos resultados depende da recuperação dos trechos e do comportamento da LLM.

## Pré-requisitos

- **Python 3.14.6** — versão usada no desenvolvimento e nos testes locais. Outras versões podem funcionar, mas não foram validadas aqui.
- **pip** e **venv** para gerenciar dependências em ambiente virtual.
- **Docker** com suporte ao **Docker Compose** (`docker compose`).
- **Chave de API do Google Gemini** com acesso aos modelos configurados.
- Conexão com a internet para instalação de pacotes e chamadas à API.
- **DBeaver** (opcional), para inspecionar os dados do PostgreSQL.

> **Importante:** o arquivo `requirements.txt` deve conter as versões efetivamente testadas com Python 3.14.6. Antes da entrega, atualize-o a partir do ambiente funcional e valide a instalação em um novo ambiente virtual. As versões originalmente fornecidas no repositório podem ser incompatíveis com Python 3.14.

## Estrutura do projeto

```text
mba-ia-desafio-ingestao-busca/
├── .env.example
├── .gitignore
├── README.md
├── docker-compose.yml
├── document.pdf
├── requirements.txt
└── src/
    ├── ingest.py
    ├── search.py
    └── chat.py
```

O arquivo `.env` e o ambiente virtual `venv/` são locais e **não devem ser enviados ao GitHub**.

## Instalação e configuração

Execute os comandos abaixo **a partir da raiz do repositório**.

### 1. Clonar o repositório

```bash
git clone https://github.com/eltongarbin/mba-ia-desafio-ingestao-busca.git
cd mba-ia-desafio-ingestao-busca
```

### 2. Criar e ativar o ambiente virtual

```bash
python3.14 -m venv venv
source venv/bin/activate
python --version
```

No Windows, a ativação pode ser feita com `venv\Scripts\activate`, desde que haja uma versão compatível do Python instalada.

### 3. Instalar as dependências

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

O comando `pip check` deve terminar sem apontar conflitos de dependências.

### 4. Configurar as variáveis de ambiente

Crie o arquivo `.env` usando `.env.example` como base:

```bash
cp .env.example .env
```

Garanta que ele contenha as configurações abaixo (substitua o valor da chave por uma chave válida):

```dotenv
GOOGLE_API_KEY=your_google_api_key
GOOGLE_EMBEDDING_MODEL=gemini-embedding-001
GOOGLE_CHAT_MODEL=gemini-3.5-flash-lite
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=pdf_chunks
PDF_PATH=document.pdf
```

Crie a chave no [Google AI Studio](https://aistudio.google.com/api-keys). Verifique no seu projeto Google quais modelos de chat estão disponíveis; se necessário, altere `GOOGLE_CHAT_MODEL` para um modelo acessível e compatível com o código.

> O `.env.example` do repositório original pode conter modelos antigos. Utilize os valores acima ou outros modelos compatíveis, mantendo o **mesmo modelo de embedding e as mesmas dimensões** na ingestão e na busca. Esta implementação utiliza vetores de **768 dimensões**.

### 5. Subir o PostgreSQL com pgVector

```bash
docker compose up -d
docker compose ps
```

O Compose cria o serviço `postgres`, usando a imagem `pgvector/pgvector:pg17`, e executa a criação da extensão `vector` por meio do serviço `bootstrap_vector_ext`.

Configuração local padrão:

| Campo | Valor |
| --- | --- |
| Host | `localhost` |
| Porta | `5432` |
| Banco | `rag` |
| Usuário | `postgres` |
| Senha | `postgres` |
| Serviço Docker Compose | `postgres` |

> As credenciais são de **desenvolvimento local**. Não utilize essas senhas em produção.

## Execução

### 1. Ingerir o PDF

```bash
python src/ingest.py
```

O programa carrega o `document.pdf`, divide o texto em chunks, gera embeddings e os armazena no PostgreSQL. Os dados já presentes, quando identificados pelos mesmos IDs determinísticos, não são processados novamente.

**Exemplo de saída da primeira execução:**

```text
Carregando PDF...
Total de páginas: 34
Total de chunks gerados: 67

Inicializando embeddings do Gemini...
Conectando ao PostgreSQL...

Chunks já armazenados: 0
Chunks pendentes: 67
Armazenados 5/67 chunks pendentes
Armazenados 10/67 chunks pendentes
...
Armazenados 67/67 chunks pendentes

Ingestão concluída com sucesso!
```

**Exemplo de saída ao repetir a ingestão sem alterar o PDF:**

```text
Total de páginas: 34
Total de chunks gerados: 67
Chunks já armazenados: 67
Chunks pendentes: 0
Ingestão concluída com sucesso!
```

Os números acima correspondem ao PDF fornecido no desafio e aos testes locais. Se o documento ou a configuração mudar, a quantidade de chunks poderá ser diferente.

### 2. Fazer uma pergunta diretamente

```bash
python src/search.py "Qual é o faturamento da Alfa Energia S.A.?"
```

**Exemplo de saída:**

```text
Pergunta: Qual é o faturamento da Alfa Energia S.A.?
Buscando documentos e gerando resposta...

Resposta: O faturamento da Alfa Energia S.A. é de R$ 722.875.391,46.
```

### 3. Iniciar o chat interativo

```bash
python src/chat.py
```

O chat aceita múltiplas perguntas durante a mesma execução e termina ao digitar `sair`, `exit` ou `quit`.

**Exemplo de interação:**

```text
Chat iniciado. Digite 'sair' para encerrar.

Você: Em que ano foi fundada a Alfa Energia S.A.?

Assistente: A Alfa Energia S.A. foi fundada em 1972.

Você: Qual é a capital da França?

Assistente: Não tenho informações necessárias para responder sua pergunta.

Você: sair
Chat encerrado.
```

**Observação:** cada pergunta é recuperada independentemente. O chat não implementa memória de conversação; prefira perguntas completas que mencionem explicitamente a empresa ou informação desejada.

## Exemplos de uso e resultados

| Pergunta | Resultado esperado |
| --- | --- |
| `Qual é o faturamento da Alfa Energia S.A.?` | `R$ 722.875.391,46` |
| `Em que ano foi fundada a Alfa Energia S.A.?` | `1972` |
| `Qual é o faturamento da Alfa IA Indústria?` | `R$ 548.789.613,65` |
| `Qual é a capital da França?` | `Não tenho informações necessárias para responder sua pergunta.` |
| `Você acha que a Alfa Energia S.A. é um bom investimento?` | `Não tenho informações necessárias para responder sua pergunta.` |

Os exemplos são **resultados esperados**, não respostas garantidas em todas as execuções. A aplicação usa uma LLM; a qualidade depende dos trechos recuperados e das regras do prompt.

## Validação no PostgreSQL

É possível conectar o [DBeaver](https://dbeaver.io/) ao banco usando os dados da seção de instalação. Abra um editor SQL e execute:

```sql
SELECT
    COUNT(*) AS total_embeddings,
    COUNT(DISTINCT e.document) AS unique_documents,
    MIN(vector_dims(e.embedding)) AS min_dimensions,
    MAX(vector_dims(e.embedding)) AS max_dimensions
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c
    ON e.collection_id = c.uuid
WHERE c.name = 'pdf_chunks';
```

Para o PDF e a configuração testados, o resultado esperado é:

```text
total_embeddings | unique_documents | min_dimensions | max_dimensions
-----------------+------------------+----------------+---------------
67               | 67               | 768            | 768
```

Para consultar um exemplo de texto e seus metadados:

```sql
SELECT e.document, e.cmetadata
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c
    ON e.collection_id = c.uuid
WHERE c.name = 'pdf_chunks'
LIMIT 5;
```

## Testes de aceitação

Antes de entregar, verifique:

- [ ] `python -m pip check` não apresenta conflitos.
- [ ] `docker compose ps` mostra o PostgreSQL em execução.
- [ ] A primeira ingestão cria os chunks e persiste os embeddings.
- [ ] O DBeaver confirma **67 documentos únicos** com **768 dimensões**.
- [ ] Uma segunda ingestão exibe **0 chunks pendentes**.
- [ ] `search.py` recupera corretamente faturamento e ano de fundação.
- [ ] Perguntas fora do contexto retornam a mensagem de informação insuficiente.
- [ ] `chat.py` aceita perguntas consecutivas e encerra com `sair`.
- [ ] `.env`, `venv/` e credenciais não estão versionados.
- [ ] O `requirements.txt` permite instalar o projeto em um ambiente virtual limpo.

## Troubleshooting

### `ModuleNotFoundError: No module named 'dotenv'` ou `langchain_community`

Confirme que o ambiente virtual está ativo e instale todas as dependências:

```bash
source venv/bin/activate
python -m pip install -r requirements.txt
```

### Falha na instalação de dependências com Python 3.14

Versões antigas de `psycopg-binary`, `psycopg2-binary`, `pydantic-core` ou `tiktoken` podem ser incompatíveis com Python 3.14. Utilize um `requirements.txt` gerado a partir de um ambiente funcional e com versões compatíveis. Para verificar conflitos:

```bash
python --version
python -m pip check
```

O erro `PyO3's maximum supported version` geralmente indica uma dependência nativa antiga que precisa ser atualizada, não um erro no código da aplicação.

### `429 RESOURCE_EXHAUSTED` no Gemini

Indica que alguma cota da API foi excedida (por exemplo, tokens por minuto). O `ingest.py` utiliza processamento em lotes, pausas e novas tentativas para lidar com limites temporários. Aguarde a liberação da cota e execute novamente; os chunks com IDs já armazenados podem ser ignorados.

Consulte [limites e uso da API Gemini](https://ai.google.dev/gemini-api/docs/rate-limits). A camada gratuita não elimina os limites de utilização.

### Erro de autenticação ou modelo não disponível no Gemini

Confira `GOOGLE_API_KEY`, `GOOGLE_EMBEDDING_MODEL` e `GOOGLE_CHAT_MODEL` no `.env`. Verifique se a chave tem permissão para usar os modelos selecionados. A disponibilidade dos modelos pode mudar ao longo do tempo.

### Banco indisponível ou `connection refused`

Verifique o Docker e a porta local:

```bash
docker compose up -d
docker compose ps
docker compose logs postgres
```

Confira o valor de `DATABASE_URL` e se a porta `5432` não está ocupada por outra instância do PostgreSQL.

### Erro `relation "langchain_pg_embedding" does not exist`

A tabela é criada pela integração durante a inicialização da coleção. Execute a ingestão antes de consultar os dados no DBeaver. Verifique também se o DBeaver está conectado ao banco `rag` correto.

### Quantidade de embeddings maior que a quantidade de chunks

Pode ocorrer quando diferentes execuções inserem o mesmo texto com IDs distintos. O script utiliza IDs determinísticos para evitar novas duplicações em execuções idênticas. Evite excluir registros de uma coleção compartilhada sem conferir o conteúdo; em ambiente de teste, é possível reiniciar o banco do zero.

### Dimensão do vetor incompatível

A implementação atual utiliza embeddings com **768 dimensões**. Se mudar o modelo de embeddings ou a dimensão configurada, os registros antigos podem não ser compatíveis. Em um ambiente de teste descartável, recrie a coleção e execute novamente a ingestão.

### Aviso de depreciação do `langchain-community`

O aviso não impede necessariamente o funcionamento do `PyPDFLoader`. Ele indica que a integração poderá precisar de migração para um pacote mantido separadamente em versões futuras.

### `ValueError: Unexpected model response format`

O Gemini pode devolver conteúdo estruturado em blocos. Na integração utilizada, prefira extrair a resposta textual com `response.text`, em vez de supor que `response.content` seja sempre uma string.

### Reiniciar o banco de dados do zero

**Atenção: remove permanentemente os dados nos volumes gerenciados por este Compose.** Use apenas em ambiente de teste quando não precisar preservar os embeddings.

```bash
docker compose down -v
docker compose up -d
python src/ingest.py
```

Esse procedimento não apaga dependências Python nem o código-fonte. Caso a persistência use um bind mount, os arquivos do host não são excluídos por `down -v`.

---

**Projeto acadêmico:** desafio de Ingestão e Busca Semântica com Python, LangChain, PostgreSQL + pgVector e Google Gemini.
