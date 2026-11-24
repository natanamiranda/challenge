Medical AI Assistant

Assistente médico baseado em Inteligência Artificial Generativa e Retrieval-Augmented Generation (RAG), desenvolvido como solução de apoio à consulta de informações médicas.

O sistema utiliza uma base de conhecimento médica para recuperar informações relevantes, um modelo de linguagem local para geração de respostas e mecanismos de validação para identificar conteúdos que exigem revisão humana.

Aviso: Este projeto possui finalidade educacional e demonstrativa. Não substitui avaliação, diagnóstico ou orientação de profissionais de saúde. As respostas geradas não possuem aprovação clínica automática.

1. Objetivos

O projeto tem como objetivos:

Desenvolver um assistente médico utilizando Inteligência Artificial Generativa.

Implementar uma arquitetura RAG para recuperação de informações médicas.

Utilizar modelos locais por meio do Ollama.

Construir uma base vetorial com documentos médicos.

Integrar informações fictícias de pacientes por meio de banco de dados relacional.

Implementar validações de segurança e identificação de respostas que exigem revisão humana.

Disponibilizar uma interface para interação e demonstração da solução.

2. Arquitetura

A solução foi organizada em componentes independentes, separando interface, processamento, recuperação de conhecimento, geração de respostas, validação e persistência de dados.

2.1 Diagrama de arquitetura

flowchart TD
    A[Usuário] --> B[Dashboard Streamlit]
    B --> C[Medical Assistant]
    C --> D[RAG]
    D --> E[Base Vetorial ChromaDB]
    D --> F[Base MedQuAD]
    C --> G[Ollama - Qwen 2.5 3B]
    C --> H[Validação de Segurança]
    B --> I[Patient Service]
    I --> J[SQLite]
    H --> K[Resposta e Revisão Humana]
    K --> B

2.2 Componentes

Componente

Responsabilidade

Dashboard

Interface de interação com o usuário, desenvolvida em Streamlit.

Medical Assistant

Orquestra o processamento das perguntas e a geração das respostas.

RAG

Recupera documentos relevantes da base de conhecimento.

ChromaDB

Armazena os vetores utilizados na busca semântica.

MedQuAD

Base de conhecimento médica utilizada na recuperação de informações.

Ollama

Executa localmente o modelo de linguagem e o modelo de embeddings.

Safety

Realiza verificações lexicais e identifica conteúdos que exigem atenção.

Patient Service

Consulta informações fictícias de pacientes.

SQLite

Persiste os dados estruturados dos pacientes, exames, alergias e condições.

2.3 Fluxo de processamento

O usuário envia uma pergunta pela interface.

O assistente identifica o tipo de solicitação.

O componente RAG realiza a busca semântica na base vetorial.

Os documentos recuperados são utilizados como contexto para a resposta.

O modelo de linguagem local pode ser utilizado na geração de respostas.

O componente de segurança verifica o conteúdo produzido.

A resposta é apresentada com suas fontes e indicação de necessidade de revisão humana.

3. Tecnologias utilizadas

Tecnologia

Aplicação

Python 3.11

Linguagem principal

LangChain

Integração dos componentes de RAG

ChromaDB

Banco de dados vetorial

Ollama

Execução local dos modelos

Qwen 2.5 3B

Modelo de linguagem

Nomic Embed Text

Modelo de embeddings

SQLite

Banco de dados relacional

SQLAlchemy

Mapeamento objeto-relacional

Streamlit

Interface da aplicação

MedQuAD

Base de conhecimento médica

4. Estrutura do projeto

medical-ai-assistant/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── medquad/
│   ├── processed/
│   │   └── medquad.jsonl
│   ├── protocols/
│   │   └── medical_glossary.json
│   ├── synthetic/
│   └── chroma_db/
│
├── docs/
│   └── relatorio_tecnico.md
│
├── notebooks/
│
├── outputs/
│
├── scripts/
│   ├── prepare_medquad.py
│   ├── ingest_documents.py
│   ├── seed_database.py
│   ├── evaluate_assistant.py
│   ├── test_medquad.py
│   ├── test_rag.py
│   ├── test_safety.py
│   ├── test_assistant.py
│   ├── test_patient_service.py
│   ├── test_retrieval_quality.py
│   └── check_vector_db.py
│
├── src/
│   └── medical_assistant/
│       ├── assistant.py
│       ├── config.py
│       ├── database.py
│       ├── models.py
│       ├── patient_service.py
│       ├── rag.py
│       └── safety.py
│
├── tests/
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt

Observação: os scripts de validação estão organizados no diretório scripts/ e são executados diretamente pelo Python. O diretório tests/ está reservado para uma possível expansão futura de testes automatizados.

5. Configuração do projeto

5.1 Pré-requisitos

Antes de iniciar, instale:

Python 3.11.

Git.

Ollama.

O projeto utiliza modelos executados localmente pelo Ollama, sem necessidade de uma API externa para inferência.

5.2 Clonar o repositório

git clone <URL_DO_REPOSITORIO>
cd medical-ai-assistant

5.3 Criar e ativar o ambiente virtual

No Windows PowerShell:

python -m venv .venv

Ative o ambiente:

.\.venv\Scripts\Activate.ps1

Caso necessário, permita a execução de scripts na sessão atual:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

5.4 Instalar as dependências

pip install -r requirements.txt

5.5 Configurar o Ollama

Instale e inicie o Ollama.

Baixe os modelos utilizados:

ollama pull qwen2.5:3b

ollama pull nomic-embed-text

Verifique os modelos instalados:

ollama list

5.6 Configurar as variáveis de ambiente

Crie o arquivo .env na raiz do projeto, utilizando o .env.example como referência.

Configuração:

APP_NAME=Medical AI Assistant
APP_ENV=development
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
DATABASE_URL=sqlite:///./data/medical_assistant.db
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

As variáveis definem o nome da aplicação, ambiente de execução, endereço do Ollama, modelo de linguagem, conexão com o SQLite e modelo de embeddings.

Importante: o arquivo .env contém configurações locais e não deve ser versionado. Utilize o .env.example como modelo de configuração.

5.7 Preparar a base de conhecimento MedQuAD

O projeto utiliza a base MedQuAD para disponibilizar informações médicas durante a recuperação de documentos.

Para obter a base de conhecimento:

git clone https://github.com/abachaa/MedQuAD.git data/raw/medquad

Em seguida, execute o processamento dos arquivos XML:

python scripts/prepare_medquad.py

Esse processo:

Localiza os arquivos XML da base.

Extrai perguntas e respostas.

Organiza os metadados das fontes.

Remove registros inválidos ou sem conteúdo suficiente.

Gera o arquivo data/processed/medquad.jsonl.

5.8 Criar a base vetorial

Após preparar os documentos, execute:

python scripts/ingest_documents.py

Esse processo:

Carrega até 300 registros da base processada.

Divide os documentos em fragmentos.

Gera embeddings utilizando o modelo Nomic.

Armazena os vetores no ChromaDB.

A base vetorial é armazenada em data/chroma_db.

Atenção: o script de indexação remove a base vetorial anterior antes de criar uma nova. Execute-o somente quando precisar reconstruir o índice. Não é necessário repetir esse processo em ambientes que já possuem a base vetorial preparada.

5.9 Inicializar o banco de dados

Execute:

python scripts/seed_database.py

O script cria e popula o banco SQLite com pacientes fictícios, exames, alergias e condições clínicas para demonstração da integração com dados estruturados.

Os dados utilizados são sintéticos e não representam pacientes reais.

5.10 Executar a aplicação

Com o ambiente configurado, o Ollama em execução e as bases de conhecimento disponíveis, execute:

streamlit run dashboard/app.py

A interface será disponibilizada no navegador pelo Streamlit.

6. Avaliação e testes

O projeto possui scripts independentes para avaliar os principais componentes da solução. Eles são executados diretamente pelo interpretador Python, sem depender de uma suíte Pytest.

6.1 Avaliação do assistente

python scripts/evaluate_assistant.py

Avalia exemplos de perguntas e respostas, incluindo verificações de qualidade e segurança.

6.2 Testes dos componentes

Camada de segurança:

python scripts/test_safety.py

Verifica a identificação de expressões inadequadas, afirmações clínicas sensíveis e respostas sem fontes.

Recuperação RAG:

python scripts/test_rag.py

Verifica a recuperação de documentos relevantes na base vetorial.

Geração de respostas:

python scripts/test_assistant.py

Verifica o processamento de perguntas e a geração de respostas pelo assistente.

Serviço de pacientes:

python scripts/test_patient_service.py

Verifica a consulta dos dados fictícios de pacientes.

Qualidade da recuperação:

python scripts/test_retrieval_quality.py

Executa verificações específicas relacionadas à qualidade da recuperação de documentos.

6.3 Verificação da base vetorial

python scripts/check_vector_db.py

Permite verificar a estrutura e a quantidade de documentos indexados na base vetorial.

6.4 Resultados observados durante o desenvolvimento

Nos cenários executados durante o desenvolvimento:

O sistema recuperou informações relacionadas à leucemia linfoblástica aguda em adultos.

A busca RAG recuperou documentos relacionados à pergunta, incluindo informações da fonte CancerGov.

O assistente apresentou respostas em português a partir do conteúdo recuperado.

Os testes da camada de segurança identificaram corretamente os três cenários negativos avaliados.

Os testes de avaliação do assistente não identificaram problemas lexicais nos dois cenários avaliados.

As respostas permaneceram sinalizadas como pendentes de revisão humana.

A integração com o banco SQLite permitiu consultar dados fictícios de pacientes.

O dashboard permitiu selecionar pacientes, consultar informações e visualizar respostas e fontes.

Os resultados representam os cenários efetivamente avaliados durante o desenvolvimento e não constituem validação clínica, garantia de precisão médica ou cobertura completa de todos os casos possíveis.

7. Segurança e limitações

Por se tratar de uma aplicação relacionada à saúde, foram consideradas verificações para reduzir riscos de respostas inadequadas.

A solução contempla:

Verificações lexicais de segurança.

Identificação de expressões potencialmente inadequadas.

Verificação de termos sensíveis.

Identificação de respostas sem fontes recuperadas.

Indicação de necessidade de revisão humana.

Apresentação das fontes recuperadas pela camada RAG.

Uso de dados sintéticos para demonstração da integração com pacientes.

7.1 Revisão humana

As respostas geradas permanecem pendentes de revisão humana.

As verificações automáticas são mecanismos auxiliares e não comprovam a precisão semântica, factual ou clínica das informações.

A aprovação humana é obrigatória no fluxo demonstrativo. Entretanto, o protótipo não implementa um processo persistente de aprovação ou rejeição por profissionais de saúde.

7.2 Limitações conhecidas

A validação de segurança é baseada em regras lexicais e não representa uma avaliação clínica completa.

Não existe aprovação clínica automática.

A revisão humana é sinalizada, mas não há um fluxo persistente de aprovação ou rejeição.

O modelo de linguagem é utilizado em sua versão pré-treinada, sem fine-tuning específico para o domínio médico.

A recuperação depende da qualidade e da cobertura dos documentos disponíveis na base.

A indexação atual utiliza uma seleção limitada de até 300 registros da base MedQuAD.

A geração de respostas pode apresentar variações entre execuções.

A aplicação não substitui sistemas clínicos, profissionais de saúde ou protocolos institucionais.

8. Base de conhecimento

A principal base de conhecimento utilizada é a MedQuAD (Medical Question Answering Dataset).

Referência:

Ben Abacha, A.; Demner-Fushman, D. (2019). A Question-Entailment Approach to Question Answering. BMC Bioinformatics.

Repositório da base:

https://github.com/abachaa/MedQuAD

A base MedQuAD é disponibilizada sob licença CC BY 4.0, observadas as condições de atribuição e eventuais direitos aplicáveis aos materiais de origem.

9. Considerações finais

O Medical AI Assistant demonstra a aplicação de Inteligência Artificial Generativa integrada a uma arquitetura RAG para recuperação de informações médicas.

A solução combina recuperação semântica, modelos locais de linguagem, persistência de dados estruturados e verificações de segurança.

O projeto também evidencia a importância de transparência, rastreabilidade das fontes e revisão humana em aplicações de Inteligência Artificial no contexto da saúde.

10. Licença

Projeto desenvolvido para fins educacionais e acadêmicos.

A utilização e redistribuição dos componentes e bases de conhecimento de terceiros devem respeitar suas respectivas licenças.