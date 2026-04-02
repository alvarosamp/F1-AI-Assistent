# 🏎️ F1 AI Race Engineer

Sistema inteligente de simulação e estratégia de corridas de Fórmula 1 utilizando Machine Learning, simulação computacional e observabilidade com OpenTelemetry.

---

## 📌 Visão Geral

Este projeto tem como objetivo simular uma corrida de Fórmula 1 e aplicar técnicas de Inteligência Artificial para:

* Prever tempo de volta
* Estimar desgaste de pneus
* Definir estratégias de pit stop
* Simular corridas completas
* Monitorar desempenho do sistema com observabilidade

---

## 🧠 Arquitetura do Sistema

```
Dados reais (FastF1)
        ↓
Engenharia de Features
        ↓
Modelo de Machine Learning
        ↓
Simulador de Corrida
        ↓
API (FastAPI)
        ↓
Observabilidade (OpenTelemetry)
```

---

## 📁 Estrutura do Projeto

```
f1-ai-engineer/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── simulation/
│   ├── strategy/
│   └── utils/
│
├── api/
├── observability/
├── dashboard/
├── tests/
│
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## ⚙️ Tecnologias Utilizadas

* Python
* FastF1 (dados reais de F1)
* Pandas / NumPy
* Scikit-learn / XGBoost
* FastAPI
* Streamlit
* OpenTelemetry
* Docker (futuro)

---

## 🚀 Como Rodar o Projeto

### 1. Clone o repositório

```
git clone https://github.com/seu-usuario/f1-ai-engineer.git
cd f1-ai-engineer
```

---

### 2. Crie o ambiente virtual

```
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Windows
.venv\Scripts\activate
```

---

### 3. Instale as dependências

```
pip install -r requirements.txt
```

---

## 📊 Coleta de Dados

```
python src/data/load_data.py
```

Este script utiliza a biblioteca FastF1 para coletar dados reais de corridas.

---

## 🧠 Treinamento do Modelo

```
python src/models/train_model.py
```

O modelo utiliza algoritmos de Machine Learning para prever o tempo de volta.

---

## 🏁 Simulação de Corrida

```
python src/simulation/race_simulator.py
```

A simulação executa uma corrida completa considerando:

* Desgaste de pneus
* Tempo de volta
* Estratégia de pit stop

---

## 🌐 API

Para iniciar a API:

```
uvicorn api.main:app --reload
```

Acesse:

```
http://127.0.0.1:8000/simulate
```

---

## 🔍 Observabilidade (OpenTelemetry)

O sistema utiliza OpenTelemetry para monitorar:

* Tempo de execução do modelo
* Tempo da simulação
* Latência da API

Exemplo de execução:

```
opentelemetry-instrument uvicorn api.main:app --reload
```

---

## 📈 Próximas Melhorias

* Implementar Reinforcement Learning para estratégia
* Adicionar múltiplos pilotos na simulação
* Integração com dados climáticos
* Dashboard interativo com Streamlit
* Deploy com Docker + Kubernetes
* Monitoramento com Prometheus + Grafana

---

## 🎯 Objetivo do Projeto

Este projeto foi desenvolvido com foco em:

* Aplicação prática de Machine Learning
* Construção de sistemas de simulação
* Engenharia de dados
* Arquitetura de sistemas escaláveis
* Preparação para ambientes de produção

---

## 👨‍💻 Autor

Desenvolvido por [Seu Nome]

---

## 📄 Licença

Este projeto está sob a licença MIT.
