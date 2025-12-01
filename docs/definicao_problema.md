# Definição do Problema

## 1. Contexto

O monitoramento da vazão de rios é essencial para diversos setores da sociedade brasileira. Vazões elevadas podem provocar enchentes e perdas materiais, enquanto vazões reduzidas impactam diretamente o abastecimento humano, a irrigação e a geração de energia hidrelétrica. Apesar da importância estratégica, as ferramentas existentes de previsão de vazão disponíveis ao público são limitadas ou pouco acessíveis.

Este projeto, desenvolvido para a disciplina **Tópicos Avançados em Inteligência Artificial (DEC0015/UFSC)**, busca preencher essa lacuna ao propor uma solução baseada em IA capaz de prever a vazão de rios brasileiros de forma acessível, transparente e aplicável a órgãos públicos, agricultores e usuários gerais.

## 2. Problema

Pretende-se resolver o seguinte problema:

**"Como prever a vazão diária de um rio brasileiro usando dados climáticos e hidrológicos históricos?"**

A solução deve:

- Utilizar dados oficiais da ANA e NASA POWER;
- Permitir previsões diárias de vazão;
- Apresentar os resultados de forma clara e acessível ao usuário final;
- Ser integrada a uma interface web funcional para consulta.

## 3. Escopo

O projeto contempla:

- Coleta de dados hidrológicos (vazão diária da ANA).
- Coleta de dados climáticos (precipitação, temperatura e umidade via NASA POWER / Open-Meteo).
- Preparação, limpeza e alinhamento temporal das séries.
- Desenvolvimento e teste de modelos de aprendizado de máquina (baseline + redes neurais, incluindo LSTM).
- Implementação de uma API para servir previsões.
- Construção de uma interface web simples para visualização.

## 4. Entregáveis

- Modelo de previsão treinado e validado.
- API/Backend para servir previsões.
- Dashboard web para visualização.
- Relatório técnico final no formato de artigo.
- Código-fonte completo no repositório.

## 5. Motivação

O Brasil apresenta forte dependência de recursos hídricos para geração de energia e agricultura. Logo, a previsão de vazão — combinando IA e dados abertos — representa um avanço na democratização do acesso à informação hidrológica e na capacidade de tomada de decisão.

