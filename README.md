# Mecânica dos Fluidos — Projeto 1: Escoamento de Hagen–Poiseuille com MEF e FEniCSx

Repositório do **Projeto 1 de Mecânica dos Fluidos**, dedicado ao estudo do escoamento laminar estacionário de Hagen–Poiseuille em um tubo cilíndrico, desde a modelagem física e dedução analítica até a formulação variacional e solução pelo **Método dos Elementos Finitos (MEF)** com **FEniCSx (`dolfinx`)**.

---

## Estrutura do Repositório

- [`Tarefa_01/Projeto_1_Hagen_Poiseuille_FEniCSx_1.ipynb`](Tarefa_01/Projeto_1_Hagen_Poiseuille_FEniCSx_1.ipynb): **Notebook Jupyter principal (executado)** contendo toda a fundamentação teórica, respostas às questões conceituais, código em Python/FEniCSx, tabelas de erro/convergência e gráficos.
- [`Tarefa_01/executar_projeto1.py`](Tarefa_01/executar_projeto1.py): Script Python completo com todo o código do projeto para execução direta via terminal.
- [`Tarefa_01/figuras/`](Tarefa_01/figuras/): Gráficos em alta resolução (`300 dpi`) gerados pelo código para inclusão no artigo em formato SBC:
  - `fig1_verificacao_N4.png`: Verificação inicial do perfil de velocidade ($N = 4$, $P_1$).
  - `fig2_perfis_malha_P1.png`: Evolução do perfil numérico para $N = 2, 4, 8, 16, 32$ ($P_1$).
  - `fig3_convergencia_P1.png`: Taxa de convergência quadrática $\mathcal{O}(h^2)$ na norma $L^2$ e erro máximo nodal.
  - `fig4_comparacao_P1_P2.png`: Comparação entre elementos lineares ($P_1$) e quadráticos ($P_2$).
  - `fig5_convergencia_vazao.png`: Convergência da vazão volumérica numérica $Q_h$ para $Q_{\text{exato}}$.
  - `fig6_investigacao_fisica.png`: Estudo paramétrico ($Q \propto \Delta p$, $Q \propto 1/\eta$, $Q \propto a^4$).
- `Tarefa_01/Aula_1_Fluidos.ipynb`, `Aula_2_Fluidos.ipynb`, `Aula_3_Fluidos_MEF_FEniCSx.ipynb`: Notebooks de referência das aulas.

---

## Como Executar (WSL / Ubuntu com FEniCSx)

1. No terminal do Ubuntu (WSL), instale o FEniCSx caso ainda não esteja instalado:
   ```bash
   sudo apt update
   sudo apt install -y software-properties-common
   sudo add-apt-repository -y ppa:fenics-packages/fenics
   sudo apt update
   sudo apt install -y fenicsx python3-dolfinx python3-matplotlib python3-sympy python3-ipykernel python3-nbconvert
   ```

2. Abra o notebook `Tarefa_01/Projeto_1_Hagen_Poiseuille_FEniCSx_1.ipynb` no VS Code conectado ao WSL e clique em **Run All** (ou execute `python3 Tarefa_01/executar_projeto1.py`).
