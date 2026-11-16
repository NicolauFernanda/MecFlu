# Guia Passo a Passo: Como Rodar o Projeto pela Primeira Vez

Este guia foi preparado para ajudar qualquer integrante do grupo a configurar o ambiente do zero, rodar o código em **Python + FEniCSx (`dolfinx`)** e colaborar pelo **GitHub**.

---

## 1. Por que precisamos do WSL (Ubuntu) no Windows?

A biblioteca **FEniCSx (`dolfinx`)** utilizada na disciplina de Mecânica dos Fluidos **não funciona no Python comum do Windows** — ela roda apenas em **Linux**.  
Por isso, no Windows, utilizamos o **WSL (Windows Subsystem for Linux)** com o **Ubuntu**.

---

## 2. Instalando o WSL (apenas se você ainda não tiver o Ubuntu no PC)

Abra o **Prompt de Comando (`cmd`)** ou **PowerShell** do Windows e digite:

```cmd
wsl --install -d Ubuntu-24.04
```

*(Se você já tem o Ubuntu instalado, pule direto para o Passo 3).*

---

## 3. Instalando o FEniCSx e bibliotecas no Ubuntu (WSL)

No **Prompt de Comando do Windows (`cmd`)**, copie e cole o comando abaixo e aperte **Enter** (ele já instala o FEniCSx, NumPy, Matplotlib, SymPy e suporte ao Jupyter sem precisar digitar senha do Linux):

```cmd
wsl -u root bash -c "apt update && apt install -y software-properties-common && add-apt-repository -y ppa:fenics-packages/fenics && apt update && apt install -y fenicsx python3-dolfinx python3-matplotlib python3-sympy python3-ipykernel python3-nbconvert"
```

Para testar se a instalação deu certo, rode no terminal do Windows:

```cmd
wsl python3 -c "import dolfinx; print('Sucesso! DOLFINx versao:', dolfinx.__version__)"
```

Se aparecer `Sucesso! DOLFINx versao: ...`, seu ambiente já está pronto!

---

## 4. Baixando o Repositório do Grupo (Git Clone)

Escolha uma pasta no seu computador (por exemplo, na sua Área de Trabalho), abra o terminal e rode:

```cmd
git clone https://github.com/NicolauFernanda/MecFlu.git
cd MecFlu
```

---

## 5. Como Abrir e Executar no VS Code

Para que o VS Code use o Python do **Ubuntu (WSL)** (e não peça para instalar Python no Windows):

1. Abra o **VS Code**.
2. Na barra lateral esquerda, clique em **Extensões** (ícone de 4 quadradinhos) e instale:
   - **WSL** (da Microsoft)
   - **Python** (da Microsoft)
   - **Jupyter** (da Microsoft)
3. Conecte o VS Code ao Ubuntu clicando no botão azul **`><`** no **canto inferior esquerdo** da tela do VS Code e escolhendo **`Connect to WSL`** (ou **`Reopen Folder in WSL`**).
4. Abra o arquivo:
   ```text
   Tarefa_01/Projeto_1_Hagen_Poiseuille_FEniCSx_1.ipynb
   ```
5. No canto superior direito do notebook, clique em **`Select Kernel`** $\to$ **`Python Environments...`** $\to$ escolha **`Python 3 (/usr/bin/python3)`**.
   *(Se o VS Code perguntar "Install/Enable suggested extensions: Python + Jupyter", clique nessa opção para ativá-las dentro do WSL e depois escolha `Python 3 (/usr/bin/python3)`).*
6. Clique em **`Run All`** (`▷▷` no topo do notebook) para executar todas as células da primeira à última!

---

### Alternativa Rápida: Rodar tudo pelo Terminal (sem depender do botão do VS Code)

Se preferir rodar o projeto inteiro direto pelo terminal do Windows (dentro da pasta `MecFlu`), basta executar:

- **Para rodar o script Python puro e gerar os 6 gráficos em `Tarefa_01/figuras/`:**
  ```cmd
  wsl python3 Tarefa_01/executar_projeto1.py
  ```

- **Para executar e atualizar todas as células do notebook `.ipynb` automaticamente:**
  ```cmd
  wsl bash -c "cd Tarefa_01 && python3 -m nbconvert --to notebook --execute --inplace Projeto_1_Hagen_Poiseuille_FEniCSx_1.ipynb"
  ```

---

## 6. Como Colaborar no GitHub com o Grupo

Depois de aceitar o convite de colaborador no GitHub:

1. **Antes de começar a editar (sempre baixe as atualizações que os colegas fizeram):**
   ```cmd
   git pull
   ```

2. **Depois de salvar suas alterações nos arquivos:**
   ```cmd
   git add .
   git commit -m "Descreva aqui o que voce alterou"
   git push
   ```
