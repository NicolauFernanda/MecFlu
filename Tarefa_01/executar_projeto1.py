# ============================================================
# MECÂNICA DOS FLUIDOS — PROJETO 1: ESCOAMENTO DE HAGEN-POISEUILLE
# Script Python completo para execução direta no FEniCSx (WSL)
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Permite salvar figuras mesmo sem interface gráfica aberta
import matplotlib.pyplot as plt

from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import ufl

# Garante que executa na pasta Tarefa_01 onde está o script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figuras", exist_ok=True)

print("============================================================")
print("SEÇÃO 6 — AMBIENTE COMPUTACIONAL")
print("============================================================")
print("NumPy  :", np.__version__)
print("DOLFINx:", dolfinx.__version__)
print("petsc4py:", PETSc.Sys.getVersion())
print("Ambiente carregado com sucesso!\n")

# ============================================================
# SEÇÃO 7 — PARÂMETROS FÍSICOS
# ============================================================
print("============================================================")
print("SEÇÃO 7 — PARÂMETROS FÍSICOS (SI)")
print("============================================================")
a = 1.0        # Raio interno do tubo [m]
l = 1.0        # Comprimento do tubo [m]
eta_val = 1.0  # Viscosidade dinâmica do fluido [Pa.s]
p1 = 4.0       # Pressão na entrada do tubo [Pa]
p2 = 0.0       # Pressão na saída do tubo [Pa]

G_val = (p1 - p2) / l
v_max_exato = (G_val / (4.0 * eta_val)) * (a ** 2)
Q_exato_ref = (np.pi * (a ** 4) * G_val) / (8.0 * eta_val)

print(f"Raio do tubo (a)          = {a:.2f} m")
print(f"Comprimento do tubo (l)   = {l:.2f} m")
print(f"Viscosidade (eta)         = {eta_val:.2f} Pa.s")
print(f"Diferença de pressão      = {p1 - p2:.2f} Pa")
print(f"Gradiente de pressão (G)  = {G_val:.2f} Pa/m")
print(f"Velocidade máxima (r = 0) = {v_max_exato:.4f} m/s")
print(f"Vazão analítica (Q_exato) = {Q_exato_ref:.6f} m^3/s\n")

# ============================================================
# SEÇÕES 8 a 13 — SOLUÇÃO INICIAL (N = 4, P1) E VERIFICAÇÃO
# ============================================================
print("============================================================")
print("SEÇÕES 8 A 13 — SOLUÇÃO INICIAL (N = 4, P1) E VERIFICAÇÃO")
print("============================================================")
N = 4
dominio = mesh.create_interval(MPI.COMM_WORLD, N, [0.0, a])
V = fem.functionspace(dominio, ("Lagrange", 1))

v = ufl.TrialFunction(V)
w = ufl.TestFunction(V)

x = ufl.SpatialCoordinate(dominio)
r = x[0]
eta = fem.Constant(dominio, PETSc.ScalarType(eta_val))
G = fem.Constant(dominio, PETSc.ScalarType(G_val))

a_form = eta * r * ufl.inner(ufl.grad(v), ufl.grad(w)) * ufl.dx
ell = G * r * w * ufl.dx

fdim = dominio.topology.dim - 1
parede = mesh.locate_entities_boundary(dominio, fdim, lambda x: np.isclose(x[0], a))
dofs_parede = fem.locate_dofs_topological(V, fdim, parede)
bc_parede = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_parede, V)

problema = LinearProblem(
    a_form,
    ell,
    bcs=[bc_parede],
    petsc_options_prefix="hagen_poiseuille_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
vh = problema.solve()

def v_exato_func(r_pts, a_val=a, G_v=G_val, eta_v=eta_val):
    return (G_v / (4.0 * eta_v)) * (a_val**2 - r_pts**2)

r_dofs = V.tabulate_dof_coordinates()[:, 0]
v_num = vh.x.array.real
ordem = np.argsort(r_dofs)
r_dofs = r_dofs[ordem]
v_num = v_num[ordem]

v_ex_dofs = v_exato_func(r_dofs)
erro_abs_nodal = np.abs(v_num - v_ex_dofs)

print(f"{'Nó':>4} | {'r [m]':>10} | {'v_h (MEF) [m/s]':>16} | {'v_exato [m/s]':>16} | {'Erro Absoluto':>14}")
print("-" * 72)
for i in range(len(r_dofs)):
    print(f"{i:4d} | {r_dofs[i]:10.4f} | {v_num[i]:16.6f} | {v_ex_dofs[i]:16.6f} | {erro_abs_nodal[i]:14.6e}")

erro_max_nodal = np.max(erro_abs_nodal)
x_ufl = ufl.SpatialCoordinate(dominio)
v_exata_ufl = (G_val / (4.0 * eta_val)) * (a**2 - x_ufl[0]**2)
form_erro_L2 = fem.form((vh - v_exata_ufl)**2 * ufl.dx)
erro_L2 = np.sqrt(dominio.comm.allreduce(fem.assemble_scalar(form_erro_L2), op=MPI.SUM))

print("-" * 72)
print(f"Erro máximo nos nós (norma inf): {erro_max_nodal:.6e} m/s")
print(f"Erro global na norma L2        : {erro_L2:.6e}\n")

r_fino = np.linspace(0.0, a, 300)
v_fino = v_exato_func(r_fino)

plt.figure(figsize=(7, 5))
plt.plot(r_fino, v_fino, "k-", linewidth=2.2, label="Solução analítica exata")
plt.plot(r_dofs, v_num, "ro--", linewidth=1.5, markersize=7, label=f"MEF P1 (N = {N})")
plt.xlabel(r"Raio $r$ [m]")
plt.ylabel(r"Velocidade axial $v(r)$ [m/s]")
plt.title(f"Verificação do Perfil de Velocidade de Hagen-Poiseuille (N = {N})")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("figuras/fig1_verificacao_N4.png", dpi=300)
plt.close()

# ============================================================
# SEÇÃO 14 — ESTUDO DE MALHA COM P1 (N = 2, 4, 8, 16, 32)
# ============================================================
print("============================================================")
print("SEÇÃO 14 — ESTUDO DE MALHA COM P1 (N = 2, 4, 8, 16, 32)")
print("============================================================")

def resolver_hagen_poiseuille(N_elem, grau=1, a_val=a, G_v=G_val, eta_v=eta_val):
    dom = mesh.create_interval(MPI.COMM_WORLD, N_elem, [0.0, a_val])
    V_esp = fem.functionspace(dom, ("Lagrange", grau))
    v_trial = ufl.TrialFunction(V_esp)
    w_test = ufl.TestFunction(V_esp)
    x_coord = ufl.SpatialCoordinate(dom)
    r_coord = x_coord[0]
    eta_c = fem.Constant(dom, PETSc.ScalarType(eta_v))
    G_c = fem.Constant(dom, PETSc.ScalarType(G_v))
    a_ufl = eta_c * r_coord * ufl.inner(ufl.grad(v_trial), ufl.grad(w_test)) * ufl.dx
    L_ufl = G_c * r_coord * w_test * ufl.dx
    fdim_local = dom.topology.dim - 1
    facets_parede = mesh.locate_entities_boundary(dom, fdim_local, lambda x: np.isclose(x[0], a_val))
    dofs_p = fem.locate_dofs_topological(V_esp, fdim_local, facets_parede)
    bc_p = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_p, V_esp)
    prob = LinearProblem(
        a_ufl, L_ufl, bcs=[bc_p],
        petsc_options_prefix=f"hp_P{grau}_N{N_elem}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    vh_sol = prob.solve()
    return dom, V_esp, vh_sol

def calcular_erros(dom, V_esp, vh_sol, a_val=a, G_v=G_val, eta_v=eta_val):
    coords_r = V_esp.tabulate_dof_coordinates()[:, 0]
    valores_v = vh_sol.x.array.real
    valores_exatos = (G_v / (4.0 * eta_v)) * (a_val**2 - coords_r**2)
    erro_nodal_max = np.max(np.abs(valores_v - valores_exatos))
    x_coord = ufl.SpatialCoordinate(dom)
    v_ex_ufl = (G_v / (4.0 * eta_v)) * (a_val**2 - x_coord[0]**2)
    erro_form = fem.form((vh_sol - v_ex_ufl)**2 * ufl.dx)
    erro_local = fem.assemble_scalar(erro_form)
    erro_L2_val = np.sqrt(dom.comm.allreduce(erro_local, op=MPI.SUM))
    return erro_nodal_max, erro_L2_val

Ns = [2, 4, 8, 16, 32]
resultados_P1 = []
solucoes_P1 = {}

for N_atual in Ns:
    dom_N, V_N, vh_N = resolver_hagen_poiseuille(N_atual, grau=1)
    solucoes_P1[N_atual] = (dom_N, V_N, vh_N)
    h_atual = a / N_atual
    ndofs_atual = V_N.dofmap.index_map.size_global
    erro_max, erro_l2 = calcular_erros(dom_N, V_N, vh_N)
    resultados_P1.append((N_atual, h_atual, ndofs_atual, erro_max, erro_l2))

print(f"{'N':>4} | {'h [m]':>9} | {'DOFs':>6} | {'Erro Máx Nodal':>15} | {'Erro L2':>15} | {'Fator Queda L2':>14}")
print("-" * 78)
for idx, (N_i, h_i, ndof_i, emax_i, el2_i) in enumerate(resultados_P1):
    if idx == 0:
        fator_str = "      ---     "
    else:
        el2_anterior = resultados_P1[idx - 1][4]
        fator = el2_anterior / el2_i
        fator_str = f"{fator:14.2f}x"
    print(f"{N_i:4d} | {h_i:9.5f} | {ndof_i:6d} | {emax_i:15.6e} | {el2_i:15.6e} | {fator_str}")
print()

plt.figure(figsize=(8, 5))
plt.plot(r_fino, v_fino, "k-", linewidth=2.5, label="Solução analítica exata")
for N_atual in Ns:
    dom_N, V_N, vh_N = solucoes_P1[N_atual]
    coords = V_N.tabulate_dof_coordinates()[:, 0]
    vals = vh_N.x.array.real
    ordem = np.argsort(coords)
    plt.plot(coords[ordem], vals[ordem], "o--", markersize=5, label=f"MEF P1 — N={N_atual}")
plt.xlabel(r"Raio $r$ [m]")
plt.ylabel(r"Velocidade axial $v(r)$ [m/s]")
plt.title("Convergência do Perfil de Velocidade com o Refinamento da Malha (P1)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("figuras/fig2_perfis_malha_P1.png", dpi=300)
plt.close()

hs = np.array([item[1] for item in resultados_P1])
erros_max_arr = np.array([item[3] for item in resultados_P1])
erros_l2_arr = np.array([item[4] for item in resultados_P1])
ref_h2 = erros_l2_arr[0] * (hs / hs[0])**2

plt.figure(figsize=(7, 5))
plt.loglog(hs, erros_l2_arr, "bo-", linewidth=2, markersize=7, label=r"Erro na norma $L^2$")
plt.loglog(hs, erros_max_arr, "rs--", linewidth=1.8, markersize=6, label="Erro máximo nodal")
plt.loglog(hs, ref_h2, "k:", linewidth=1.8, label=r"Referência teórica $O(h^2)$")
plt.xlabel(r"Tamanho do elemento $h = a/N$ [m]")
plt.ylabel("Erro")
plt.title(r"Taxa de Convergência com Elementos Lineares $P_1$")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("figuras/fig3_convergencia_P1.png", dpi=300)
plt.close()

# ============================================================
# SEÇÃO 15 — COMPARAÇÃO P1 x P2
# ============================================================
print("============================================================")
print("SEÇÃO 15 — COMPARAÇÃO P1 x P2")
print("============================================================")
Ns_comp = [2, 4, 8, 16, 32]
resultados_P2 = []
solucoes_P2 = {}

for N_atual in Ns_comp:
    dom_2, V_2, vh_2 = resolver_hagen_poiseuille(N_atual, grau=2)
    solucoes_P2[N_atual] = (dom_2, V_2, vh_2)
    h_atual = a / N_atual
    ndofs_2 = V_2.dofmap.index_map.size_global
    erro_max_2, erro_l2_2 = calcular_erros(dom_2, V_2, vh_2)
    resultados_P2.append((N_atual, h_atual, ndofs_2, erro_max_2, erro_l2_2))

print(f"{'N':>4} | {'DOFs P1':>8} | {'Erro L2 (P1)':>14} | {'DOFs P2':>8} | {'Erro L2 (P2)':>14} | {'Erro Máx Nodal (P2)':>20}")
print("-" * 82)
for i in range(len(Ns_comp)):
    N_i, _, dof_p1, _, l2_p1 = resultados_P1[i]
    _, _, dof_p2, max_p2, l2_p2 = resultados_P2[i]
    print(f"{N_i:4d} | {dof_p1:8d} | {l2_p1:14.6e} | {dof_p2:8d} | {l2_p2:14.6e} | {max_p2:20.6e}")
print()

N_exemplo = 2
_, V_p1_ex, vh_p1_ex = solucoes_P1[N_exemplo]
_, V_p2_ex, vh_p2_ex = solucoes_P2[N_exemplo]

r_p1 = V_p1_ex.tabulate_dof_coordinates()[:, 0]
v_p1 = vh_p1_ex.x.array.real
ord_p1 = np.argsort(r_p1)
r_p1, v_p1 = r_p1[ord_p1], v_p1[ord_p1]

r_p2 = V_p2_ex.tabulate_dof_coordinates()[:, 0]
v_p2 = vh_p2_ex.x.array.real
ord_p2 = np.argsort(r_p2)
r_p2, v_p2 = r_p2[ord_p2], v_p2[ord_p2]

plt.figure(figsize=(7, 5))
plt.plot(r_fino, v_fino, "k-", linewidth=2.5, label="Solução analítica exata")
plt.plot(r_p1, v_p1, "ro--", linewidth=1.8, markersize=7, label=f"MEF P1 (N = {N_exemplo}, 3 DOFs)")
plt.plot(r_p2, v_p2, "bs", markersize=7, label=f"Nós do MEF P2 (N = {N_exemplo}, 5 DOFs)")
plt.xlabel(r"Raio $r$ [m]")
plt.ylabel(r"Velocidade axial $v(r)$ [m/s]")
plt.title(f"Comparação P1 vs P2 em Malha Grossa (N = {N_exemplo})")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("figuras/fig4_comparacao_P1_P2.png", dpi=300)
plt.close()

# ============================================================
# SEÇÃO 16 — VAZÃO NUMÉRICA
# ============================================================
print("============================================================")
print("SEÇÃO 16 — VAZÃO NUMÉRICA (Q_h)")
print("============================================================")

def calcular_vazao(dom, vh_sol):
    x_coord = ufl.SpatialCoordinate(dom)
    r_coord = x_coord[0]
    forma_vazao = fem.form(2.0 * np.pi * vh_sol * r_coord * ufl.dx)
    vazao_local = fem.assemble_scalar(forma_vazao)
    vazao_global = dom.comm.allreduce(vazao_local, op=MPI.SUM)
    return vazao_global

Q_exato = (np.pi * (a ** 4) * G_val) / (8.0 * eta_val)
print(f"Vazão analítica exata (Q_exato) = {Q_exato:.8f} m^3/s\n")
print(f"{'N':>4} | {'Q_h (P1) [m^3/s]':>18} | {'Erro Rel. P1 [%]':>17} | {'Q_h (P2) [m^3/s]':>18} | {'Erro Rel. P2 [%]':>17}")
print("-" * 84)

vazoes_P1 = []
for N_atual in Ns:
    dom_1, _, vh_1 = solucoes_P1[N_atual]
    dom_2, _, vh_2 = solucoes_P2[N_atual]
    Q_h_1 = calcular_vazao(dom_1, vh_1)
    Q_h_2 = calcular_vazao(dom_2, vh_2)
    erro_rel_1 = abs(Q_h_1 - Q_exato) / Q_exato * 100.0
    erro_rel_2 = abs(Q_h_2 - Q_exato) / Q_exato * 100.0
    vazoes_P1.append(Q_h_1)
    print(f"{N_atual:4d} | {Q_h_1:18.8f} | {erro_rel_1:16.4f}% | {Q_h_2:18.8f} | {erro_rel_2:16.4e}%")
print()

plt.figure(figsize=(7, 4.5))
plt.axhline(Q_exato, color="k", linestyle="-", linewidth=2, label=f"Q_exato = {Q_exato:.4f} m³/s")
plt.plot(Ns, vazoes_P1, "ro--", linewidth=1.8, markersize=7, label="Q_h numérico (MEF P1)")
plt.xlabel("Número de elementos (N)")
plt.ylabel(r"Vazão volumétrica $Q$ [m$^3$/s]")
plt.title("Convergência da Vazão Numérica com o Refinamento da Malha")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("figuras/fig5_convergencia_vazao.png", dpi=300)
plt.close()

# ============================================================
# SEÇÃO 17 — INVESTIGAÇÃO FÍSICA (+10% NO RAIO)
# ============================================================
print("============================================================")
print("SEÇÃO 17 — EXPERIMENTO OBRIGATÓRIO (+10% NO RAIO)")
print("============================================================")
a_novo = 1.10 * a
N_exp = 32

dom_orig, _, vh_orig = resolver_hagen_poiseuille(N_exp, grau=2, a_val=a, G_v=G_val, eta_v=eta_val)
dom_novo, _, vh_novo = resolver_hagen_poiseuille(N_exp, grau=2, a_val=a_novo, G_v=G_val, eta_v=eta_val)

Q_num_orig = calcular_vazao(dom_orig, vh_orig)
Q_num_novo = calcular_vazao(dom_novo, vh_novo)

variacao_teorica_pct = ((1.10 ** 4) - 1.0) * 100.0
variacao_numerica_pct = ((Q_num_novo - Q_num_orig) / Q_num_orig) * 100.0

print(f"Raio original (a)              = {a:.4f} m  -> Q_h = {Q_num_orig:.6f} m^3/s")
print(f"Raio aumentado (a_novo = 1.1a) = {a_novo:.4f} m  -> Q_h = {Q_num_novo:.6f} m^3/s")
print(f"Razão numérica Q_novo / Q_orig = {Q_num_novo / Q_num_orig:.6f} (Teórico: 1.10^4 = {1.10**4:.6f})")
print(f"Aumento percentual teórico     = +{variacao_teorica_pct:.2f}%")
print(f"Aumento percentual numérico    = +{variacao_numerica_pct:.2f}%\n")

dp_lista = [2.0, 4.0, 6.0, 8.0, 10.0]
Q_dp_num = [calcular_vazao(*resolver_hagen_poiseuille(16, grau=2, a_val=a, G_v=dp/l, eta_v=eta_val)[::2]) for dp in dp_lista]

eta_lista = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
Q_eta_num = [calcular_vazao(*resolver_hagen_poiseuille(16, grau=2, a_val=a, G_v=G_val, eta_v=et)[::2]) for et in eta_lista]

a_lista = [0.6, 0.8, 1.0, 1.1, 1.2, 1.4]
Q_a_num = [calcular_vazao(*resolver_hagen_poiseuille(16, grau=2, a_val=at, G_v=G_val, eta_v=eta_val)[::2]) for at in a_lista]

dp_fino = np.linspace(1.5, 10.5, 100)
eta_fino = np.linspace(0.45, 4.2, 100)
a_fino_param = np.linspace(0.55, 1.45, 100)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
axes[0].plot(dp_fino, (np.pi * a**4 * (dp_fino / l)) / (8 * eta_val), "k-", label=r"Teórico ($Q \propto \Delta p$)")
axes[0].plot(dp_lista, Q_dp_num, "ro", markersize=7, label="MEF FEniCSx")
axes[0].set_xlabel(r"Diferença de pressão $\Delta p = p_1 - p_2$ [Pa]")
axes[0].set_ylabel(r"Vazão $Q$ [m$^3$/s]")
axes[0].set_title(r"Dependência Linear com $\Delta p$")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(eta_fino, (np.pi * a**4 * G_val) / (8 * eta_fino), "k-", label=r"Teórico ($Q \propto 1/\eta$)")
axes[1].plot(eta_lista, Q_eta_num, "bs", markersize=7, label="MEF FEniCSx")
axes[1].set_xlabel(r"Viscosidade $\eta$ [Pa$\cdot$s]")
axes[1].set_ylabel(r"Vazão $Q$ [m$^3$/s]")
axes[1].set_title(r"Dependência Inversa com $\eta$")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

axes[2].plot(a_fino_param, (np.pi * (a_fino_param**4) * G_val) / (8 * eta_val), "k-", label=r"Teórico ($Q \propto a^4$)")
axes[2].plot(a_lista, Q_a_num, "g^", markersize=8, label="MEF FEniCSx")
axes[2].set_xlabel(r"Raio do tubo $a$ [m]")
axes[2].set_ylabel(r"Vazão $Q$ [m$^3$/s]")
axes[2].set_title(r"Dependência de 4ª Potência com $a$")
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.savefig("figuras/fig6_investigacao_fisica.png", dpi=300)
plt.close()

print("Todas as 6 figuras foram salvas na pasta 'Tarefa_01/figuras/'!")
