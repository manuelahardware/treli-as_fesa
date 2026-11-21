
import numpy as np


class ErroTrelica(Exception):
    pass


def resolver_trelica(nos, barras, apoios, cargas, E=200e9, A=0.001):
    """
    Analisa uma treliça plana pelo método da rigidez.

    Unidades:
    Coordenadas: m
    Forças: N
    Módulo de elasticidade: Pa
    Área: m²

    Apoios:
    pino: restringe X e Y
    rolete_y: restringe Y
    rolete_x: restringe X
    """

    nos = np.array(nos, dtype=float)
    n = len(nos)
    b = len(barras)

    if nos.ndim != 2 or nos.shape[1] != 2:
        raise ErroTrelica("Os nós devem possuir coordenadas X e Y.")

    if n < 2 or b < 1:
        raise ErroTrelica("Defina pelo menos dois nós e uma barra.")

    if E <= 0 or A <= 0:
        raise ErroTrelica("E e A devem ser positivos.")

    if not np.all(np.isfinite(nos)):
        raise ErroTrelica("Coordenadas inválidas.")

    K = np.zeros((2 * n, 2 * n))
    F = np.zeros(2 * n)

    comprimentos = []
    direcoes = []

    # Montagem da matriz de rigidez global
    for i, j in barras:

        if not (0 <= i < n and 0 <= j < n) or i == j:
            raise ErroTrelica("Conexão entre nós inválida.")

        dx = nos[j, 0] - nos[i, 0]
        dy = nos[j, 1] - nos[i, 1]

        L = np.hypot(dx, dy)

        if L <= 1e-12:
            raise ErroTrelica("Existe uma barra de comprimento zero.")

        c = dx / L
        s = dy / L

        ke = (E * A / L) * np.array([
            [c*c,  c*s, -c*c, -c*s],
            [c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s, c*c, c*s],
            [-c*s, -s*s, c*s, s*s]
        ])

        g = [2*i, 2*i+1, 2*j, 2*j+1]

        for a in range(4):
            for d in range(4):
                K[g[a], g[d]] += ke[a, d]

        comprimentos.append(L)
        direcoes.append((c, s))

    # Aplicação das cargas
    for no, (fx, fy) in cargas.items():

        if not 0 <= no < n:
            raise ErroTrelica(f"Nó {no+1} inexistente.")

        F[2*no] += fx
        F[2*no+1] += fy

    # Restrições dos apoios
    restritos = set()

    for no, tipo in apoios.items():

        if not 0 <= no < n:
            raise ErroTrelica(f"Nó {no+1} inexistente.")

        tipo = tipo.lower()

        if tipo == "pino":
            restritos.update([2*no, 2*no+1])

        elif tipo == "rolete_y":
            restritos.add(2*no+1)

        elif tipo == "rolete_x":
            restritos.add(2*no)

        else:
            raise ErroTrelica(f"Tipo de apoio inválido: {tipo}")

    livres = sorted(set(range(2*n)) - restritos)

    if not livres:
        raise ErroTrelica(
            "Todos os graus de liberdade estão restringidos."
        )

    Kll = K[np.ix_(livres, livres)]

    if np.linalg.matrix_rank(Kll) < len(livres):
        raise ErroTrelica(
            "Estrutura instável ou com mecanismos. "
            "Verifique barras e apoios."
        )

    # Resolução dos deslocamentos
    u = np.zeros(2*n)

    try:
        u[livres] = np.linalg.solve(Kll, F[livres])
    except np.linalg.LinAlgError:
        raise ErroTrelica("Não foi possível resolver a estrutura.")

    # Reações de apoio
    R = K @ u - F

    reacoes = []

    for dof in sorted(restritos):
        no = dof // 2
        direcao = "x" if dof % 2 == 0 else "y"

        reacoes.append((no, direcao, R[dof]))

    # Cálculo dos esforços normais e tensões
    esforcos = []
    tensoes = []
    tipos = []

    for k, (i, j) in enumerate(barras):

        c, s = direcoes[k]
        L = comprimentos[k]

        g = [2*i, 2*i+1, 2*j, 2*j+1]
        ue = u[g]

        deformacao = (
            -c*ue[0] - s*ue[1]
            + c*ue[2] + s*ue[3]
        ) / L

        N = E * A * deformacao
        sigma = N / A

        esforcos.append(N)
        tensoes.append(sigma)

        if N > 1e-6:
            tipos.append("Tração")
        elif N < -1e-6:
            tipos.append("Compressão")
        else:
            tipos.append("Nula")

    return {
        "deslocamentos": u,
        "reacoes": reacoes,
        "esforcos": np.array(esforcos),
        "tensoes": np.array(tensoes),
        "tipos": tipos,
        "matriz_rigidez": K,
        "comprimentos": comprimentos
    }


# Exemplo de utilização
if __name__ == "__main__":

    nos = [
        [0, 0],
        [4, 0],
        [2, 3]
    ]

    barras = [
        (0, 1),
        (0, 2),
        (1, 2)
    ]

    apoios = {
        0: "pino",
        1: "rolete_y"
    }

    cargas = {
        2: (0, -10000)
    }

    try:
        resultado = resolver_trelica(
            nos, barras, apoios, cargas
        )

        print("\n===== RESULTADOS DA TRELIÇA =====")

        print("\nDeslocamentos:")
        for i in range(len(nos)):
            print(
                f"Nó {i+1}: "
                f"X = {resultado['deslocamentos'][2*i]:.8e} m | "
                f"Y = {resultado['deslocamentos'][2*i+1]:.8e} m"
            )

        print("\nReações de apoio:")
        for no, direcao, valor in resultado["reacoes"]:
            print(
                f"Nó {no+1} - {direcao.upper()}: "
                f"{valor:.2f} N"
            )

        print("\nEsforços nas barras:")
        for i, (esforco, tipo, tensao) in enumerate(
            zip(
                resultado["esforcos"],
                resultado["tipos"],
                resultado["tensoes"]
            )
        ):
            print(
                f"Barra {i+1}: {tipo} | "
                f"Esforço = {esforco:.2f} N | "
                f"Tensão = {tensao:.2f} Pa"
            )

    except ErroTrelica as erro:
        print("Erro:", erro)
