#!/usr/bin/env python3
"""
Chevrolet Monza 1.8 1988 - Gerador de Arquivo para Impressão 3D
Escala 1:24
Baseado nos desenhos técnicos: vistas frontal, lateral, traseira e teto.

Dimensões reais do veículo:
  Comprimento: 4395 mm
  Largura:     1660 mm
  Altura:      1248 mm
  Entre-eixos: 2518 mm
"""

import numpy as np
from skimage import measure
import struct
import math
import os
import sys

# ====================================================
# DIMENSÕES NA ESCALA 1:24  (todas em mm)
# ====================================================

SCALE = 24

L   = 4395 / SCALE     # 183.1 mm  comprimento total
W   = 1660 / SCALE     # 69.2  mm  largura total
H   = 1248 / SCALE     # 52.0  mm  altura total
WB  = 2518 / SCALE     # 104.9 mm  entre-eixos
FO  =  887 / SCALE     # 36.9  mm  balanço dianteiro
RO  =  990 / SCALE     # 41.3  mm  balanço traseiro

HW  = W / 2            # 34.6  mm  meia-largura do corpo
HC  = (W / 2) - 3.8    # 30.8  mm  meia-largura da cabine
HB  = 27.5             # mm    altura da beltline / ombreiro
HH  = 19.8             # mm    altura do capô

# Posições chave no eixo X (comprimento)
X_PC_D   = 0.0         # para-choque dianteiro (frente)
X_CAPO   = 8.0         # início do capô
X_WS_INF = FO - 4.0   # base inferior do para-brisa
X_WS_SUP = FO + 13.0  # topo do para-brisa / início do teto
X_TETO   = X_WS_SUP + (WB - 26.0)   # fim do teto
X_LUN    = X_TETO + 14.0             # fim da luneta traseira
X_MALA   = X_LUN                     # início do porta-malas
X_PC_T   = L - 8.0    # início do para-choque traseiro
X_PC_FIM = L          # fim do carro

# Rodas
RR  = 580 / (2 * SCALE)   # 12.08 mm raio externo (pneu)
RW  = 195 / SCALE         #  8.13 mm largura do pneu
RA  = 370 / (2 * SCALE)   #  7.71 mm raio do aro
ZE  = RR                   # altura do eixo (roda toca o chão)
X_EIXO_D = FO              # X do eixo dianteiro
X_EIXO_T = FO + WB        # X do eixo traseiro
Y_RODA   = HW + RW/2 + 0.8 # posição Y (lateral) do centro da roda

# ====================================================
# FUNÇÕES DE PERFIL
# ====================================================

def z_profile(x):
    """
    Retorna a altura máxima do veículo em cada posição X.
    Vista lateral (perfil de silhueta).
    """
    z = np.zeros_like(x, dtype=np.float32)

    # Para-choque dianteiro (nariz)
    m = (x >= X_PC_D) & (x < X_CAPO)
    z[m] = np.interp(x[m], [X_PC_D, X_CAPO], [12.0, HH])

    # Capô (sobe levemente de frente para trás)
    m = (x >= X_CAPO) & (x < X_WS_INF)
    z[m] = np.interp(x[m], [X_CAPO, X_WS_INF], [HH, HH + 1.0])

    # Para-brisa (subida íngreme)
    m = (x >= X_WS_INF) & (x < X_WS_SUP)
    z[m] = np.interp(x[m], [X_WS_INF, X_WS_SUP], [HH + 1.0, H])

    # Teto plano
    m = (x >= X_WS_SUP) & (x < X_TETO)
    z[m] = H

    # Luneta traseira (descida)
    m = (x >= X_TETO) & (x < X_LUN)
    z[m] = np.interp(x[m], [X_TETO, X_LUN], [H, HB])

    # Tampa do porta-malas
    m = (x >= X_MALA) & (x < X_PC_T)
    z[m] = HB

    # Para-choque traseiro (desce)
    m = (x >= X_PC_T) & (x <= X_PC_FIM)
    z[m] = np.interp(x[m], [X_PC_T, X_PC_FIM], [HB, 12.0])

    return z


def y_profile(x, z):
    """
    Retorna a meia-largura máxima em cada posição (x, z).
    Abaixo da beltline = largura total; acima = largura da cabine.
    """
    y = np.full_like(x, HW, dtype=np.float32)
    above = z > HB

    # Dentro do teto: largura da cabine
    in_roof = (x >= X_WS_SUP) & (x <= X_TETO) & above
    y[in_roof] = HC

    # Transição no para-brisa (X_WS_INF → X_WS_SUP)
    in_ws = (x >= X_WS_INF) & (x < X_WS_SUP) & above
    t_ws = np.clip((x[in_ws] - X_WS_INF) / (X_WS_SUP - X_WS_INF), 0, 1)
    y[in_ws] = HW + (HC - HW) * t_ws

    # Transição na luneta (X_TETO → X_LUN)
    in_rw = (x > X_TETO) & (x <= X_LUN) & above
    t_rw = np.clip((x[in_rw] - X_TETO) / (X_LUN - X_TETO), 0, 1)
    y[in_rw] = HC + (HW - HC) * t_rw

    return y


# ====================================================
# CAMPO ESCALAR DO VEÍCULO
# ====================================================

def car_field(X, Y, Z):
    """
    Campo escalar: < 0 = interior do veículo, > 0 = exterior.
    Utilizado pelo algoritmo Marching Cubes.
    """
    z_top = z_profile(X.ravel()).reshape(X.shape)
    y_max = y_profile(X.ravel(), Z.ravel()).reshape(X.shape)

    f = np.maximum.reduce([
        Z - z_top,            # acima do perfil lateral
        -Z,                   # abaixo do chão
        np.abs(Y) - y_max,   # fora da largura
        -X,                   # antes da frente
        X - L,                # depois da traseira
    ])
    return f


def car_with_arches(X, Y, Z):
    """Campo do veículo com passagens de roda."""
    body = car_field(X, Y, Z)

    r_arch = RR + 1.8  # arco ligeiramente maior que a roda
    arch_d = np.sqrt((X - X_EIXO_D)**2 + (Z - ZE)**2) - r_arch
    arch_t = np.sqrt((X - X_EIXO_T)**2 + (Z - ZE)**2) - r_arch

    result = np.maximum(body, -arch_d)
    result = np.maximum(result, -arch_t)
    return result


# ====================================================
# UTILITÁRIOS STL
# ====================================================

def save_stl(filename, verts, faces):
    """Salva malha triangular como arquivo STL binário."""
    n = len(faces)
    with open(filename, 'wb') as f:
        header = b'Monza 1.8 1988 | Escala 1:24 | Impressao 3D'[:80].ljust(80, b'\x00')
        f.write(header)
        f.write(struct.pack('<I', n))
        for tri in faces:
            v0 = verts[tri[0]].astype(np.float32)
            v1 = verts[tri[1]].astype(np.float32)
            v2 = verts[tri[2]].astype(np.float32)
            e1, e2 = v1 - v0, v2 - v0
            nv = np.cross(e1, e2)
            nm = np.linalg.norm(nv)
            if nm > 1e-12:
                nv /= nm
            else:
                nv = np.array([0.0, 0.0, 1.0])
            f.write(struct.pack('<fff', *nv.astype(np.float32)))
            f.write(struct.pack('<fff', *v0))
            f.write(struct.pack('<fff', *v1))
            f.write(struct.pack('<fff', *v2))
            f.write(struct.pack('<H', 0))
    sz = os.path.getsize(filename) / 1024
    print(f"  -> {os.path.basename(filename)}  ({n:,} triângulos, {sz:.0f} KB)")


def cylinder_y_mesh(cx, cy, cz, r, hl, n_sides=40):
    """
    Gera vértices e faces de um cilindro orientado no eixo Y,
    centrado em (cx, cy, cz), raio r, meia-largura hl.
    """
    angles = np.linspace(0, 2 * math.pi, n_sides, endpoint=False)
    # Anéis esquerdo e direito
    pts_L = np.column_stack([
        cx + r * np.cos(angles),
        np.full(n_sides, cy - hl),
        cz + r * np.sin(angles),
    ])
    pts_R = np.column_stack([
        cx + r * np.cos(angles),
        np.full(n_sides, cy + hl),
        cz + r * np.sin(angles),
    ])
    cL = np.array([[cx, cy - hl, cz]])
    cR = np.array([[cx, cy + hl, cz]])

    verts = np.vstack([pts_L, pts_R, cL, cR])  # 0..n-1, n..2n-1, 2n, 2n+1
    n = n_sides
    faces = []

    for i in range(n):
        j = (i + 1) % n
        # Barrel (lateral)
        faces.append([i, j, j + n])
        faces.append([i, j + n, i + n])
        # Tampa esquerda (normal -Y)
        faces.append([2 * n, j, i])
        # Tampa direita (normal +Y)
        faces.append([2 * n + 1, i + n, j + n])

    return verts, np.array(faces, dtype=np.int32)


def build_wheels(filename):
    """Gera as 4 rodas do Monza em um único arquivo STL."""
    all_v = []
    all_f = []
    offset = 0

    wheel_positions = [
        (X_EIXO_D, -Y_RODA, ZE),   # dianteira esquerda
        (X_EIXO_D,  Y_RODA, ZE),   # dianteira direita
        (X_EIXO_T, -Y_RODA, ZE),   # traseira esquerda
        (X_EIXO_T,  Y_RODA, ZE),   # traseira direita
    ]

    for (wx, wy, wz) in wheel_positions:
        hw = RW / 2

        # Pneu externo (cilindro sólido)
        v, f = cylinder_y_mesh(wx, wy, wz, RR, hw, n_sides=48)
        all_v.append(v)
        all_f.append(f + offset)
        offset += len(v)

        # Aro central (cilindro menor, em destaque)
        v, f = cylinder_y_mesh(wx, wy, wz, RA, hw + 0.5, n_sides=48)
        all_v.append(v)
        all_f.append(f + offset)
        offset += len(v)

    verts = np.vstack(all_v)
    faces = np.vstack(all_f)
    save_stl(filename, verts, faces)


# ====================================================
# PROGRAMA PRINCIPAL
# ====================================================

def main():
    output_dir = "/home/user/FOLHA-DE-PAGAMENTO"
    RES    = 0.6   # mm por voxel (resolução da grade)
    MARGIN = 6.0   # margem extra em mm

    print("=" * 65)
    print("  Chevrolet Monza 1.8 1988 — Gerador STL para Impressão 3D")
    print(f"  Escala: 1:{SCALE}")
    print(f"  Dimensões do modelo: {L:.1f} x {W:.1f} x {H:.1f} mm")
    print("=" * 65)

    # ------ CARROCERIA ------
    print("\n[1/4] Criando grade voxel...")
    x_arr = np.arange(-MARGIN, L + MARGIN, RES, dtype=np.float32)
    y_arr = np.arange(-HW - MARGIN, HW + MARGIN, RES, dtype=np.float32)
    z_arr = np.arange(-MARGIN, H + MARGIN + 2, RES, dtype=np.float32)

    nx, ny, nz = len(x_arr), len(y_arr), len(z_arr)
    total = nx * ny * nz
    print(f"      {nx} x {ny} x {nz} = {total:,} voxels  ({total*4/1e6:.1f} MB)")

    X, Y, Z = np.meshgrid(x_arr, y_arr, z_arr, indexing='ij')

    print("[2/4] Calculando campo escalar do veículo...")
    field = car_with_arches(X, Y, Z)
    del X, Y, Z  # libera memória

    print("[3/4] Extraindo malha com Marching Cubes...")
    verts, faces, normals, _ = measure.marching_cubes(
        field, level=0.0, spacing=(RES, RES, RES)
    )
    # Ajustar posição real (o meshgrid começa em x_arr[0], etc.)
    verts[:, 0] += x_arr[0]
    verts[:, 1] += y_arr[0]
    verts[:, 2] += z_arr[0]

    body_file = os.path.join(output_dir, "monza_1988_carroceria_1-24.stl")
    print("[4/4] Salvando carroceria STL...")
    save_stl(body_file, verts, faces)

    # ------ RODAS ------
    print("\n[+] Gerando rodas (4 peças)...")
    wheels_file = os.path.join(output_dir, "monza_1988_rodas_1-24.stl")
    build_wheels(wheels_file)

    # ------ EIXO ------
    print("[+] Gerando eixos...")
    axle_v, axle_f = [], []
    offset = 0
    for ax, az in [(X_EIXO_D, ZE), (X_EIXO_T, ZE)]:
        v, f = cylinder_y_mesh(ax, 0, az, 1.5, Y_RODA + RW/2, n_sides=16)
        axle_v.append(v)
        axle_f.append(f + offset)
        offset += len(v)
    axle_file = os.path.join(output_dir, "monza_1988_eixos_1-24.stl")
    save_stl(axle_file, np.vstack(axle_v), np.vstack(axle_f))

    # ------ SUMÁRIO ------
    print("\n" + "=" * 65)
    print("  ARQUIVOS GERADOS:")
    for f in [body_file, wheels_file, axle_file]:
        name = os.path.basename(f)
        sz = os.path.getsize(f) / 1024
        print(f"    {name:<42} {sz:>7.0f} KB")
    print("=" * 65)
    print("""
  CONFIGURAÇÕES RECOMENDADAS PARA IMPRESSÃO 3D:
    Altura de camada   : 0.1 – 0.2 mm
    Preenchimento      : 15 – 20 %
    Suportes           : Sim (para-choques e espelhos)
    Material           : PLA ou PETG
    Temperatura PLA    : 200 – 210 °C
    Temperatura PETG   : 230 – 240 °C
    Velocidade         : 40 – 60 mm/s
    Cama aquecida      : 60 °C (PLA) / 70 – 80 °C (PETG)

  MONTAGEM:
    1. Imprima a carroceria com suportes
    2. Imprima as 4 rodas separadamente
    3. Imprima os 2 eixos
    4. Encaixe os eixos nas passagens de roda
    5. Cole as rodas nos eixos (cola cianoacrilato)
    """)


if __name__ == "__main__":
    main()
