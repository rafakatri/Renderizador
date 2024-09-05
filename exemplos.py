#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Carregador de exemplos X3D.

Desenvolvido por: Luciano Soares <lpsoares@insper.edu.br>
Disciplina: Computação Gráfica
Data: 17 de Agosto de 2021
"""

import sys
import subprocess
import signal
import time

DIR = "docs/exemplos/"

# List para controlar subprocesses
subprocesses = []

def signal_handler(sig, frame):
    print("Terminating subprocesses...")
    for proc in subprocesses:
        proc.terminate()
    sys.exit(0)

# Registrando sinal para SIGINT
signal.signal(signal.SIGINT, signal_handler)

TESTE = []

# Exemplos 2D
# Rasterização
TESTE.append(["pontos", "-i", DIR+"2D/pontos/pontos.x3d", "-w", "30", "-h", "20", "-p"])
TESTE.append(["linhas", "-i", DIR+"2D/linhas/linhas.x3d", "-w", "30", "-h", "20", "-p"])
TESTE.append(["octogono", "-i", DIR+"2D/linhas/octogono.x3d", "-w", "30", "-h", "20", "-p"])
TESTE.append(["linhas_fora", "-i", DIR+"2D/linhas/linhas_fora.x3d", "-w", "30", "-h", "20", "-p"])
TESTE.append(["var_lin", "-i", DIR+"2D/linhas/varias_linhas.x3d", "-w", "600", "-h", "400", "-p"])
TESTE.append(["circulo", "-i", DIR+"2D/linhas/circulo.x3d", "-w", "30", "-h", "20", "-p"])
TESTE.append(["tri_2D", "-i", DIR+"2D/triangulos/triangulos.x3d", "-w", "30", "-h", "20", "-p"])
TESTE.append(["helice", "-i", DIR+"2D/triangulos/helice.x3d", "-w", "30", "-h", "20", "-p"])
TESTE.append(["tri_alta", "-i", DIR+"2D/triangulos/triangulos_alta.x3d", "-w", "600", "-h", "400", "-p"])

# Exemplos 3D
# Visualização 3D sem shading
TESTE.append(["um_tri", "-i", DIR+"3D/triangulos/um_triangulo.x3d", "-w", "300", "-h", "200", "-p"])
TESTE.append(["var_tri", "-i", DIR+"3D/triangulos/varios_triangulos.x3d", "-w", "300", "-h", "200", "-p"])
TESTE.append(["zoom", "-i", DIR+"3D/triangulos/zoom.x3d", "-w", "300", "-h", "200", "-p"])

# Geometrias 3D
TESTE.append(["tira_tri", "-i", DIR+"3D/triangulos/tiratrig.x3d", "-w", "300", "-h", "200", "-p"])
TESTE.append(["letras", "-i", DIR+"3D/cores/letras.x3d", "-w", "300", "-h", "200", "-p"])
TESTE.append(["avatar", "-i", DIR+"3D/avatar/avatar.x3d", "-w", "300", "-h", "200", "-p"])

# Z-buffer e Transparência
TESTE.append(["retang", "-i", DIR+"3D/retangulos/retangulos.x3d", "-w", "300", "-h", "200", "-p"])
TESTE.append(["transp", "-i", DIR+"3D/transparencia/transparencia.x3d", "-w", "300", "-h", "200", "-p"])

# Interpolações de Cores
TESTE.append(["cores", "-i", DIR+"3D/cores/cores.x3d", "-w", "300", "-h", "200", "-p"])

# Texturas
TESTE.append(["textura", "-i", DIR+"3D/texturas/textura.x3d", "-w", "300", "-h", "200", "-p"])
TESTE.append(["texturas", "-i", DIR+"3D/texturas/texturas.x3d", "-w", "300", "-h", "200", "-p"])

# Primitivas 3D
TESTE.append(["primitivas", "-i", DIR+"3D/box/primitivas.x3d", "-w", "300", "-h", "200", "-p"])

# Iluminação
TESTE.append(["tri_3D", "-i", DIR+"3D/triangulos/triang3d.x3d", "-w", "300", "-h", "200", "-p"])
TESTE.append(["caixas", "-i", DIR+"3D/box/box.x3d", "-w", "300", "-h", "200", "-p"])
TESTE.append(["esferas", "-i", DIR+"3D/iluminacao/esferas.x3d", "-w", "180", "-h", "120", "-p"])

# Animações
TESTE.append(["onda", "-i", DIR+"3D/animacoes/onda.x3d", "-w", "300", "-h", "200"])
TESTE.append(["piramide", "-i", DIR+"3D/animacoes/piramide.x3d", "-w", "300", "-h", "200"])

# Novos
TESTE.append(["leques", "-i", DIR+"3D/cores/leques.x3d", "-w", "480", "-h", "320", "-p"])
TESTE.append(["flechas", "-i", DIR+"3D/cores/flechas.x3d", "-w", "480", "-h", "320", "-p"])

# Lista os exemplos registrados (em 3 colunas)
colunas = 4
t = -(len(TESTE)//-colunas)
for i in range(t):
    for j in range(colunas):
        d = i+j*t
        if d < len(TESTE):
            print("{0:2} : {1:15}".format(d, TESTE[d][0]), end="")
    print()

# Se um parâmetro fornecido, usar ele como escolha do exemplo
outra_opcoes = []  # caso usuario passe opções que deverão ser repassadas, por exemplo: --quiet
if len(sys.argv) > 1:
    escolhas = sys.argv[1:]
else:
    escolhas = [input("Escolha o exemplo: ")]

# Verifica se a escolha do exemplo foi por faixa, índice ou argumento da lista
opcoes = []
for escolha in escolhas:
    if ".." in escolha:
        try:
            faixa = escolha.split("..")
            for i in range(int(faixa[0]), int(faixa[1])+1):
                opcoes.append(TESTE[i])
        except:
            sys.exit("Opção inválida!")
    elif escolha.isnumeric():
        numero = int(escolha)
        if 0 <= numero < len(TESTE):
            opcoes.append(TESTE[int(escolha)])
        else:
            sys.exit("Opção inválida!")
    else:
        texto = [element for element in TESTE if element[0] == escolha]
        if len(texto) > 0:
            opcoes.append(texto[0])
        else:
            sys.exit("Opção inválida!")

# Roda renderizador com os parâmetros necessário para o exemplo escolhido
interpreter = sys.executable
for opcao in opcoes:
    print('Abrindo arquivo: "{0}"'.format(opcao[2]))
    print("> ", interpreter, "renderizador/renderizador.py", " ".join(opcao[1:]), "\n")

    proc = subprocess.Popen([interpreter, "renderizador/renderizador.py"] + opcao[1:])
    subprocesses.append(proc)

# Mantem código rodando até que o usuário pressione Ctrl+C
try:
    while True:
        # Verifica se algum subprocesso ainda rodando
        running = any(proc.poll() is None for proc in subprocesses)
        if not running:
            break
        time.sleep(1)
except KeyboardInterrupt:
    signal_handler(signal.SIGINT, None)