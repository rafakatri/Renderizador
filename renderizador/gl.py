#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: Rafael Eli Katri
Disciplina: Computação Gráfica
Data: 14/08
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy
from math import floor

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante

    tf_stack = [np.identity(4)]
    perspective_matrix = []
    screen_matrix = []
    visualization_matrix = []
    transform_matrix = []

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far

        GL.z_buffer = -np.inf * np.ones((GL.width, GL.height))

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polypoint2D
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        #print("Polypoint2D : pontos = {0}".format(point)) # imprime no terminal pontos
        #print("Polypoint2D : colors = {0}".format(colors)) # imprime no terminal as cores

        n = len(point)//2

        COLOR_TYPE = "emissiveColor"

        color = [int(el * 255) for el in colors[COLOR_TYPE]]

        for i in range(n):
            ind = i * 2
            x, y = floor(point[ind]), floor(point[ind + 1])
            if not(x < 0 or y < 0 or x > GL.width or y > GL.height):
                gpu.GPU.draw_pixel([int(point[ind]) , int(point[ind + 1])], gpu.GPU.RGB8, color) 

        # Exemplo:
        #pos_x = GL.width//2
        #pos_y = GL.height//2
        #gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 0])  # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)
        
    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polyline2D
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        #print("Polyline2D : lineSegments = {0}".format(lineSegments)) # imprime no terminal
        #print("Polyline2D : colors = {0}".format(colors)) # imprime no terminal as cores
        
        n = len(lineSegments)//2

        COLOR_TYPE = "emissiveColor"

        color = [int(el * 255) for el in colors[COLOR_TYPE]]

        for i in range(n - 1):
            ind = i * 2
            u1, v1, u2, v2 = lineSegments[ind:ind+4]

            delta_v = v2 - v1
            delta_u = u2 - u1

            if delta_u != 0:
                s = delta_v/delta_u
            else:
                s = math.inf
            
            if abs(s) < 1:
                if u2 < u1:
                    u1, v1, u2, v2 = u2, v2, u1, v1

                u = u1 
                v = v1
                               
                while (u <= u2):
                    if not(u >= GL.width or u <= 0 or v >= GL.height or v <= 0):
                        gpu.GPU.draw_pixel([int(u) , int(v)], gpu.GPU.RGB8, color)
                    v += s
                    u += 1

            else:
                if v2 < v1:
                    u1, v1, u2, v2 = u2, v2, u1, v1

                v = v1
                u = u1
                
                while (v <= v2):
                    if not(u >= GL.width or u <= 0 or v >= GL.height or v <= 0):
                        gpu.GPU.draw_pixel([int(u) , int(v)], gpu.GPU.RGB8, color)
                    u += 1/s
                    v += 1

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Circle2D
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).

        print("Circle2D : radius = {0}".format(radius)) # imprime no terminal
        print("Circle2D : colors = {0}".format(colors)) # imprime no terminal as cores
        
        # Exemplo:
        pos_x = GL.width//2
        pos_y = GL.height//2
        gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 255])  # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)


    @staticmethod
    def triangleSet2D(vertices, colors, z=[1,1,1], colorPerVertex=False, textCoord=False, currentTexture=False):
        """Função usada para renderizar TriangleSet2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#TriangleSet2D
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).
        #print("TriangleSet2D : vertices = {0}".format(vertices)) # imprime no terminal
        #print("TriangleSet2D : colors = {0}".format(colors)) # imprime no terminal as cores


        def is_inside(pixel, vertex):
            i_center, j_center = pixel
            a_u, a_v, b_u, b_v, c_u, c_v = vertex

            l1 = (b_v - a_v) * i_center - (b_u - a_u) * j_center + a_v * (b_u - a_u) - a_u * (b_v - a_v)
            l2 = (c_v - b_v) * i_center - (c_u - b_u) * j_center + b_v * (c_u - b_u) - b_u * (c_v - b_v)
            l3 = (a_v - c_v) * i_center - (a_u - c_u) * j_center + c_v * (a_u - c_u) - c_u * (a_v - c_v)

            return all([l1 > 0, l2 > 0, l3 > 0])
        
        def barycentric_coordinates(pixel, vertex):
            i_center, j_center = pixel
            a_u, a_v, b_u, b_v, c_u, c_v = vertex

            total_area = abs(a_u * (b_v - c_v) + b_u * (c_v - a_v) + c_u * (a_v - b_v))/2

            a0 = abs(i_center * (b_v - c_v) + b_u * (c_v - j_center) + c_u * (j_center - b_v))/2
            a1 = abs(i_center * (c_v - a_v) + c_u * (a_v - j_center) + a_u * (j_center - c_v))/2

            alfa = a0/total_area
            beta = a1/total_area
            gama = 1 - alfa - beta                
            z_value = 1/(alfa * (1/z[0]) + beta * (1/z[1]) + gama * (1/z[2]))

            return (alfa, beta, gama , z_value)

        def get_uv(pixel, vertex, texture):
            u_t0, v_t0, u_t1, v_t1, u_t2, v_t2 = texture

            alfa, beta, gama, z_value = barycentric_coordinates(pixel, vertex)
            u = z_value * ((alfa * u_t0/z[0]) + (beta * u_t1/z[1]) + (gama * u_t2/z[2]))
            v = z_value * ((alfa * v_t0/z[0]) + (beta * v_t1/z[1]) + (gama * v_t2/z[2]))
            return (u,v)

        def create_minimap(img):
            minimap = [img]

            temp_img = img
            while temp_img.shape[0] > 1 and temp_img.shape[1] > 1:
                h = max(1, temp_img.shape[0] // 2)
                w = max(1, temp_img.shape[1] // 2)

                downscaled_img = np.zeros((h, w, temp_img.shape[2]), dtype=temp_img.dtype)

                for i in range(h):
                    for j in range(w):
                        segment = temp_img[2 * i:2 * i + 2, 2 * j:2 * j + 2]
                        downscaled_img[i, j] = np.mean(segment, axis=(0, 1))

                minimap.append(downscaled_img)
                temp_img = downscaled_img

            return minimap
        

        def get_minimap_level(dudx, dudy, dvdx, dvdy):
            v1 = (dudx**2 + dvdx**2)**0.5
            v2 = (dudx**2 + dvdx**2)**0.5

            l = max(v1, v2)
            return int(math.log2(l))


        n = len(vertices)//6

        COLOR_TYPE = "emissiveColor"

        color = [int(el * 255) for el in colors[COLOR_TYPE]]

        for num in range(n):
            ind = num * 6
            ind_cor = num * 9
            a_u, a_v, b_u, b_v, c_u, c_v = vertices[ind:ind+6]
            
            if colorPerVertex:
                a_r, a_g, a_b, b_r, b_g, b_b, c_r, c_g, c_b = colorPerVertex[ind_cor:ind_cor+9]

            if textCoord:
                u_t0, v_t0, u_t1, v_t1, u_t2, v_t2 = textCoord[ind: ind+6]
                
                image = gpu.GPU.load_texture(currentTexture[0])
                image = np.flip(image[:, :, :3], axis=1)
                minimaps = create_minimap(image)

            #bounding box
            min_x = min([a_u, b_u, c_u])
            max_x = max([a_u, b_u, c_u])
            min_y = min([a_v, b_v, c_v])
            max_y = max([a_v, b_v, c_v])

            for j in range(int(min_y), int(max_y+1)):
                for i in range(int(min_x), int(max_x+1)):
                    i_center = i + 0.5
                    j_center = j + 0.5
                    pixel = (i_center, j_center)  

                    if is_inside(pixel, vertices[ind:ind+6]) and not(i >= GL.width or i <= 0 or j >= GL.height or j <= 0):
                        alfa, beta, gama, z_value = barycentric_coordinates(pixel, vertices[ind:ind+6])

                        if colorPerVertex:

                            r = z_value * ((alfa * a_r/z[0]) + (beta * b_r/z[1]) + (gama * c_r/z[2]))
                            g = z_value * ((alfa * a_g/z[0]) + (beta * b_g/z[1]) + (gama * c_g/z[2]))
                            b = z_value * ((alfa * a_b/z[0]) + (beta * b_b/z[1]) + (gama * c_b/z[2]))

                            color = [int(el * 255) for el in [r,g,b]]

                        elif textCoord:
                            u = z_value * ((alfa * u_t0/z[0]) + (beta * u_t1/z[1]) + (gama * u_t2/z[2]))
                            v = z_value * ((alfa * v_t0/z[0]) + (beta * v_t1/z[1]) + (gama * v_t2/z[2]))

                            u_10, v_10 = get_uv((i_center + 1, j_center), vertices[ind:ind+6], textCoord[ind: ind+6])
                            u_01, v_01 = get_uv((i_center, j_center + 1), vertices[ind:ind+6], textCoord[ind: ind+6])


                            dudx = image.shape[0] * (u_10 - u)
                            dudy = image.shape[0] * (u_01 - u)

                            dvdx = image.shape[0] * (v_10 - v)
                            dvdy = image.shape[0] * (v_01 - v)

                            d = get_minimap_level(dudx, dudy, dvdx, dvdy)
                            minimap_level = minimaps[d]

                            x = int(u * minimap_level.shape[0])
                            y = int(v * minimap_level.shape[1])
                            color = minimap_level[x][y][0:3]
                        
                        # z-buffer and transparency
                        if z_value > GL.z_buffer[i, j]:
                            GL.z_buffer[i, j] = z_value
                            color_final = []
                            for index in range(3):
                                cor_anterior = gpu.GPU.read_pixel([i, j], gpu.GPU.RGB8)[index] * colors["transparency"]
                                cor_nova = color[index] * (1 - colors["transparency"])
                                color_final.append(int(cor_anterior + cor_nova))
                            gpu.GPU.draw_pixel([i, j], gpu.GPU.RGB8, color_final)
                        else:
                            pass


    @staticmethod
    def triangleSet(point, colors, colorPerVertex=False, textCoord=False, currentTexture = False):
        """Função usada para renderizar TriangleSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleSet
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        #print("TriangleSet : pontos = {0}".format(point)) # imprime no terminal pontos
        #print("TriangleSet : colors = {0}".format(colors)) # imprime no terminal as cores

        n = len(point)//9

        for num in range(n):
            ind = num * 9
            a_u, a_v, a_w, b_u, b_v, b_w, c_u, c_v, c_w = point[ind:ind+9]

            #apply tf matrix to vertex
            tf_a = np.matmul(GL.transform_matrix, np.array([[a_u], [a_v], [a_w], [1]]))
            tf_b = np.matmul(GL.transform_matrix, np.array([[b_u], [b_v], [b_w], [1]]))
            tf_c = np.matmul(GL.transform_matrix, np.array([[c_u], [c_v], [c_w], [1]]))
            
            #camera space
            camera_space_a = np.matmul(GL.visualization_matrix, tf_a)
            camera_space_b = np.matmul(GL.visualization_matrix, tf_b)
            camera_space_c = np.matmul(GL.visualization_matrix, tf_c)

            #apply perspective
            perspective_a = np.matmul(GL.perspective_matrix, camera_space_a)
            perspective_b = np.matmul(GL.perspective_matrix, camera_space_b)
            perspective_c = np.matmul(GL.perspective_matrix, camera_space_c)

            #normalize
            ndc_a = perspective_a/perspective_a[3]
            ndc_b = perspective_b/perspective_b[3]
            ndc_c = perspective_c/perspective_c[3]

            #screen space
            screen_a = np.matmul(GL.screen_matrix, ndc_a)
            screen_b = np.matmul(GL.screen_matrix, ndc_b)
            screen_c = np.matmul(GL.screen_matrix, ndc_c)

            #z-values on camera space
            z = [camera_space_a[2], camera_space_b[2], camera_space_c[2]]

            GL.triangleSet2D([screen_a[0][0], screen_a[1][0], screen_b[0][0], screen_b[1][0], screen_c[0][0], screen_c[1][0]], colors, z, colorPerVertex, textCoord, currentTexture)


    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        #print("Viewpoint : ", end='')
        #print("position = {0} ".format(position), end='')
        #print("orientation = {0} ".format(orientation), end='')
        #print("fieldOfView = {0} ".format(fieldOfView))

        aspect_ratio = GL.width/GL.height
        fovy = 2 * np.arctan(np.tan(fieldOfView/2) * (GL.height/(GL.height**2 + GL.width**2)**0.5))
        top = GL.near * np.tan(fovy)
        right = top * aspect_ratio


        GL.transform_in(position, False, False)
        
        translate_inv = np.linalg.inv(GL.transform_out())

        GL.transform_in(False, False, orientation)

        orientation_inv = np.linalg.inv(GL.transform_out())

        GL.visualization_matrix = np.matmul(orientation_inv, translate_inv)

        GL.perspective_matrix = [[GL.near/right, 0, 0, 0],
                                 [0, GL.near/top, 0, 0],
                                 [0, 0, -(GL.far + GL.near)/(GL.far - GL.near), (-2 * GL.far * GL.near)/(GL.far - GL.near)],
                                 [0, 0, -1, 0],
                                ]
        
        GL.screen_matrix = [
            [GL.width/2, 0, 0, GL.width/2],
            [0, -GL.height/2, 0, GL.height/2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]



    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo para depois potencialmente usar em outras chamadas. 
        # Quando começar a usar Transforms dentre de outros Transforms, mais a frente no curso
        # Você precisará usar alguma estrutura de dados pilha para organizar as matrizes.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        #print("Transform : ")
        t = np.identity(4)
        s = np.identity(4)
        r = np.identity(4)
        
        if translation:
            #print("translation = {0} ".format(translation), end='') # imprime no terminal
            t = np.array([
                [1, 0, 0, translation[0]],
                [0, 1, 0, translation[1]],
                [0, 0, 1, translation[2]],
                [0, 0, 0, 1],
            ])
        if scale:
            #print("scale = {0} ".format(scale), end='') # imprime no terminal
            s = np.array([
                [scale[0], 0, 0, 0],
                [0, scale[1], 0, 0],
                [0, 0, scale[2], 0],
                [0, 0, 0, 1],
            ])
        
        if rotation:
            #print("rotation = {0} ".format(rotation), end='') # imprime no terminal
            cos = np.cos(rotation[3]/2)
            sen = np.sin(rotation[3]/2)
            q_r = cos
            q_i = rotation[0] * sen
            q_j = rotation[1] * sen
            q_k = rotation[2] * sen
            r = np.array([[1-2*(q_j**2 + q_k**2),2 * (q_i*q_j - q_k*q_r), 2 * (q_i*q_k + q_j*q_r), 0],
              [2 * (q_i*q_j + q_k*q_r), 1-2*(q_i**2 + q_k**2), 2 * (q_j*q_k - q_i*q_r), 0],
              [2 * (q_i*q_k - q_j*q_r), 2 * (q_j*q_k + q_i*q_r), 1-2*(q_i**2 + q_j**2), 0],
              [0, 0, 0, 1]
             ])
        #print("")
        GL.tf_stack.append(GL.tf_stack[-1] @ t @ r @ s)
        GL.transform_matrix = GL.tf_stack[-1]

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        #print(f"Saindo de Transform: {GL.tf_stack[-1]}")
        return GL.tf_stack.pop()

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleStripSet
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        #print("TriangleStripSet : pontos = {0} ".format(point), end='')
        #for i, strip in enumerate(stripCount):
        #    print("strip[{0}] = {1} ".format(i, strip), end='')
        #print("")
        #print("TriangleStripSet : colors = {0}".format(colors)) # imprime no terminal as cores

        points = []
        ind = 0

        for el in stripCount:
            for i in range(el-2):
                idx0 = ind * 3
                idx1 = (ind+1) * 3
                idx2 = (ind+2) * 3

                if i %2 == 0:
                    points.extend(point[idx0:idx0+3])
                    points.extend(point[idx1:idx1+3])
                    points.extend(point[idx2:idx2+3])
                else:
                    points.extend(point[idx0:idx0+3])
                    points.extend(point[idx2:idx2+3])
                    points.extend(point[idx1:idx1+3])
                ind += 1
            GL.triangleSet(points, colors)
            points = []
            ind += 2

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        #print("IndexedTriangleStripSet : pontos = {0}, index = {1}".format(point, index))
        #print("IndexedTriangleStripSet : colors = {0}".format(colors)) # imprime as cores

        point_list = []
        index_list = []
        for i in range(len(index)):
            
            if (index[i] == -1):
                for j in range(len(index_list) - 2):
                    idx0 = index_list[j] * 3
                    idx1 = index_list[j+1] * 3
                    idx2 = index_list[j+2] * 3

                    if j % 2 == 0:
                        point_list.extend(point[idx0:idx0+3])
                        point_list.extend(point[idx1:idx1+3])
                        point_list.extend(point[idx2:idx2+3])
                    else:
                        point_list.extend(point[idx0:idx0+3])
                        point_list.extend(point[idx2:idx2+3])
                        point_list.extend(point[idx1:idx1+3])

                    GL.triangleSet(point_list, colors)
                    point_list = []

                index_list = []

            else:
                index_list.append(index[i])


    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#IndexedFaceSet
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.

        # Os prints abaixo são só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        #print("IndexedFaceSet : ")
        #if coord:
            #print("\tpontos(x, y, z) = {0}, coordIndex = {1}".format(coord, coordIndex))
        #print("colorPerVertex = {0}".format(colorPerVertex))
        #if colorPerVertex and color and colorIndex:
        #    print("\tcores(r, g, b) = {0}, colorIndex = {1}".format(color, colorIndex))
        #if texCoord and texCoordIndex:
        #    print("\tpontos(u, v) = {0}, texCoordIndex = {1}".format(texCoord, texCoordIndex))
        #if current_texture:
        #    image = gpu.GPU.load_texture(current_texture[0])
        #    print("\t Matriz com image = {0}".format(image))
        #    print("\t Dimensões da image = {0}".format(image.shape))
        #print("IndexedFaceSet : colors = {0}".format(colors))  # imprime no terminal as cores
        point_list = []
        color_list = []
        text_list = []
        index_list = []
        for i in range(len(coordIndex)):
            
            if (coordIndex[i] == -1):
                for j in range(1, len(index_list) - 1):
                    idx0 = index_list[0] * 3
                    idx1 = index_list[j] * 3
                    idx2 = index_list[j+1] * 3

                    point_list.extend(coord[idx0:idx0+3])
                    point_list.extend(coord[idx1:idx1+3])
                    point_list.extend(coord[idx2:idx2+3])

                    if color is not None:
                        color_list.extend(color[idx0:idx0+3])
                        color_list.extend(color[idx1:idx1+3])
                        color_list.extend(color[idx2:idx2+3])
                        GL.triangleSet(point_list, colors, colorPerVertex=color_list)

                    elif texCoord is not None:
                        id_txt0 = index_list[0] * 2
                        id_txt1 = index_list[j] * 2
                        id_txt2 = index_list[j+1] * 2

                        text_list.extend(texCoord[id_txt0:id_txt0+2])
                        text_list.extend(texCoord[id_txt1:id_txt1+2])
                        text_list.extend(texCoord[id_txt2:id_txt2+2])
                        GL.triangleSet(point_list, colors, textCoord=text_list, currentTexture=current_texture)

                    else:
                        GL.triangleSet(point_list, colors)

                    point_list = []
                    color_list = []
                    text_list = []

                index_list = []

            else:
                index_list.append(coordIndex[i])

            

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores
    
    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cone : bottomRadius = {0}".format(bottomRadius)) # imprime no terminal o raio da base do cone
        print("Cone : height = {0}".format(height)) # imprime no terminal a altura do cone
        print("Cone : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder
        # A função cylinder é usada para desenhar cilindros na cena. O cilindro é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da base do cilindro e o argumento height especifica a altura do cilindro.
        # O cilindro é alinhado com o eixo Y local. O cilindro é fechado por padrão em ambas as extremidades.
        # Para desenha esse cilindro você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cylinder : radius = {0}".format(radius)) # imprime no terminal o raio do cilindro
        print("Cylinder : height = {0}".format(height)) # imprime no terminal a altura do cilindro
        print("Cylinder : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("NavigationInfo : headlight = {0}".format(headlight)) # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
