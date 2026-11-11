# Inferência do Random Forest: dois caminhos

Duas formas de rodar a inferência do Random Forest do `RF_macro.py` sobre uma
cena inteira. O modelo é o mesmo nos dois; muda **quando** e **como** as
árvores são percorridas.

Cada script detecta a GPU sozinho. Sem CUDA, cai para a CPU com o mesmo modelo
e a mesma saída — muda o tempo, não o resultado.

| script | o que faz | `max_depth` |
|---|---|---|
| `rf_floresta.py` | percorre as 500 árvores, pixel por pixel | parâmetro |
| `rf_tabela.py` | lê a resposta numa tabela pré-calculada | não se aplica |

`rf_kernel.py` é a parte de GPU (achatamento da floresta e o kernel Triton).
Sem CUDA nenhum dos dois scripts precisa dele.

---

## 1. O que foi reproduzido e o que não foi

- **Features idênticas**: PAN, GLCM entropia, rugosidade (STD + gradiente).
- **Hiperparâmetros idênticos**: 500 árvores, `n_jobs=-1`, blocos de 2048,
  `random_state=42`, 300.000 px de treino com cota igual por classe.
- **Rótulos diferentes**: a amostra usada foi a `mascara_amostra_v1` deste
  projeto, com **3 classes** (vegetação=3, não vegetação=15, água=33). Os
  números de **acurácia** abaixo valem para essa amostra e não são comparáveis
  com outra. Os de **tempo** são.
- **Sem `gdal.SieveFilter`**: os mapas em `mapas/` são a saída crua do RF.

---

## 2. Caminho 1 — a floresta (`rf_floresta.py`)

A travessia normal. `max_depth` é parâmetro de `--treinar`: `0` é a
profundidade livre do `RF_macro.py` original, ou um valor fixo.

```bash
# treinar (grava modelo.joblib para a CPU e floresta/ para a GPU)
python rf_floresta.py --treinar --modelo runs/rf_livre \
    --imagem <cena>.tif --rotulos <rotulos>.tif --max-profundidade 0

python rf_floresta.py --treinar --modelo runs/rf_d12 \
    --imagem <cena>.tif --rotulos <rotulos>.tif --max-profundidade 12

# classificar, GPU se houver
python rf_floresta.py --modelo runs/rf_d12 \
    --imagem <cena>.tif --saida f.tif

# forçar CPU (o rf.predict do sklearn, idêntico ao RF_macro.py)
python rf_floresta.py --modelo runs/rf_d12 \
    --imagem <cena>.tif --saida f.tif --dispositivo cpu
```

- **CPU** → `<modelo>/modelo.joblib`, `rf.predict` do sklearn
- **GPU** → `<modelo>/floresta/`, kernel Triton, uma thread por pixel
- As duas vias leem o mesmo modelo (mesmo número de nós) e produzem o mesmo
  mapa: conferido **100,0000%** na cena inteira

O treino separa validação por **bloco espacial** de 4096 px, não por pixel
sorteado: pixels vizinhos são quase idênticos, então sortear pixel a pixel põe
o mesmo conteúdo nos dois lados e infla a acurácia.

### Os dois modelos treinados

| | profundidade livre | `max_depth=12` |
|---|---|---|
| profundidade | 75 | 12 |
| nós | 37.111.334 | 1.552.486 |
| modelo em memória | 1,48 GB | 60 MB |
| treino | 24 s | 18 s |
| acurácia (validação, 3 M px) | 0,7836 | 0,8022 |

A diferença de velocidade entre os dois não vem da quantidade de comparações,
e sim de onde os nós ficam — ver a seção 5.

### Fatiamento

O `predict` é chamado em fatias (262.144 px na CPU, 1.048.576 na GPU). Isso não
é cosmético: entregar os 4,19 M px de um bloco 2048² de uma vez joga o sklearn
em swap — fatiando, o **mesmo modelo** ficou 2,4× mais rápido. No kernel
Triton, a fatia grande com a floresta de 1,48 GB causa acesso ilegal de
memória.

---

## 3. Caminho 2 — a tabela (`rf_tabela.py`)

O modelo olha 3 números por pixel e nada mais. A PAN é Byte, logo só pode ter
256 valores. Cada textura foi cortada em 256 faixas, com as bordas nos quantis
dos limiares que a própria floresta usa — resolução fina onde ela decide,
grossa onde ela não pergunta nada.

Isso fecha o espaço de entrada em 256 × 256 × 256 = **16.777.216 combinações
possíveis**.

`rf_tabela.py --montar` roda a floresta uma vez em cada combinação (a PAN
inteira, e o valor do centro de cada faixa de textura, em escala real) e grava
a resposta em um byte. **16,8 MB, 11 segundos, sem imagem aberta.**

Na inferência nenhuma árvore é percorrida:

```
PAN  142   -> a = 142         (é o próprio valor, é Byte)
GLCM 0,83  -> b = 171         (em qual das 256 faixas cai)
rug  1,21  -> c =  94

índice = (142 × 256 + 171) × 256 + 94 = 9.343.838
classe = tabela[9.343.838]
```

**1 leitura por pixel**, numa tabela de 16,8 MB que cabe em cache — contra
~43.000 leituras em 1,48 GB.

```bash
# montar (uma vez; 11 s na GPU, alguns minutos na CPU com o sklearn)
python rf_tabela.py --montar --modelo runs/rf_livre --tabela tabela_256.npz

# conferir contra a travessia da MESMA floresta, num tile
python rf_tabela.py --verificar --modelo runs/rf_livre \
    --tabela tabela_256.npz --imagem <cena>.tif

# classificar
python rf_tabela.py --tabela tabela_256.npz --imagem <cena>.tif --saida t.tif
python rf_tabela.py --tabela tabela_256.npz --imagem <cena>.tif --saida t.tif \
    --dispositivo cpu
```

CPU usa `np.searchsorted` + indexação; GPU usa `torch.bucketize` + gather.
Também **100,0000%** idênticos entre si.

A montagem na grade regular é, por si, mais rápida que em pixels reais:
vizinhos da grade caem no mesmo caminho da árvore, então a GPU lê o nó uma vez
e serve 32 threads. Medido: 1,504 Mentradas/s na grade contra 0,0795 Mpx/s nos
pixels reais — 19×.

### Por que dá a mesma resposta

A floresta é uma função de 3 variáveis e nada mais — sem contexto, sem
vizinho, sem posição. Dois pixels com os mesmos 3 números recebem
obrigatoriamente a mesma classe, então a resposta pode ser guardada e
reaproveitada.

**Na PAN é exato.** Os 471 limiares que a floresta usa nessa feature caem todos
na grade de 0,5 entre inteiros, e nenhum pixel pode cair entre dois inteiros.
Essa feature não perde nada.

**Nas texturas é uma quantização não uniforme.** Elas são float, com ~1,1 e
~1,7 milhão de limiares distintos — impossível tabular sem perda. Cada
intervalo passa a ser representado pelo seu ponto médio:

```
faixa 246 = [+1,882170 , +1,920927]      ->  representante +1,901549

GLCM +1,886011  ->  faixa 246  ->  responde pelo que foi calculado em +1,901549
GLCM +1,918888  ->  faixa 246  ->  a MESMA resposta
```

A quantização não é uniforme de propósito. Cortar em 256 pedaços de largura
igual desperdiçaria resolução: a rugosidade vai de −1,8 a +11,3 mas a floresta
decide quase tudo perto de zero, então metade das faixas cairia numa cauda onde
ela não pergunta nada. Com as bordas nos quantis dos limiares dela, a faixa
fica estreita onde ela decide muito e larga onde não decide — e faixa larga num
trecho sem limiar não muda resposta nenhuma, porque todo valor dali desce o
mesmo caminho.

As faixas das pontas vão até ±infinito, então **nenhum valor pode cair fora da
tabela**. E essas duas também são exatas: abaixo do menor limiar (ou acima do
maior) a floresta não tem mais nada a perguntar.

### O custo da aproximação, medido

```
pixels em que a tabela discorda da floresta livre:  3,5%
nesses pixels, quem acertou:   tabela 48,3%  |  floresta 50,6%
acurácia (validação, 3 M px):  tabela 0,7843 |  floresta 0,7836
```

Os 3,5% caem em cima de bordas de decisão, onde a própria floresta está
indecisa. Grades mais finas foram testadas e **não melhoraram** a acurácia:
512³ deu 0,7837 (67 MB) e 1024³ deu 0,7833 (268 MB), contra 0,7843 do 256³.
Isso indica que 256 faixas já estão mais finas que a precisão real do modelo,
e por isso só a tabela de 256³ acompanha esta pasta. `--bins` em
`rf_tabela.py --montar` reconstrói qualquer uma delas.

### Limites

O custo da tabela é **fixo** (16,7 M travessias na montagem), não por pixel:

| imagem | floresta percorre | tabela percorre |
|---|---|---|
| 8×8 = 64 px | 64 | 16.777.216 |
| 2048×2048 = 4,19 M | 4.194.304 | 16.777.216 |
| ~4096×4096 = 16,7 M | 16.777.216 | 16.777.216 |
| 29068×32624 = 295,9 M | 295.863.990 | 16.777.216 |
| 80 Gpx | 80.000.000.000 | 16.777.216 |

Ponto de virada: **~4096×4096 px**. Abaixo disso a floresta percorre menos que
a tabela, e `rf_floresta.py` é o caminho mais rápido.

Duas restrições estruturais:

1. **Depende de haver poucas features e uma delas ser Byte.** Com 4 features a
   tabela iria a 4,3 GB; com 5, a 1,1 TB. Não é uma técnica geral.
2. **Uma tabela por modelo treinado.** Retreinou, monte de novo (11 s).

---

## 4. Tempos medidos — cena inteira

`colinas_do_sul`, 29068 × 32624, **295.863.990 px válidos**. Cada medição em
seu próprio processo. O tempo está separado em `io` (ler o bloco, gravar a
saída) e `calculo`.

| caminho | dispositivo | total | io | cálculo | Mpx/s cálculo |
|---|---|---|---|---|---|
| floresta, livre | CPU | — | — | ~4300 s * | 0,0688 * |
| floresta, livre | GPU | — | — | — | 0,0795 * |
| floresta, `max_depth=12` | CPU | 1169 s | 125 s | 1035 s | 0,2859 |
| floresta, `max_depth=12` | GPU | 166 s | 114 s | 43 s | 6,8550 |
| tabela 256³ | CPU (1 núcleo) | 143 s | 109 s | 26 s | 11,5059 |
| tabela 256³ | GPU | 131 s | 113 s | **8 s** | 36,8285 |

\* As duas linhas da floresta livre foram medidas num tile de 2048²
(4.192.723 px, 16 fatias, zero swap) e extrapoladas; a cena inteira não foi
rodada por esse caminho. Em GPU ela não completa nesta máquina: é um notebook,
e após ~160 s de kernel saturado a placa vai a 86 °C, o clock cai de 1777 para
1537 MHz e o driver devolve `illegal memory access` (reproduzido 4 vezes, com
fatias de 1 M, 131 k e 65 k px — sempre na terceira passada; todo bloco passa
isoladamente, e a VRAM fica estável em 1,49 GB, portanto é limite térmico e não
erro do kernel).

Tomando a floresta livre em CPU como referência de cálculo, os ganhos são
4,2× (`d12` CPU), 99,6× (`d12` GPU), 167,2× (tabela CPU) e 535,3× (tabela GPU).

### Consequência

Pelo caminho da tabela, o disco passa a ser 86% do tempo total (113 s de io
contra 8 s de cálculo). A partir daí otimizar o modelo não altera mais o
resultado prático; o que resta é leitura paralela, armazenamento mais rápido,
ou dividir a cena entre processos ou máquinas — cada uma leva uma cópia de
16,8 MB e não precisa de GPU.

---

## 5. Por que o ganho da GPU varia de 1,2× a 535×

Mesmo kernel, mesma placa, mesmo código de varredura. Só muda o modelo:

| | GPU sobre CPU, no cálculo |
|---|---|
| floresta, profundidade livre | **1,2×** |
| floresta, `max_depth=12` | **24×** |
| tabela 256³ | **3,2×** (e 535× sobre a floresta livre em CPU) |

### O que a árvore realmente faz

Classificar um pixel é descer 500 árvores. Cada passo é um `if`: lê o nó, compara
um número, vai para a esquerda ou para a direita. A comparação em si é de graça
— um único ciclo de processador. O que custa é **ler o nó**.

E cada leitura depende da anterior:

```
lê o nó      0  ->  ele diz "vá para o nó 530"
lê o nó    530  ->  ele diz "vá para o nó 8842"
lê o nó   8842  ->  ...
```

O endereço do próximo nó só existe depois de ler o atual. Essa fila é
**sequencialmente obrigatória**: não há como pedir as leituras adiantadas, nem
em CPU nem em GPU. Num pixel real da colinas, medido: ~12.000 nós visitados e
~43.000 leituras de memória, espalhadas por 1,48 GB.

### Onde a GPU ganha, e onde não

A GPU não acelera uma fila dessas. Ela ganha por quantidade: cada pixel é
independente dos outros, então são 295 milhões de filas paralelas.

```
CPU   ~10 núcleos     ->  ~10 pixels descendo a árvore ao mesmo tempo
GPU   ~3.500 núcleos  ->  ~46.000 pixels descendo ao mesmo tempo
```

O problema é que a GPU paga caro por cada leitura: a latência da VRAM é ~600 ns
contra ~80 ns da RAM da CPU, 7,5× pior. Ela compensa isso com **acesso
coalescido** — 32 threads vizinhas pedindo endereços vizinhos são atendidas numa
transação só.

Uma travessia de árvore não tem coalescência nenhuma. Cada thread está num nó
diferente, num canto diferente dos 1,48 GB. São 32 transações separadas em vez
de uma. Resultado: os núcleos ficam ocupados **esperando memória**, não
calculando — a ocupação medida mostra 130 blocos por SM, ~8× sobrecarregado, e
ainda assim a placa não anda. Daí os 1,2×.

### Por que `max_depth=12` muda tudo

Não é por fazer menos comparações (11 níveis em vez de 24, só 2×). É por onde
os nós passam a morar:

```
floresta livre : 37.111.334 nós, 1,48 GB   -> nada cabe em cache, ~600 ns por leitura
max_depth=12   :  1.552.486 nós,   60 MB   -> os primeiros níveis ficam em cache
```

Os primeiros 12 níveis das 500 árvores somam ~2 MB, e **todo pixel passa por
eles**. Isso fica parado no cache, a ~20 ns em vez de ~600 ns. A leitura deixa
de ser o gargalo e a GPU finalmente usa os núcleos para calcular: 24×.

Pelo mesmo motivo a CPU também melhora 4,2× com `max_depth=12`, embora ela faça
as mesmas 2× menos comparações. A velocidade está vindo do cache, não da
aritmética.

### E a tabela é o extremo disso

```
leituras por pixel     : 1 (contra ~43.000)
tamanho do modelo      : 16,8 MB (contra 1,48 GB)
fila sequencial        : nenhuma -- o endereço é calculado com 3 multiplicações
coalescência           : pixels vizinhos caem em endereços vizinhos
```

Aqui a GPU faz exatamente o que ela é boa em fazer, e por isso o lookup é 3,2×
mais rápido em GPU que em CPU (25,7 s contra 8,0 s de cálculo na cena inteira),
mesmo sendo uma operação trivial nas duas.

O ganho da GPU é pequeno em termos relativos justamente porque o cálculo já
quase não existe: 26 s na CPU dentro de um total de 143 s. O que importa aqui
não é mais a GPU, é a seção 4.

### Em uma frase

A travessia de árvore é limitada por **latência de memória aleatória**, que é a
única coisa em que a CPU é melhor que a GPU. Encolher o modelo (`max_depth`) ou
eliminar a travessia (tabela) é o que libera a GPU — e não o contrário.

---

## 6. Concordância entre as saídas

Resolução cheia, nos 31,9 M px cobertos por todas as saídas disponíveis:

```
floresta d12   CPU == GPU      100,0000%
tabela 256³    CPU == GPU      100,0000%
floresta livre  vs  tabela      96,5383%     (os 3,5% da quantização)
floresta livre  vs  d12         93,4938%     (modelos diferentes)
```

Distribuição de classes (vegetação / não vegetação / água):

```
floresta livre   78,96%  12,62%   8,43%
floresta d12     81,56%  10,63%   7,81%
tabela 256³      78,96%  12,59%   8,45%
```

A tabela reproduz a distribuição da floresta livre; o `d12` produz um mapa
visivelmente mais "liso", com menos sal-e-pimenta.

---

## 7. Conteúdo da pasta

```
LEIA.md                      este arquivo

rf_floresta.py               treino + travessia das árvores (GPU ou CPU)
rf_tabela.py                 montagem + lookup na tabela (GPU ou CPU)
rf_kernel.py                 a parte de GPU: achatamento e kernel Triton

tabela_256.npz               a tabela pronta, 16,8 MB

mapas/
  pan_livre_d12_tabela.png   PAN + os mapas lado a lado
  mapa_d12.png               cena inteira, max_depth=12
  mapa_tabela.png            cena inteira, tabela 256³

tempos/
  cpu_d12.json               as medições da seção 4
  gpu_d12.json
  cpu_tabela.json
  gpu_tabela.json
```

Cada script tem a documentação completa na própria docstring (`--help`).

Dependências: GDAL, numpy, scikit-learn, joblib. Para GPU: torch + triton.
`rf_kernel.py` só é importado quando há CUDA.
