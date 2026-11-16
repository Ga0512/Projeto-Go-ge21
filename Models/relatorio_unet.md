# Modelo U-Net++ para LULC em imagem pancromática

## 1. Conceito do modelo

A U-Net é uma rede convolucional de **segmentação semântica**: atribui uma classe a cada pixel, devolvendo um mapa do tamanho da entrada. Tem a forma de um "U" — um *encoder* que reduz a resolução pela metade a cada nível, ganhando contexto, e um *decoder* que a recupera. O elemento definidor são as **conexões de salto**, que concatenam cada nível do encoder ao nível correspondente do decoder: o encoder ganha contexto mas perde localização da borda, e as conexões devolvem a informação espacial fina. É por isso que a U-Net delimita bem o contorno de um polígono, e não apenas a sua existência.

Adotou-se a **U-Net++**, variante em que as conexões de salto passam por uma grade densa de blocos intermediários, de modo que o decoder recebe features já refinadas e alinhadas ao nível em que serão usadas. O ganho é precisão de borda — a métrica crítica quando o produto é um mapa de uso e cobertura cujo objeto de interesse é a fronteira entre classes. Configuração: 4 níveis, 32 filtros na base, patches de 256 × 256 px (640 m de terreno a 2,5 m).

A entrada **não é a imagem bruta**. Sendo a imagem pancromática, não há informação espectral para separar classes, apenas brilho; a discriminação vem da **textura**, em 3 canais: PAN esticada entre os percentis 2 e 98 da cena, entropia GLCM (janela 7, 8 níveis) e rugosidade (média normalizada do desvio-padrão local 7×7 e do gradiente de Sobel). Um pasto e um fragmento florestal podem ter brilho médio semelhante; a sua rugosidade e entropia locais, não.

## 2. Estratégia de treinamento

**Perda:** entropia cruzada + Dice suave. A primeira otimiza a decisão pixel a pixel; o Dice otimiza diretamente a sobreposição, compensando em parte o desbalanço entre classes. Pixels ignorados não produzem gradiente.

**Aumentação:** 8 combinações de espelhamento e rotação de 90°, aplicadas a imagem e máscara em conjunto; ganho e deslocamento de brilho aleatórios **só na banda PAN**, porque os canais de textura são padronizados por escore-z e invariantes a brilho.

**Seleção de modelo:** melhor época pelo **mIoU de validação**, com parada antecipada. O teste é avaliado **uma única vez, no fim**, com os pesos dessa época — nenhuma decisão é tomada observando o teste, e é isso que torna os números uma estimativa honesta de generalização.

**Divisão dos dados (v18):** 4.304 patches de treino, 740 de validação, 656 de teste, sobre 4 cenas. Partição **espacial**, não aleatória por patch, para que patches sobrepostos da mesma região não caiam em conjuntos diferentes inflando o desempenho.

**Rebalanceamento (v19).** Diagnosticou-se que 85,6% dos pixels de não vegetação do treino vêm de uma só cena (Jataí) e 14,4% de Colinas do Sul. Duas intervenções foram testadas de forma **pré-registrada**, com critérios de aceitação definidos antes de rodar: ponderação de classe na entropia cruzada (peso 2 para não vegetação, nunca no Dice) e reamostragem do sorteio de patches com peso proporcional ao déficit da classe na cena, `w = 1 + k · déficit · (frac_nveg do patch / maior frac_nveg da cena)`, com k = 3 — pico de 2,76× a média e amostra efetiva de 90% dos patches, sem excluir nenhuma região.

## 3. Validação

Protocolo fixo, idêntico para as três famílias de modelo do projeto (U-Net++, ViT e XGBoost por pixel), o que torna os resultados diretamente comparáveis. Métricas: acurácia global, mIoU, e acurácia/precisão/recall/F1/IoU por classe, mais matriz de confusão e desagregação por cena. Todas calculadas duas vezes — no patch inteiro e no "interior", descartando 15 px de margem. A margem existe porque as janelas de textura do modelo por pixel (até 31 px) só cabem inteiras ali; **comparações entre famílias usam sempre o interior**, para não premiar um modelo por artefato de borda.

Teste (interior):

| | OA | mIoU | Vegetação | Não veg. | Água |
|---|---|---|---|---|---|
| v18 | 0,9570 | 0,9205 | 0,9283 | 0,9003 | 0,9331 |
| v19a (peso) | 0,9154 | 0,8674 | 0,8565 | 0,8195 | 0,9261 |
| v19b (reamostragem) | 0,9512 | 0,9104 | 0,9181 | 0,8890 | 0,9239 |

Por cena, IoU / recall / precisão:

| | Colinas — Não veg. | Jataí — Vegetação |
|---|---|---|
| v18 | 0,030 / 0,030 / 0,852 | 0,865 / 0,922 / 0,933 |
| v19a | 0,175 / 0,196 / 0,628 | 0,543 / 0,569 / 0,921 |
| v19b | 0,175 / 0,191 / 0,673 | 0,817 / 0,897 / 0,902 |

A ponderação de classe falhou em dois dos quatro critérios pré-registrados e foi descartada: o ganho em Colinas custou um colapso da vegetação em Jataí (0,865 → 0,543). A reamostragem atendeu três dos quatro, com perda marginal de mIoU global e recuperação parcial da vegetação de Jataí (0,817 contra o alvo de 0,84).

**Cena inteira.** O modelo foi aplicado às duas cenas completas, em faixas horizontais de 1.024 linhas com sobreposição de 300 px e predição em janelas deslizantes de passo 128 px, ponderadas com peso maior no centro — a borda do patch é onde o modelo erra mais. Não há descontinuidade visível entre faixas.

| cena | dimensão | válidos | Vegetação | Não veg. | Água |
|---|---|---|---|---|---|
| Colinas do Sul | 29.068 × 32.624 (948 Mpx) | 295,9 Mpx | 87,0% | 7,5% | 5,6% |
| Jataí | 50.444 × 57.804 (2.916 Mpx) | 886,8 Mpx | 25,7% | 74,3% | 0,0% |

A inspeção visual integra o protocolo e confirma coerência estrutural: em Jataí a vegetação forma um padrão dendrítico ao longo da drenagem, compatível com mata ciliar; em Colinas a água acompanha a margem do reservatório e as clareiras saem como polígonos compactos.

## 4. Limitações observadas

**A limitação dominante é a referência, não o modelo.** Os rótulos derivam do MapBiomas, de resolução nativa 30 m, reamostrado para a grade de 2,5 m. Cada pixel de rótulo cobre 144 pixels de imagem, e o erro de contorno herdado é da ordem de 12 px. O modelo é treinado para reproduzir uma referência mais grosseira que a sua própria saída, e é penalizado quando acerta uma borda que a referência errou. Em WorldView-1 (0,486 m, referência de 10 m) a desproporção é maior ainda. **São pseudo-rótulos, não verdade de campo.**

**Desbalanço geográfico do rótulo.** Colinas do Sul é o caso mais bem caracterizado: 758 mil pixels de não vegetação no teste, dos quais o v18 acertava 23 mil (recall 0,030) com precisão alta (0,852). Não é confusão entre classes, é **timidez** — o modelo só declara não vegetação quando tem muita certeza, e acerta quando declara. A causa é amostral: a cena contribui com 14,4% dos pixels da classe no treino, e só 599 patches têm mais de 10% dela.

**Há um teto que o rebalanceamento não atravessa.** É o achado mais importante: dois mecanismos **independentes** — ponderação e reamostragem — pararam no mesmo número (IoU 0,1754 e 0,1752), com 80,8% da não vegetação de Colinas ainda classificada como vegetação. A convergência indica que o limitante é a **informação no dado rotulado**, não a formulação da perda. Reamostrar compra iteração sobre os mesmos pixels; não cria amostra nova. A área distinta de Colinas é de 192,67 km², e cada pixel já aparece em média 6,06 vezes nos patches.

**Trade-off entre cenas.** Nenhuma configuração domina o v18 em todos os critérios: tudo que melhorou a classe minoritária em uma cena degradou a majoritária na outra.

**Outras.** Cerca de 69% da cena de Colinas é nodata (moldura do mosaico), o que reduz a área útil e precisa ser tratado em qualquer estatística de área. O treinamento roda em GPU de 6 GB, o que limita o tamanho de lote e a exploração de arquiteturas ou patches maiores. E a validação do código é feita por inspeção do relatório e das visualizações, sem suíte de testes, o que torna refatorações arriscadas.

## 5. Melhorias a serem feitas

**Alta prioridade — atacar a causa raiz.**

1. **Rotular manualmente não vegetação em Colinas do Sul.** É a única ação que remove o teto identificado acima. Não precisa ser a cena inteira: algumas centenas de polígonos de pasto, solo exposto e área urbana concentrados no sul e sudoeste, onde o erro é sistemático, mudam a distribuição amostral de um jeito que nenhum reajuste de perda consegue. Alternativa de custo menor: rotulagem assistida — predizer, corrigir só o errado, retreinar.
2. **Ampliar o número de cenas.** Com 4 cenas, desbalanço de classe e desbalanço geográfico são inseparáveis.

**Média prioridade — ganhos sem dado novo.**

3. **Varrer o parâmetro de reamostragem.** k = 3 foi a primeira tentativa; k = 1,5 deve recuperar parte da vegetação de Jataí preservando a maior parte do ganho em Colinas, já que o ganho satura mas a degradação não.
4. **Combinar reamostragem com ponderação moderada (v19c)** — ainda não executado.
5. **Aumentação em tempo de teste:** predizer as 8 simetrias e agregar. Custa 8× o tempo de inferência e devolve ganho pequeno mas gratuito em borda, sem retreinar.
6. **Calibração e limiar por classe.** Como o problema é timidez e não confusão (precisão 0,67 contra recall 0,19), ajustar o limiar da classe não vegetação sobre a validação é uma intervenção pós-treino de custo quase nulo, que troca precisão por recall explicitamente.

**Qualidade da referência.**

7. **Construir um conjunto de avaliação independente do MapBiomas**, ainda que pequeno, interpretado visualmente na resolução nativa. Sem isso não se separa erro do modelo de erro da referência.
8. **Avaliar transferência entre sensores.** Já existem modelos de WorldView-1; saber se um modelo de SPOT5 transfere para WorldView e vice-versa é informação direta sobre o custo de estender o método a novas áreas.

**Infraestrutura.**

9. **Registrar o commit do código e as versões de biblioteca em cada execução.** Hoje um resultado é rastreável até os hiperparâmetros, mas não até a versão exata do código.
10. **Um teste de fumaça** sobre o dataset mínimo já existente — uma época e conferência do esquema do relatório. Pré-requisito para refatorar com segurança.
