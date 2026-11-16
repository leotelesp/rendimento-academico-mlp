# Predição do rendimento acadêmico com MLP

Atividade 02 de Sistemas Inteligentes — professora Malga Costa.

Projeto em Python para comparar redes com ReLU e Tanh na classificação da aprovação em Língua Portuguesa. Fonte: [Student Performance — UCI](https://archive.ics.uci.edu/dataset/320/student-performance), Paulo Cortez, DOI 10.24432/C5TG7T, licença CC BY 4.0.

## Situação do projeto

Treinamento executado pelo grupo. Este pacote contém resultados reais, dados e relatório revisado. Nenhum repositório foi criado automaticamente. Para apenas regenerar o PDF, use atualizar_relatorio.py, sem repetir o treinamento.

## 1 Abrir no VS Code

Extraia o ZIP. No VS Code, use **Arquivo > Abrir Pasta** e selecione a pasta que contém este README. Instale a extensão **Python**, da Microsoft, se ainda não estiver instalada. Abra **Terminal > Novo Terminal**.

Confira os interpretadores disponíveis:

```powershell
py -0p
```

Os comandos abaixo usam Python 3.11 de 64 bits; Python 3.12 também pode ser usado, substituindo `-3.11` por `-3.12`. Se nenhuma dessas versões estiver instalada, instale uma antes de continuar. Não é necessário desinstalar outras versões.

## 2 Criar ambiente e instalar bibliotecas

Execute um comando por vez, no terminal PowerShell da pasta do projeto:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Esses comandos usam o Python do ambiente diretamente e dispensam ativação de scripts no PowerShell. A instalação do TensorFlow pode demorar. Não precisa configurar GPU.

No VS Code, pressione **Ctrl+Shift+P**, escolha **Python: Select Interpreter** e selecione `.venv\Scripts\python.exe`.

## 3 Identificar o grupo

Edite `config.json`: preencha autores, instituição, curso e cidade. Mantenha aspas e vírgulas. O trabalho é em dupla ou trio. Esses dados serão colocados na capa do relatório.

## 4 Executar

```powershell
.\.venv\Scripts\python.exe rendimento_academico.py
```

Os vetores de erros aparecerão no terminal a cada lote. Aguarde a mensagem `Concluído`. Fechar o terminal interrompe a execução.

O programa cria `entrega_rendimento` e `Entrega_Rendimento_Academico.zip`, com:

- CSV original e documentação da UCI, quando presente no pacote;
- código e configuração do grupo;
- divisão de treino, validação e teste;
- parâmetros de padronização, modelos e pré-processamento;
- histórico, vetores de erros e métricas;
- gráficos e matrizes de confusão;
- relatório em PDF gerado com resultados da execução.

Se houver falha no download, verifique a conexão com a UCI. Se houver erro de instalação ou execução, copie a mensagem completa para diagnóstico. Não substitua métricas por números fictícios.

## 5 Revisar o trabalho

Abra o PDF gerado e confira todas as páginas, nomes, gráficos, quebras e texto. A formatação usa A4, margens 3/2 cm, Times 12 e entrelinhas 1,5; confira as exigências específicas da instituição antes da entrega.

Período confirmado no artigo original de Cortez e Silva (2008): ano letivo de 2005–2006. A data de depósito da base na UCI é distinta da coleta. Os dados são de escolas portuguesas, históricos e não representam diretamente estudantes brasileiros atuais.

Compare os modelos com a baseline majoritária, incluindo acurácia balanceada e recall da reprovação. O grupo deve compreender, revisar e adaptar o trabalho e seguir as regras da disciplina sobre assistência de IA.

## 6 Enviar ao GitHub

Crie um repositório vazio na sua conta, inicialmente privado se preferir. Não marque a criação de README, licença ou gitignore, pois os arquivos já estão nesta pasta. Substitua a URL abaixo pela URL real do repositório.

```powershell
git init
git add .
git status
git commit -m "Adiciona projeto de rendimento academico com MLP"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
git push -u origin main
```

Revise `git status` antes do commit. O `.gitignore` evita enviar o ambiente virtual, arquivos temporários, modelos binários e os grandes registros de erros. Esses resultados continuam no ZIP de entrega. O CSV, o relatório e gráficos gerados podem ser versionados.

Depois de executar e revisar os resultados:

```powershell
git add .
git status
git commit -m "Adiciona resultados do treinamento e relatorio"
git push
```

Se o repositório for privado, a professora precisará de acesso. Não envie senha ou token no código. Não execute `git init` em outra pasta ou dentro de um repositório existente sem conferir sua estrutura.

## Resumo da metodologia

649 registros de Português; alvo `G3 >= 10`; 12 atributos de contexto; G1/G2 e faltas acumuladas excluídos. Divisão estratificada 60/20/20, semente 42. Padronização e one-hot ajustados somente no treino. Três camadas ocultas de 32, 16 e 8 neurônios; saída sigmoid. ReLU e Tanh, taxas 0,001 e 0,0003, Adam, batch 32, máximo 120 épocas, parada após 15 épocas sem melhora na validação. A seleção usa validação, nunca teste. Cada lote exibe e salva o vetor `y - probabilidade`; a função de perda otimizada é a entropia cruzada.

Entrega prevista no enunciado: 18/11/2026. Entregar dataset, código e relatório técnico.


## Relatório revisado

O treinamento foi executado pelo grupo. Esta revisão preserva todos os resultados e acrescenta o período de coleta, a análise dos históricos e a discussão da classe minoritária. As referências foram reunidas.

Para atualizar só o PDF nesta pasta, sem treinar novamente, execute com o Python do ambiente: `python atualizar_relatorio.py`. Mantenha os arquivos CSV, JSON e PNG ao lado desse script. No projeto principal, também é possível usar `python rendimento_academico.py --somente-relatorio` quando os resultados estiverem na subpasta `entrega_rendimento`.
