from pathlib import Path
import json
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4
from xml.sax.saxutils import escape
OUT=Path(__file__).resolve().parent
config=json.loads((OUT/'config.json').read_text(encoding='utf-8'))
AUTORES=config['autores']; INSTITUICAO=config['instituicao']; CURSO=config['curso']; CIDADE=config['cidade']


def gerar_relatorio(a, runs, metrics, selected):
    """PDF gerado apenas após treinamento e avaliação; revisar visualmente no Colab."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CorpoABNT', fontName='Times-Roman', fontSize=12, leading=18, alignment=TA_JUSTIFY, firstLineIndent=1.25 * cm, spaceAfter=6))
    styles.add(ParagraphStyle(name='CentroABNT', fontName='Times-Roman', fontSize=12, leading=18, alignment=TA_CENTER, spaceAfter=12))
    story = []
    references = False
    styles.add(ParagraphStyle(name='Referencia', fontName='Times-Roman', fontSize=12, leading=12, alignment=0, spaceAfter=12, wordWrap='LTR'))

    def p(t):
        story.append(Paragraph(escape(t), styles['Referencia' if references else 'CorpoABNT']))

    def h(t):
        story.append(Paragraph(escape(t), styles['Heading2']))

    def center(t):
        story.append(Paragraph(escape(t), styles['CentroABNT']))

    def page():
        story.append(PageBreak())

    def pic(name, w, hgt):
        story.append(Image(str(OUT / name), width=w * cm, height=hgt * cm))
    center(INSTITUICAO)
    center(CURSO)
    story.append(Spacer(1, 2 * cm))
    center(AUTORES)
    story.append(Spacer(1, 3 * cm))
    center('PREDIÇÃO DO RENDIMENTO ACADÊMICO COM REDES NEURAIS MULTICAMADAS')
    story.append(Spacer(1, 5 * cm))
    center(CIDADE)
    center('2026')
    page()
    center(AUTORES)
    story.append(Spacer(1, 2 * cm))
    center('PREDIÇÃO DO RENDIMENTO ACADÊMICO COM REDES NEURAIS MULTICAMADAS')
    p('Relatório técnico apresentado à disciplina Sistemas Inteligentes, ministrada pela professora Malga Costa, como parte da Atividade 02 de Inteligência Artificial Aplicada a Dados Reais.')
    story.append(Spacer(1, 7 * cm))
    center(CIDADE)
    center('2026')
    page()
    h('1 Introdução e contexto')
    p('O trabalho avalia redes neurais multicamadas na classificação do rendimento acadêmico em Língua Portuguesa. A identificação de estudantes que podem precisar de apoio é um problema educacional relevante. O experimento utiliza uma base histórica pública; seus resultados medem associação estatística e não comprovam que um atributo cause aprovação ou reprovação.')
    p('O objetivo é comparar ReLU e Tanh com a mesma divisão de dados e arquitetura. A saída indica aprovação quando G3 é maior ou igual a 10, em uma escala de 0 a 20. Trata-se de uma definição operacional do experimento, não de uma regra geral para instituições brasileiras.')
    h('2 Descrição e preparação do dataset')
    p(f"Foi utilizado student-por.csv, da base Student Performance de Paulo Cortez, publicada pela UCI. O arquivo possui {a['registros']} registros e {a['colunas']} colunas: 30 características contextuais e três notas. A coleta ocorreu em duas escolas portuguesas, com questionários e registros escolares. A referência do estudo é de 2008 e a página registra depósito em 2014. Segundo Cortez e Silva (2008), os dados foram coletados no ano letivo de 2005–2006, em duas escolas públicas da região do Alentejo, Portugal. Esse período é distinto do ano da publicação e da data de depósito no repositório.")
    p(f"A inspeção encontrou {a['ausentes']} valores ausentes e {a['duplicados']} linhas integralmente duplicadas. Não houve imputação ou exclusão. A classe aprovação contém {a['aprovados']} registros e a classe reprovação contém {a['reprovados']}. A base de Matemática não foi concatenada, evitando a repetição de estudantes que cursaram ambas as disciplinas.")
    p('Foram selecionadas 12 variáveis: age, Medu, Fedu, traveltime, studytime, failures, school, schoolsup, famsup, paid, higher e internet. Idade e reprovações anteriores contextualizam a trajetória escolar; escolaridade parental e acesso à internet representam condições de estudo; deslocamento e tempo de estudo descrevem a rotina; apoios escolares, familiares e aulas extras representam recursos disponíveis; escola e intenção de ensino superior descrevem o contexto acadêmico. Essas justificativas são hipóteses de relevância, não conclusões de causalidade.')
    p('G3 é usado somente para criar o alvo. G1 e G2 foram excluídos para estudar a previsão sem notas parciais; elas não seriam vazamento se a previsão fosse feita após sua divulgação, mas mudariam o momento e o objetivo da aplicação. O total de faltas também foi excluído porque não há recorte temporal que garanta sua disponibilidade antecipada. Sexo, saúde, álcool e outras informações pessoais não integram o conjunto escolhido. Ainda assim, atributos contextuais podem carregar vieses.')
    page()
    h('3 Metodologia')
    p(f"A divisão estratificada, com semente 42, produziu {a['treino']} registros de treino, {a['validacao']} de validação e {a['teste']} de teste, aproximadamente 60%, 20% e 20%. O StandardScaler aplica z = (x - média do treino) / desvio do treino às seis variáveis numéricas. Variáveis ordinais são tratadas como números, uma simplificação que pressupõe intervalos comparáveis. As seis categóricas usam one-hot encoding. O ajuste ocorre somente no treino; validação e teste recebem a mesma transformação, resultando em {a['entradas']} entradas.")
    p('A arquitetura possui três camadas ocultas, de 32, 16 e 8 neurônios, seguidas de uma saída sigmoid. Todas as camadas ocultas usam ReLU ou Tanh na respectiva configuração. A sigmoid produz a probabilidade estimada de aprovação, convertida em classe pelo limiar fixo de 0,5. O treinamento utiliza TensorFlow/Keras, Adam e entropia cruzada binária.')
    p('Foram testadas taxas de aprendizado 0,001 e 0,0003 para cada ativação. A segunda proporciona atualizações menores. O batch de 32 equilibra número de atualizações e variabilidade dos gradientes em uma base pequena. O limite é de 120 épocas, com parada após 15 épocas sem redução da perda de validação e restauração dos pesos da melhor época. Esses valores constituem uma busca pequena e justificada, sem alegação de ótimo global.')
    p('A cada atualização de lote é exibido o vetor residual y - probabilidade antes da atualização dos pesos. Cada componente é salvo com época, lote e linha original em erros_*.csv. Esse vetor não é o mesmo que a perda escalar: a entropia cruzada é a função otimizada. Ao final de cada época são registradas perda e acurácia de treino e validação.')
    p('A taxa é escolhida por menor perda de validação dentro de cada ativação, e a configuração final também é selecionada pela validação. Somente depois o teste é avaliado para comparação final. Não se ajusta o limiar com o teste. A referência ingênua sempre prevê a classe majoritária do treino. Acurácia balanceada, F1 macro e recall da reprovação complementam a acurácia para revelar falhas na classe minoritária.')
    page()
    h('4 Resultados obtidos')
    for r in runs:
        p(f"{r['nome']}: {r['epocas_executadas']} épocas executadas; melhor época {r['melhor_epoca']}; perda de validação {r['perda_validacao']:.4f}.")
    p('Configuração selecionada antes de examinar o teste: ' + selected + '.')
    for _, r in metrics.iterrows():
        p(f'{r.modelo}: acurácia {r.acuracia:.2%}; acurácia balanceada {r.acuracia_balanceada:.2%}; F1 macro {r.f1_macro:.4f}; recall da reprovação {r.recall_reprovacao:.2%}; ROC AUC {r.auc:.4f}.')
    p('A baseline constante tem AUC 0,5 por não ordenar os estudantes por risco; sua perda não é usada para seleção. As métricas completas estão em metricas_teste.csv.')
    pic('curvas.png', 16, 11.2)
    center('Figura 1 - Perda e acurácia por época. Fonte: execução do experimento.')
    page()
    h('4.1 Matrizes de confusão')
    pic('matriz_relu.png', 12, 9.6)
    center('Figura 2 - ReLU. Fonte: execução do experimento.')
    pic('matriz_tanh.png', 12, 9.6)
    center('Figura 3 - Tanh. Fonte: execução do experimento.')
    page()
    h('5 Discussão dos resultados e limitações')
    chosen = metrics.loc[metrics.modelo == selected].iloc[0]
    baseline = metrics.iloc[-1]
    delta = chosen.acuracia_balanceada - baseline.acuracia_balanceada
    p(f'O modelo selecionado apresentou diferença de {delta * 100:+.2f} pontos percentuais de acurácia balanceada em relação à baseline. Seu recall de reprovação foi {chosen.recall_reprovacao:.2%}, indicando a fração dos estudantes reprovados corretamente identificados. Uma acurácia alta, isoladamente, não demonstra utilidade para apoio pedagógico quando o modelo falha em reconhecer a reprovação.')
    total_reprovados = None
    for activation in ['relu', 'tanh']:
        run = min([r for r in runs if r['ativacao'] == activation], key=lambda r: r['perda_validacao'])
        prev = pd.read_csv(OUT / f"previsoes_{run['nome']}.csv")
        total_reprovados = int((prev.real == 0).sum())
        reconhecidos = int(((prev.real == 0) & (prev.previsto == 0)).sum())
        met = metrics.loc[metrics.modelo == run['nome']].iloc[0]
        p(f"No teste, {run['nome']} reconheceu {reconhecidos} dos {total_reprovados} reprovados. Sua acurácia geral foi {met.acuracia:.2%}, contra {baseline.acuracia:.2%} da baseline majoritária. A baseline não reconhece reprovados nesta partição. Essa comparação mostra a importância de avaliar a classe minoritária além do número total de acertos.")
        hist = pd.read_csv(OUT / f"historico_{run['nome']}.csv")
        best = hist.loc[hist.epoca == run['melhor_epoca']].iloc[0]
        last = hist.iloc[-1]
        p(f"Nas curvas de {run['nome']}, a menor perda de validação ocorreu na época {int(best.epoca)}, com {best.perda_validacao:.4f}. Na época final {int(last.epoca)}, ela foi {last.perda_validacao:.4f}; no mesmo intervalo, a perda de treino passou de {best.perda_treino:.4f} para {last.perda_treino:.4f}. A redução do erro de treino sem melhora da validação é compatível com sobreajuste após a melhor época. Os resultados de teste usam os pesos restaurados da melhor época, e não os da última.")
    p('A Tanh identificou mais reprovados nesta execução, mas isso não altera retrospectivamente a seleção feita pela perda de validação. As diferenças entre ativações referem-se a esta partição e semente; não demonstram superioridade geral ou significância estatística. A calibração e eventual escolha de outro limiar devem ser estudadas na validação, sem reutilizar o teste para ajustar decisões.')
    p('O estudo tem limitações: amostra pequena, apenas duas escolas de outro país, dados históricos, variáveis autorrelatadas e ausência de validação externa ou temporal. A base não assegura que todas as respostas contextuais já estivessem disponíveis no início do ano, portanto o experimento não valida um sistema de alerta antecipado em produção. A exclusão de atributos sensíveis não garante equidade; condições familiares podem funcionar como indicadores indiretos.')
    page()
    h('6 Conclusão')
    p(f'O experimento implementou uma MLP com três camadas ocultas e comparou duas ativações e duas taxas. A seleção pela validação indicou {selected}. A avaliação no teste fornece uma estimativa limitada do desempenho nessa base, sem autorizar decisões automáticas sobre estudantes. A atividade evidencia a importância de separar os dados antes do pré-processamento, registrar os erros e comparar modelos com uma baseline.')
    p('Como melhorias, propõem-se validação cruzada com múltiplas sementes, comparação com regressão logística, auditoria de desempenho por grupos, calibração de probabilidades e coleta recente com datas de disponibilidade dos atributos. Qualquer aplicação deve apoiar a análise de educadores e oferecer ajuda, sem substituir sua avaliação.')
    page()
    references = True
    h('Referências')
    p('CORTEZ, Paulo; SILVA, Alice. Using data mining to predict secondary school student performance. In: FIFTH ANNUAL FUTURE BUSINESS TECHNOLOGY CONFERENCE, 2008, Porto. Proceedings [...]. EUROSIS, 2008. p. 5–12. Texto consultado em: https://www.academia.edu/144478255/Using_data_mining_to_predict_secondary_school_student_performance. Acesso em: 16 nov. 2026.')
    p('CORTEZ, Paulo. Student Performance [Dataset]. UCI Machine Learning Repository, 2008. DOI: 10.24432/C5TG7T. Disponível em: https://archive.ics.uci.edu/dataset/320/student-performance. Acesso em: 16 nov. 2026. Licença CC BY 4.0.')
    p('TENSORFLOW. tf.keras.Sequential. Disponível em: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential. Acesso em: 16 nov. 2026.')
    p('TENSORFLOW. tf.keras.callbacks.EarlyStopping. Disponível em: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping. Acesso em: 16 nov. 2026.')

    def numbering(canvas, doc):
        if doc.page >= 3:
            canvas.setFont('Times-Roman', 10)
            canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 2 * cm, str(doc.page - 1))
    SimpleDocTemplate(str(OUT / 'Relatorio_Rendimento_Academico.pdf'), pagesize=A4, leftMargin=3 * cm, rightMargin=2 * cm, topMargin=3 * cm, bottomMargin=2 * cm, title='Predição do rendimento acadêmico com redes neurais multicamadas', author=AUTORES).build(story, onFirstPage=numbering, onLaterPages=numbering)

def atualizar_relatorio():
    a = json.loads((OUT / 'auditoria.json').read_text(encoding='utf-8'))
    runs = pd.read_csv(OUT / 'comparacao_validacao.csv').to_dict('records')
    metrics = pd.read_csv(OUT / 'metricas_teste.csv')
    selected = json.loads((OUT / 'versoes.json').read_text(encoding='utf-8'))['modelo_selecionado']
    gerar_relatorio(a, runs, metrics, selected)
    print('Relatório atualizado sem repetir o treinamento.')

if __name__ == '__main__': atualizar_relatorio()
