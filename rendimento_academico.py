"""Atividade 02 - projeto para VS Code com Python e requirements.txt.
Nenhum resultado é previamente preenchido: métricas e relatório vêm da execução.
"""
from pathlib import Path
import io, json, zipfile, urllib.request, platform, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                            recall_score, roc_auc_score, confusion_matrix,
                            ConfusionMatrixDisplay, log_loss)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4
from xml.sax.saxutils import escape

# Identificação do grupo, separada do código do modelo.
BASE = Path(__file__).resolve().parent
with (BASE / 'config.json').open(encoding='utf-8') as arquivo:
    config = json.load(arquivo)
AUTORES = config['autores']
INSTITUICAO = config['instituicao']
CURSO = config['curso']
CIDADE = config['cidade']
SEED, MAX_EPOCHS, BATCH = 42, 120, 32
OUT = BASE / 'entrega_rendimento'; OUT.mkdir(exist_ok=True)
URL = 'https://archive.ics.uci.edu/static/public/320/student%2Bperformance.zip'

def obter_dados():
    """Preserva os bytes originais do CSV e a documentação do autor."""
    target = OUT / 'student-por.csv'
    if not target.exists():
        raw = urllib.request.urlopen(URL, timeout=60).read()
        outer = zipfile.ZipFile(io.BytesIO(raw))
        # O pacote da UCI pode conter student.zip ou os arquivos diretamente.
        if 'student.zip' in outer.namelist():
            outer = zipfile.ZipFile(io.BytesIO(outer.read('student.zip')))
        for basename in ['student-por.csv', 'student.txt']:
            matches = [n for n in outer.namelist() if Path(n).name == basename]
            if matches:
                (OUT / basename).write_bytes(outer.read(matches[0]))
        if not target.exists():
            raise RuntimeError('CSV não encontrado no pacote UCI. Verifique a fonte.')
    return pd.read_csv(target, sep=';')

def metricas(y, p):
    pred = (p >= .5).astype(int)
    return dict(acuracia=float(accuracy_score(y, pred)),
                acuracia_balanceada=float(balanced_accuracy_score(y, pred)),
                f1_macro=float(f1_score(y, pred, average='macro')),
                recall_reprovacao=float(recall_score(y, pred, pos_label=0)),
                auc=float(roc_auc_score(y, p)),
                perda=float(log_loss(y, np.clip(p, 1e-7, 1-1e-7), labels=[0, 1])))

def main():
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    df = obter_dados()
    # Apenas Português: juntar Matemática duplicaria parte dos estudantes.
    nums = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures']
    cats = ['school', 'schoolsup', 'famsup', 'paid', 'higher', 'internet']
    cols = nums + cats
    assert len(df) == 649 and df.shape[1] == 33, 'Base inesperada: revise antes de continuar.'
    missing, duplicates = int(df.isna().sum().sum()), int(df.duplicated().sum())
    if missing:
        raise ValueError('Foram encontrados dados ausentes. Revisar o tratamento antes de treinar.')
    # Não remover registros só porque compartilham atributos: podem ser pessoas diferentes.
    if duplicates:
        raise ValueError('Foram encontrados registros idênticos. Inspecione a base antes de continuar.')
    y = (df.G3 >= 10).astype('int32').to_numpy()
    ids = np.arange(len(df))
    trainval, test = train_test_split(ids, test_size=.2, random_state=SEED, stratify=y)
    train, val = train_test_split(trainval, test_size=.25, random_state=SEED, stratify=y[trainval])
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    prep = ColumnTransformer([('numeric', StandardScaler(), nums),
                              ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats)])
    xt = prep.fit_transform(df.iloc[train][cols]).astype('float32')
    xv = prep.transform(df.iloc[val][cols]).astype('float32')
    xe = prep.transform(df.iloc[test][cols]).astype('float32')
    pd.DataFrame({'linha_csv': np.r_[train,val,test]+2,
                  'particao': ['treino']*len(train)+['validacao']*len(val)+['teste']*len(test)
                 }).to_csv(OUT/'particoes.csv', index=False)
    # Parâmetros aprendidos apenas no treino, para auditoria do pré-processamento.
    scaler = prep.named_transformers_['numeric']
    pd.DataFrame({'variavel':nums,'media_treino':scaler.mean_,'escala_treino':scaler.scale_}).to_csv(OUT/'padronizacao.csv',index=False)
    import joblib
    joblib.dump(prep, OUT/'preprocessamento.joblib')
    auditoria = dict(registros=len(df), colunas=df.shape[1], ausentes=missing,
                    duplicados=duplicates, aprovados=int(y.sum()), reprovados=int((y==0).sum()),
                    treino=len(train), validacao=len(val), teste=len(test), entradas=int(xt.shape[1]))
    (OUT/'auditoria.json').write_text(json.dumps(auditoria,indent=2),encoding='utf-8')
    print('AUDITORIA:', auditoria)
    lossfn = tf.keras.losses.BinaryCrossentropy()
    runs = []
    # Quatro configurações: duas ativações e duas taxas; seleção só na validação.
    for activation in ['relu','tanh']:
        for lr in [.001,.0003]:
            name = f'{activation}_{lr}'
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(SEED)
            model = tf.keras.Sequential([tf.keras.layers.Input((xt.shape[1],)),
                tf.keras.layers.Dense(32,activation=activation),
                tf.keras.layers.Dense(16,activation=activation),
                tf.keras.layers.Dense(8,activation=activation),
                tf.keras.layers.Dense(1,activation='sigmoid')])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            rng = np.random.default_rng(SEED)
            best_loss, best_weights, wait, best_epoch = np.inf, None, 0, 0
            history, vectors = [], []
            for epoch in range(1, MAX_EPOCHS+1):
                order = rng.permutation(len(train))
                for step, start in enumerate(range(0,len(order),BATCH), 1):
                    batch = order[start:start+BATCH]
                    xb = tf.convert_to_tensor(xt[batch]); yb = tf.convert_to_tensor(y[train][batch,None],dtype=tf.float32)
                    with tf.GradientTape() as tape:
                        pb = model(xb,training=True)
                        loss = lossfn(yb,pb)
                    grads = tape.gradient(loss,model.trainable_variables)
                    optimizer.apply_gradients(zip(grads,model.trainable_variables))
                    # Erro por amostra antes da atualização: r = y - probabilidade.
                    residuals = (yb-pb).numpy().ravel()
                    print(f'{name} epoca={epoch} lote={step} erros={np.round(residuals,4).tolist()}')
                    for local, residual in zip(batch,residuals):
                        vectors.append([epoch,step,int(train[local])+2,float(residual)])
                pt = model(xt,training=False).numpy().ravel()
                pv = model(xv,training=False).numpy().ravel()
                lt = float(lossfn(y[train,None],pt[:,None]).numpy())
                lv = float(lossfn(y[val,None],pv[:,None]).numpy())
                history.append([epoch,lt,lv,accuracy_score(y[train],pt>=.5),accuracy_score(y[val],pv>=.5)])
                if lv < best_loss:
                    best_loss,best_weights,best_epoch,wait = lv,model.get_weights(),epoch,0
                else:
                    wait += 1
                if wait >= 15: break
            model.set_weights(best_weights)
            model.save(OUT/f'modelo_{name}.keras')
            h = pd.DataFrame(history,columns=['epoca','perda_treino','perda_validacao','acuracia_treino','acuracia_validacao'])
            h.to_csv(OUT/f'historico_{name}.csv',index=False)
            pd.DataFrame(vectors,columns=['epoca','lote','linha_csv','erro_y_menos_probabilidade']).to_csv(OUT/f'erros_{name}.csv',index=False)
            runs.append(dict(nome=name,ativacao=activation,taxa=lr,melhor_epoca=best_epoch,
                             epocas_executadas=epoch,perda_validacao=best_loss))
    # Congelar a escolha antes de examinar qualquer métrica do teste.
    selected = min(runs,key=lambda r:r['perda_validacao'])['nome']
    selected_activation_runs = [min([r for r in runs if r['ativacao']==a],key=lambda r:r['perda_validacao']) for a in ['relu','tanh']]
    results = []
    for r in selected_activation_runs:
        model = tf.keras.models.load_model(OUT/f"modelo_{r['nome']}.keras",compile=False)
        p = model(xe,training=False).numpy().ravel()
        results.append(dict(modelo=r['nome'],**metricas(y[test],p)))
        pd.DataFrame({'linha_csv':test+2,'real':y[test],'prob_aprovacao':p,'previsto':(p>=.5).astype(int)}).to_csv(OUT/f"previsoes_{r['nome']}.csv",index=False)
        fig,ax=plt.subplots(figsize=(5,4))
        ConfusionMatrixDisplay(confusion_matrix(y[test],p>=.5),display_labels=['Reprovado','Aprovado']).plot(ax=ax,colorbar=False,cmap='Blues')
        ax.set_xlabel('Classe prevista'); ax.set_ylabel('Classe real'); ax.set_title(r['nome'])
        fig.tight_layout(); fig.savefig(OUT/f"matriz_{r['ativacao']}.png",dpi=160); plt.close(fig)
    majority = int(np.mean(y[train]) >= .5)
    baseprob = np.full(len(test), .999999 if majority else .000001)
    results.append(dict(modelo='Baseline majoritária',**metricas(y[test],baseprob)))
    pd.DataFrame(runs).to_csv(OUT/'comparacao_validacao.csv',index=False)
    metrics=pd.DataFrame(results); metrics.to_csv(OUT/'metricas_teste.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for j,r in enumerate(selected_activation_runs):
        h=pd.read_csv(OUT/f"historico_{r['nome']}.csv")
        for i,metric in enumerate(['perda','acuracia']):
            for split in ['treino','validacao']:
                axes[i,j].plot(h.epoca,h[f'{metric}_{split}'],label=split)
            axes[i,j].set_title(f"{r['nome']} - {metric}"); axes[i,j].set_xlabel('Época'); axes[i,j].legend()
    fig.tight_layout(); fig.savefig(OUT/'curvas.png',dpi=170); plt.close(fig)
    (OUT/'versoes.json').write_text(json.dumps(dict(python=platform.python_version(),tensorflow=tf.__version__,pandas=pd.__version__,numpy=np.__version__,semente=SEED,modelo_selecionado=selected),indent=2),encoding='utf-8')
    gerar_relatorio(auditoria,runs,metrics,selected)
    print('\nRESULTADOS REAIS\n',metrics.to_string(index=False))
    print('Modelo escolhido pela validação:',selected)
    source=Path(__file__).resolve()
    if source.exists(): shutil.copy2(source,OUT/source.name)
    
    for nome in ['config.json', 'requirements.txt', 'README.md']:
        shutil.copy2(BASE / nome, OUT / nome)
    shutil.make_archive(str(BASE / 'Entrega_Rendimento_Academico'),'zip',OUT)
    print('Concluído: Entrega_Rendimento_Academico.zip. Revise o PDF antes de enviar.')

def gerar_relatorio(a,runs,metrics,selected):
    """PDF gerado apenas após treinamento e avaliação; revisar visualmente no Colab."""
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CorpoABNT',fontName='Times-Roman',fontSize=12,leading=18,
                              alignment=TA_JUSTIFY,firstLineIndent=1.25*cm,spaceAfter=6))
    styles.add(ParagraphStyle(name='CentroABNT',fontName='Times-Roman',fontSize=12,leading=18,alignment=TA_CENTER,spaceAfter=12))
    story=[]
    references = False
    styles.add(ParagraphStyle(name='Referencia',fontName='Times-Roman',fontSize=12,leading=12,alignment=0,spaceAfter=12,wordWrap='LTR'))
    def p(t): story.append(Paragraph(escape(t),styles['Referencia' if references else 'CorpoABNT']))
    def h(t): story.append(Paragraph(escape(t),styles['Heading2']))
    def center(t): story.append(Paragraph(escape(t),styles['CentroABNT']))
    def page(): story.append(PageBreak())
    def pic(name,w,hgt): story.append(Image(str(OUT/name),width=w*cm,height=hgt*cm))
    center(INSTITUICAO); center(CURSO); story.append(Spacer(1,2*cm)); center(AUTORES)
    story.append(Spacer(1,3*cm)); center('PREDIÇÃO DO RENDIMENTO ACADÊMICO COM REDES NEURAIS MULTICAMADAS')
    story.append(Spacer(1,5*cm)); center(CIDADE); center('2026'); page()
    center(AUTORES); story.append(Spacer(1,2*cm)); center('PREDIÇÃO DO RENDIMENTO ACADÊMICO COM REDES NEURAIS MULTICAMADAS')
    p('Relatório técnico apresentado à disciplina Sistemas Inteligentes, ministrada pela professora Malga Costa, como parte da Atividade 02 de Inteligência Artificial Aplicada a Dados Reais.'); story.append(Spacer(1,7*cm)); center(CIDADE); center('2026'); page()
    h('1 Introdução e contexto')
    p('O trabalho avalia redes neurais multicamadas na classificação do rendimento acadêmico em Língua Portuguesa. A identificação de estudantes que podem precisar de apoio é um problema educacional relevante. O experimento utiliza uma base histórica pública; seus resultados medem associação estatística e não comprovam que um atributo cause aprovação ou reprovação.')
    p('O objetivo é comparar ReLU e Tanh com a mesma divisão de dados e arquitetura. A saída indica aprovação quando G3 é maior ou igual a 10, em uma escala de 0 a 20. Trata-se de uma definição operacional do experimento, não de uma regra geral para instituições brasileiras.')
    h('2 Descrição e preparação do dataset')
    p(f"Foi utilizado student-por.csv, da base Student Performance de Paulo Cortez, publicada pela UCI. O arquivo possui {a['registros']} registros e {a['colunas']} colunas: 30 características contextuais e três notas. A coleta ocorreu em duas escolas portuguesas, com questionários e registros escolares. A referência do estudo é de 2008 e a página registra depósito em 2014. Segundo Cortez e Silva (2008), os dados foram coletados no ano letivo de 2005–2006, em duas escolas públicas da região do Alentejo, Portugal. Esse período é distinto do ano da publicação e da data de depósito no repositório.")
    p(f"A inspeção encontrou {a['ausentes']} valores ausentes e {a['duplicados']} linhas integralmente duplicadas. Não houve imputação ou exclusão. A classe aprovação contém {a['aprovados']} registros e a classe reprovação contém {a['reprovados']}. A base de Matemática não foi concatenada, evitando a repetição de estudantes que cursaram ambas as disciplinas.")
    p('Foram selecionadas 12 variáveis: age, Medu, Fedu, traveltime, studytime, failures, school, schoolsup, famsup, paid, higher e internet. Idade e reprovações anteriores contextualizam a trajetória escolar; escolaridade parental e acesso à internet representam condições de estudo; deslocamento e tempo de estudo descrevem a rotina; apoios escolares, familiares e aulas extras representam recursos disponíveis; escola e intenção de ensino superior descrevem o contexto acadêmico. Essas justificativas são hipóteses de relevância, não conclusões de causalidade.')
    p('G3 é usado somente para criar o alvo. G1 e G2 foram excluídos para estudar a previsão sem notas parciais; elas não seriam vazamento se a previsão fosse feita após sua divulgação, mas mudariam o momento e o objetivo da aplicação. O total de faltas também foi excluído porque não há recorte temporal que garanta sua disponibilidade antecipada. Sexo, saúde, álcool e outras informações pessoais não integram o conjunto escolhido. Ainda assim, atributos contextuais podem carregar vieses.')
    page(); h('3 Metodologia')
    p(f"A divisão estratificada, com semente 42, produziu {a['treino']} registros de treino, {a['validacao']} de validação e {a['teste']} de teste, aproximadamente 60%, 20% e 20%. O StandardScaler aplica z = (x - média do treino) / desvio do treino às seis variáveis numéricas. Variáveis ordinais são tratadas como números, uma simplificação que pressupõe intervalos comparáveis. As seis categóricas usam one-hot encoding. O ajuste ocorre somente no treino; validação e teste recebem a mesma transformação, resultando em {a['entradas']} entradas.")
    p('A arquitetura possui três camadas ocultas, de 32, 16 e 8 neurônios, seguidas de uma saída sigmoid. Todas as camadas ocultas usam ReLU ou Tanh na respectiva configuração. A sigmoid produz a probabilidade estimada de aprovação, convertida em classe pelo limiar fixo de 0,5. O treinamento utiliza TensorFlow/Keras, Adam e entropia cruzada binária.')
    p('Foram testadas taxas de aprendizado 0,001 e 0,0003 para cada ativação. A segunda proporciona atualizações menores. O batch de 32 equilibra número de atualizações e variabilidade dos gradientes em uma base pequena. O limite é de 120 épocas, com parada após 15 épocas sem redução da perda de validação e restauração dos pesos da melhor época. Esses valores constituem uma busca pequena e justificada, sem alegação de ótimo global.')
    p('A cada atualização de lote é exibido o vetor residual y - probabilidade antes da atualização dos pesos. Cada componente é salvo com época, lote e linha original em erros_*.csv. Esse vetor não é o mesmo que a perda escalar: a entropia cruzada é a função otimizada. Ao final de cada época são registradas perda e acurácia de treino e validação.')
    p('A taxa é escolhida por menor perda de validação dentro de cada ativação, e a configuração final também é selecionada pela validação. Somente depois o teste é avaliado para comparação final. Não se ajusta o limiar com o teste. A referência ingênua sempre prevê a classe majoritária do treino. Acurácia balanceada, F1 macro e recall da reprovação complementam a acurácia para revelar falhas na classe minoritária.')
    page(); h('4 Resultados obtidos')
    for r in runs:
        p(f"{r['nome']}: {r['epocas_executadas']} épocas executadas; melhor época {r['melhor_epoca']}; perda de validação {r['perda_validacao']:.4f}.")
    p('Configuração selecionada antes de examinar o teste: '+selected+'.')
    for _,r in metrics.iterrows():
        p(f"{r.modelo}: acurácia {r.acuracia:.2%}; acurácia balanceada {r.acuracia_balanceada:.2%}; F1 macro {r.f1_macro:.4f}; recall da reprovação {r.recall_reprovacao:.2%}; ROC AUC {r.auc:.4f}.")
    p('A baseline constante tem AUC 0,5 por não ordenar os estudantes por risco; sua perda não é usada para seleção. As métricas completas estão em metricas_teste.csv.')
    pic('curvas.png',16,11.2); center('Figura 1 - Perda e acurácia por época. Fonte: execução do experimento.'); page()
    h('4.1 Matrizes de confusão')
    pic('matriz_relu.png',12,9.6); center('Figura 2 - ReLU. Fonte: execução do experimento.')
    pic('matriz_tanh.png',12,9.6); center('Figura 3 - Tanh. Fonte: execução do experimento.'); page()
    h('5 Discussão dos resultados e limitações')
    chosen=metrics.loc[metrics.modelo==selected].iloc[0]; baseline=metrics.iloc[-1]
    delta=chosen.acuracia_balanceada-baseline.acuracia_balanceada
    p(f"O modelo selecionado apresentou diferença de {delta*100:+.2f} pontos percentuais de acurácia balanceada em relação à baseline. Seu recall de reprovação foi {chosen.recall_reprovacao:.2%}, indicando a fração dos estudantes reprovados corretamente identificados. Uma acurácia alta, isoladamente, não demonstra utilidade para apoio pedagógico quando o modelo falha em reconhecer a reprovação.")
    total_reprovados = None
    for activation in ['relu','tanh']:
        run=min([r for r in runs if r['ativacao']==activation],key=lambda r:r['perda_validacao'])
        prev=pd.read_csv(OUT/f"previsoes_{run['nome']}.csv")
        total_reprovados=int((prev.real==0).sum())
        reconhecidos=int(((prev.real==0)&(prev.previsto==0)).sum())
        met=metrics.loc[metrics.modelo==run['nome']].iloc[0]
        p(f"No teste, {run['nome']} reconheceu {reconhecidos} dos {total_reprovados} reprovados. Sua acurácia geral foi {met.acuracia:.2%}, contra {baseline.acuracia:.2%} da baseline majoritária. A baseline não reconhece reprovados nesta partição. Essa comparação mostra a importância de avaliar a classe minoritária além do número total de acertos.")
        hist=pd.read_csv(OUT/f"historico_{run['nome']}.csv")
        best=hist.loc[hist.epoca==run['melhor_epoca']].iloc[0]
        last=hist.iloc[-1]
        p(f"Nas curvas de {run['nome']}, a menor perda de validação ocorreu na época {int(best.epoca)}, com {best.perda_validacao:.4f}. Na época final {int(last.epoca)}, ela foi {last.perda_validacao:.4f}; no mesmo intervalo, a perda de treino passou de {best.perda_treino:.4f} para {last.perda_treino:.4f}. A redução do erro de treino sem melhora da validação é compatível com sobreajuste após a melhor época. Os resultados de teste usam os pesos restaurados da melhor época, e não os da última.")
    p('A Tanh identificou mais reprovados nesta execução, mas isso não altera retrospectivamente a seleção feita pela perda de validação. As diferenças entre ativações referem-se a esta partição e semente; não demonstram superioridade geral ou significância estatística. A calibração e eventual escolha de outro limiar devem ser estudadas na validação, sem reutilizar o teste para ajustar decisões.')
    p('O estudo tem limitações: amostra pequena, apenas duas escolas de outro país, dados históricos, variáveis autorrelatadas e ausência de validação externa ou temporal. A base não assegura que todas as respostas contextuais já estivessem disponíveis no início do ano, portanto o experimento não valida um sistema de alerta antecipado em produção. A exclusão de atributos sensíveis não garante equidade; condições familiares podem funcionar como indicadores indiretos.')
    page(); h('6 Conclusão')
    p(f"O experimento implementou uma MLP com três camadas ocultas e comparou duas ativações e duas taxas. A seleção pela validação indicou {selected}. A avaliação no teste fornece uma estimativa limitada do desempenho nessa base, sem autorizar decisões automáticas sobre estudantes. A atividade evidencia a importância de separar os dados antes do pré-processamento, registrar os erros e comparar modelos com uma baseline.")
    p('Como melhorias, propõem-se validação cruzada com múltiplas sementes, comparação com regressão logística, auditoria de desempenho por grupos, calibração de probabilidades e coleta recente com datas de disponibilidade dos atributos. Qualquer aplicação deve apoiar a análise de educadores e oferecer ajuda, sem substituir sua avaliação.')
    page(); references = True; h('Referências')
    p('CORTEZ, Paulo; SILVA, Alice. Using data mining to predict secondary school student performance. In: FIFTH ANNUAL FUTURE BUSINESS TECHNOLOGY CONFERENCE, 2008, Porto. Proceedings [...]. EUROSIS, 2008. p. 5–12. Texto consultado em: https://www.academia.edu/144478255/Using_data_mining_to_predict_secondary_school_student_performance. Acesso em: 16 nov. 2026.')
    p('CORTEZ, Paulo. Student Performance [Dataset]. UCI Machine Learning Repository, 2008. DOI: 10.24432/C5TG7T. Disponível em: https://archive.ics.uci.edu/dataset/320/student-performance. Acesso em: 16 nov. 2026. Licença CC BY 4.0.')
    p('TENSORFLOW. tf.keras.Sequential. Disponível em: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential. Acesso em: 16 nov. 2026.')
    p('TENSORFLOW. tf.keras.callbacks.EarlyStopping. Disponível em: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping. Acesso em: 16 nov. 2026.')
    def numbering(canvas,doc):
        if doc.page>=3:
            canvas.setFont('Times-Roman',10); canvas.drawRightString(A4[0]-2*cm,A4[1]-2*cm,str(doc.page-1))
    SimpleDocTemplate(str(OUT/'Relatorio_Rendimento_Academico.pdf'),pagesize=A4,
        leftMargin=3*cm,rightMargin=2*cm,topMargin=3*cm,bottomMargin=2*cm,
        title='Predição do rendimento acadêmico com redes neurais multicamadas',author=AUTORES
        ).build(story,onFirstPage=numbering,onLaterPages=numbering)

def atualizar_relatorio():
    # Reutiliza os arquivos da execução; não altera pesos, partições ou métricas.
    a=json.loads((OUT/'auditoria.json').read_text(encoding='utf-8'))
    runs=pd.read_csv(OUT/'comparacao_validacao.csv').to_dict('records')
    metrics=pd.read_csv(OUT/'metricas_teste.csv')
    selected=json.loads((OUT/'versoes.json').read_text(encoding='utf-8'))['modelo_selecionado']
    gerar_relatorio(a,runs,metrics,selected)
    print('Relatório atualizado sem repetir o treinamento.')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--somente-relatorio', action='store_true')
    args=parser.parse_args()
    if args.somente_relatorio: atualizar_relatorio()
    else: main()
