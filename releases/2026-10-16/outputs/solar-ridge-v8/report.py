"""Summarize frozen full-resolution development results and uncertainty."""
import ridge as g
import numpy as np
r=g.r;H=g.HERE;e=g.e
def main():
    result=r.read(H/'assessment.json');base=result['results']['baseline'];key=next(k for k in result['results'] if k!='baseline');new=result['results'][key]
    days=sorted({row['file'][:8] for row in result['rows']['baseline']});groups={k:{day:[v for v in rows if v['file'][:8]==day] for day in days} for k,rows in result['rows'].items()};rng=np.random.default_rng(20261016);deltas=[]
    for _ in range(2000):
        sample=rng.choice(days,len(days),replace=True);values={k:e.totals([row for day in sample for row in by_day[day]]) for k,by_day in groups.items()}
        a=values['baseline'];b=values[key];deltas.append([b['pq']-a['pq'],b['thin_recall']-a['thin_recall']])
    ci=np.quantile(deltas,[.025,.975],axis=0).tolist();meets=new['thin_tp']>base['thin_tp'] and new['fp']<=base['fp'] and new['tp']>=base['tp'] and new['pq']>=base['pq']
    summary={'baseline':base,'ridge_refinement':new,'meets_local_thin_no_extra_fp_criterion':meets,'pq_change':new['pq']-base['pq'],'thin_recall_change':new['thin_recall']-base['thin_recall'],'paired_date_bootstrap_95_pq_and_thin_recall':ci,'assessment_images':106,'assessment_dates':len(days),'independent_blind':False,'submitted_or_published':False}
    r.write(H/'summary.json',summary)
    text='Solar — pełna rozdzielczość i cechy cienkich grzbietów, 16.10.2026\nAutor: Jacek Adamek\n\n'+('Wariant spełnił lokalne kryterium: więcej cienkich trafień bez dodatkowych FP i bez spadku ogólnego TP/PQ. Nie stanowi to niezależnego potwierdzenia generalizacji.\n' if meets else 'Nie uzyskano jednocześnie większej liczby cienkich trafień bez dodatkowych FP i bez spadku TP/PQ na ocenie. Nie zastąpiono bazowego modelu.\n')
    for name,v in [('Dotychczasowy RF z filtrem',base),('Nowa korekta pełnej rozdzielczości',new)]:text+=f"\n{name}: PQ {v['pq']:.6f}; TP {v['tp']}, FP {v['fp']}, FN {v['fn']}; cienkie trafienia {v['thin_tp']}/314 ({100*v['thin_recall']:.2f}%).\n"
    text+=f"\n95% opisowy bootstrap po {len(days)} datach: zmiana PQ [{ci[0][0]:+.6f}, {ci[1][0]:+.6f}], zmiana czułości cienkich [{100*ci[0][1]:+.2f}, {100*ci[1][1]:+.2f}] pkt proc. Przedział nie usuwa obciążenia po wielokrotnym doborze hipotez i używaniu danych.\n"
    text+='\nInne podejście: osobny model pikselowy ExtraTrees, 10 cech obrazu w oryginalnym 2048x2048, w tym dodatnia odpowiedź grzbietów z hesjanu przy trzech skalach. Uczenie na 60 deterministycznie wybranych zdjęciach z pierwotnej puli treningowej, 335163 próbkach; nadpróbkowanie cienkich masek wyłącznie w uczeniu. Cel: średni głos autorów masek segmentacyjnych, bez pozostałych metadanych adnotacji.\n'
    text+='\nDobór: 27 konfiguracji na 19 zdjęciach strojenia. Przed oceną zamrożono szerokość <=16 pikseli, próg nowego modelu 0.2, brak wzrostu poza istniejącą maskę (promień 0). Na strojeniu cienkie trafienia 15/49 ->16/49, łącznie TP83 i FP121 bez zmian, PQ 0.263291 ->0.263770.\n'
    text+='\nNowy model koryguje obrysy istniejących instancji. Nie tworzy nowych identyfikatorów. Ochrona powierzchni zachowuje maskę, jeżeli korekta usunęłaby ponad 75% obiektu, zostawiła mniej niż 8 pikseli lub ponad dwukrotnie go powiększyła. To różni się od wcześniejszego dodawania odłączonych propozycji i geometrycznej erozji.\n'
    text+='\nWeryfikacja: testy odpowiedzi na ciemną linię w trzech szerokościach oraz ochrony instancji. Rzeczywista inferencja jest dokładnie zgodna z obliczeniem badawczym po zablokowaniu dostępu do adnotacji i pamięci map prawdopodobieństw. Diagnostyczna pamięć float16 nie służy do oceniania ani inferencji; obliczenia prawdopodobieństw są float32.\n'
    text+='\nOgraniczenia: 106 zdjęć z 95 dat oceniano już wcześniej; to rozwój na znanych danych, nie niezależny ślepy test. Cienki obiekt jest zdefiniowany geometrycznie jak wcześniej (2*powierzchnia/piksele granicy <=8 i rozpiętość >=40 przy 2048), tylko na potrzeby oceny. To wynik lokalnego publicznego skryptu, nie nowy wynik Kaggle. Model domyślny, poprzednie wyniki i zgłoszone CSV pozostały bez zmian. Niczego nie opublikowano ani nie wysłano.\n'
    (H/'Wyniki-Solar.txt').write_text(text,encoding='utf-8');print(summary,flush=True)
if __name__=='__main__':main()
