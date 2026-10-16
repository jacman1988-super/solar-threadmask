"""Summarize all development outcomes, including rejected hypotheses."""
import experiment as e
import numpy as np
def main():
    h=e.HERE;result=e.r.read(h/'supplement/assessment.json');base=result['results']['baseline']
    key=next(k for k in result['results'] if k!='baseline');new=result['results'][key]
    days=sorted({x['file'][:8] for x in result['rows']['baseline']});rng=np.random.default_rng(20261016)
    grouped={key2:{day:[r for r in rows if r['file'][:8]==day] for day in days} for key2,rows in result['rows'].items()}
    differences=[]
    for _ in range(2000):
        draw=rng.choice(days,len(days),replace=True)
        a=e.totals([r for day in draw for r in grouped['baseline'][day]]);b=e.totals([r for day in draw for r in grouped[key][day]])
        differences.append([b['pq']-a['pq'],b['thin_recall']-a['thin_recall']])
    interval=np.quantile(differences,[.025,.975],axis=0).tolist()
    summary={'baseline':base,'candidate':new,'pq_gain':new['pq']-base['pq'],'thin_recall_gain':new['thin_recall']-base['thin_recall'],'paired_date_bootstrap_95_interval_pq_and_thin_recall':interval,
       'assessment_images':106,'assessment_dates':len(days),'independent_blind':False,'submitted':False,'recommended_as_default':new['pq']>=base['pq'] and new['thin_recall']>base['thin_recall']}
    e.r.write(h/'summary.json',summary)
    text='Solar — cienkie filamenty, 16.10.2026\nAutor: Jacek Adamek\n\nOcena lokalna: 106 zdjęć, 95 dat, osobne maski każdego autora adnotacji.\n'
    for title,v in [('Dotychczasowy RF z filtrem',base),('RF + cienkie propozycje U-Net',new)]:
        text+=f"\n{title}:\nPQ {v['pq']:.6f}\nTP {v['tp']}, FP {v['fp']}, FN {v['fn']}\nCzułość ogółem {v['recall']:.2%}\nCienkie obiekty: {v['thin_tp']}/{v['thin_tp']+v['thin_fn']}, czułość {v['thin_recall']:.2%}\n"
    text+=f"\nZmiana PQ: {summary['pq_gain']:+.6f}. Zmiana czułości dla cienkich obiektów: {100*summary['thin_recall_gain']:+.2f} pkt proc.\n"
    text+=f"95% przedział bootstrap po datach: PQ [{interval[0][0]:+.6f}, {interval[1][0]:+.6f}], cienkie [{100*interval[0][1]:+.2f}, {100*interval[1][1]:+.2f}] pkt proc. To opis niepewności na używanym wcześniej zbiorze, nie korekta obciążenia selekcji.\n"
    text+='\nMetoda: U-Net900, próg 0.7, wydłużenie >=2, przybliżona szerokość <=4 piksele w 1024. Propozycje muszą być odłączone od dotychczasowych masek i ich marginesu 2 pikseli. Parametry wybrane na 19 zdjęciach przed tą oceną. Maski bazowe pozostają identyczne.\n'
    text+='\nPierwsza hipoteza (histereza oraz trzy nowe filtry jakości) nie spełniła kryteriów na zbiorze strojenia. Zachowano wszystkie 21 prób; brak zastąpienia bazowego modelu tym wariantem. Druga hipoteza ma 18 ustawień sprawdzonych na tym samym zbiorze strojenia. Wynik dodatni na strojeniu: PQ 0.263291 -> 0.265266; cienkie 15/49 -> 16/49.\n'
    text+='\nDefinicja cienkiego obiektu: 2*powierzchnia/liczba pikseli wewnętrznej granicy <=8 oraz dłuższy rozmiar ramki >=40 przy 2048. To geometryczne przybliżenie użyte wyłącznie do oceny na maskach segmentacyjnych, nie wejście modelu.\n'
    text+='\nOgraniczenia: dane oceny były wcześniej oglądane, a hipotezy korzystają wielokrotnie z tego samego strojenia. Wyniki rozwojowe, bez niezależnego testu. PQ jest metryką publicznego skryptu organizatora, nie nowym wynikiem Kaggle. Nie zmieniono CSV zgłoszenia ani nie wykonano publikacji.\n'
    text+='\nWdrożenie lokalne: '+('wariant spełnił kryterium poprawy PQ i czułości cienkich obiektów; dostępny osobno w thin_runtime.py.' if summary['recommended_as_default'] else 'wariant pozostaje eksperymentalny; zachowaj obecny RF jako domyślny z powodu kompromisu jakości.')+'\n'
    (h/'Wyniki-Solar.txt').write_text(text,encoding='utf-8');print(summary,flush=True)
if __name__=='__main__':main()
