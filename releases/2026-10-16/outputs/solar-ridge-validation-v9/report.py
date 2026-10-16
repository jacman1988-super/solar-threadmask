"""Additional-date evidence without tuning the frozen candidate."""
import evaluate_extra as v
import numpy as np
r=v.r;e=v.e;H=v.H
def meets(a,b):return b['thin_tp']>a['thin_tp'] and b['fp']<=a['fp'] and b['tp']>=a['tp'] and b['pq']>=a['pq']
def main():
    result=r.read(H/'assessment.json');assert result['complete'] and result['images_evaluated']==40
    base=result['results']['baseline'];new=result['results']['ridge'];original=r.read(v.MODEL/'assessment.json');oldkey=next(k for k in original['rows'] if k!='baseline')
    combined_rows={'baseline':original['rows']['baseline']+result['rows']['baseline'],'ridge':original['rows'][oldkey]+result['rows']['ridge']};combined={k:e.totals(rows) for k,rows in combined_rows.items()}
    days=sorted({row['file'][:8] for row in result['rows']['baseline']});grouped={k:{day:[row for row in rows if row['file'][:8]==day] for day in days} for k,rows in result['rows'].items()};rng=np.random.default_rng(20261016);deltas=[]
    for _ in range(2000):
        draw=rng.choice(days,len(days),replace=True);values={k:e.totals([row for day in draw for row in group[day]]) for k,group in grouped.items()};a=values['baseline'];b=values['ridge'];deltas.append([b['pq']-a['pq'],b['thin_recall']-a['thin_recall']])
    interval=np.quantile(deltas,[.025,.975],axis=0).tolist();summary={'additional_40':{'baseline':base,'ridge':new,'meets_thin_no_extra_fp_criterion':meets(base,new)},'combined_146_images_135_dates':combined,'combined_meets_criterion':meets(combined['baseline'],combined['ridge']),'additional_date_bootstrap95_pq_thin_recall':interval,'no_parameter_changes':True,'fresh_blind':False,'published_or_submitted':False}
    r.write(H/'summary.json',summary)
    deltas=[]
    for name in r.read(H/'protocol.json')['images']:
        a=e.totals([row for row in result['rows']['baseline'] if row['file']==name]);b=e.totals([row for row in result['rows']['ridge'] if row['file']==name]);deltas.append({'file':name,'pq_delta':b['pq']-a['pq'],'thin_tp_delta':b['thin_tp']-a['thin_tp'],'fp_delta':b['fp']-a['fp'],'total_tp_delta':b['tp']-a['tp']})
    r.write(H/'per-image-deltas.json',sorted(deltas,key=lambda row:row['pq_delta']))
    text='Solar — dodatkowe 40 dat, 16.10.2026\nAutor: Jacek Adamek\n\nBez uczenia i bez zmiany parametrów sprawdzono zamrożoną korektę pełnej rozdzielczości z etapu v8. Daty są rozłączne ze wszystkimi trzema podziałami obecnego v5 (uczenie, strojenie i wcześniejsza ocena), nie tylko z 60 zdjęciami modelu grzbietów. Starsze eksperymenty korzystały jednak z pozostałej puli; to dodatkowa kontrola rozwojowa, nie całkowicie świeży ślepy test.\n'
    for label,values in [('Dodatkowe 40 zdjęć/40 dat',result['results']),('Łącznie 146 zdjęć/135 dat',combined)]:
        text+='\n'+label+':\n'
        for model,q in values.items():text+=f"{model}: PQ {q['pq']:.6f}; TP {q['tp']}, FP {q['fp']}, FN {q['fn']}; cienkie {q['thin_tp']}/{q['thin_tp']+q['thin_fn']} ({100*q['thin_recall']:.2f}%).\n"
    text+=f"\nWarunek większej liczby cienkich trafień bez dodatkowych FP i bez spadku TP/PQ: dodatkowe 40 — {meets(base,new)}, łącznie — {summary['combined_meets_criterion']}.\n"
    text+=f"\n95% opisowy bootstrap po nowych datach: zmiana PQ [{interval[0][0]:+.6f}, {interval[1][0]:+.6f}], zmiana czułości cienkich [{100*interval[0][1]:+.2f}, {100*interval[1][1]:+.2f}] pkt proc. To nie eliminuje wcześniejszego używania danych i wielokrotnego projektowania metod.\n"
    text+='\nSelekcja: 40 dat według zamrożonego hasha, po jednym obrazie na datę, bez oglądania nowych wyników przed wyborem. Parametry pozostały [16, 0.2, 0]; użyto identycznych wag i cech. Predykcje wykonano przed odczytem masek danego obrazu do oceny. Każdy wynik maski ma zapisany SHA256, a liczba identyfikatorów instancji jest zachowana.\n'
    text+='\nObecny model domyślny, CSV, raporty opublikowane i zgłoszenia są niezmienione. Wynik nie jest wynikiem Kaggle. Niczego nie opublikowano ani nie wysłano.\n'
    (H/'Wyniki-Solar.txt').write_text(text,encoding='utf-8');print(summary,flush=True)
if __name__=='__main__':main()
