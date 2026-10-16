"""Final exploratory consensus result, keeping prior hypothesis outcomes."""
import experiment as e
import numpy as np
def main():
    h=e.HERE;result=e.r.read(h/'consensus/assessment.json');a=result['results']['baseline'];b=result['results']['consensus']
    days=sorted({r['file'][:8] for r in result['rows']['baseline']});rng=np.random.default_rng(20261016);deltas=[]
    grouped={k:{day:[r for r in rows if r['file'][:8]==day] for day in days} for k,rows in result['rows'].items()}
    for _ in range(2000):
        draw=rng.choice(days,len(days),replace=True);v={k:e.totals([r for day in draw for r in grouped[k][day]]) for k in grouped}
        deltas.append([v['consensus']['pq']-v['baseline']['pq'],v['consensus']['thin_recall']-v['baseline']['thin_recall']])
    ci=np.quantile(deltas,[.025,.975],axis=0).tolist();better=b['pq']>=a['pq'] and b['thin_recall']>a['thin_recall']
    summary={'baseline':a,'consensus':b,'pq_change':b['pq']-a['pq'],'thin_recall_change':b['thin_recall']-a['thin_recall'],'paired_date_bootstrap_95_interval_pq_and_thin_recall':ci,'assessment_images':106,'assessment_dates':95,'independent_blind':False,'third_exploratory_hypothesis':True,'meets_local_pq_and_thin_recall_criterion':better,'submitted':False}
    e.r.write(h/'summary-final.json',summary)
    text='Solar — końcowy wynik lokalnych prób, 16.10.2026\nAutor: Jacek Adamek\n\n'
    text+=('Wariant z potwierdzeniem przez oba modele poprawił lokalne PQ i czułość cienkich obiektów. Zmiana jest niewielka i wymaga dalszej walidacji.\n' if better else 'Nie uzyskano jednoczesnej poprawy lokalnego PQ i czułości cienkich obiektów. Zachowano dotychczasowy model jako domyślny.\n')
    for name,v in [('Dotychczasowy model',a),('Cienkie propozycje z potwierdzeniem RF >= 0.6',b)]:
        text+=f"\n{name}: PQ {v['pq']:.6f}; TP {v['tp']}, FP {v['fp']}, FN {v['fn']}; cienkie wykryte {v['thin_tp']}/{v['thin_tp']+v['thin_fn']} ({100*v['thin_recall']:.2f}%).\n"
    text+=f"\n95% bootstrap po datach: zmiana PQ [{ci[0][0]:+.6f}, {ci[1][0]:+.6f}]; zmiana czułości cienkich [{100*ci[0][1]:+.2f}, {100*ci[1][1]:+.2f}] pkt proc.\n"
    text+='\nWariant trzeci dodaje wyłącznie odłączone, cienkie propozycje U-Net z progiem 0.7, wydłużeniem >=2 i szerokością <=4 piksele przy 1024. Średnie wsparcie RF musi wynosić >=0.6. Próg wsparcia wybrano spośród siedmiu wartości na 19 zdjęciach strojenia, przed trzecią oceną na 106 zdjęciach.\n'
    text+='\nTo trzecia hipoteza rozwojowa, podjęta po poznaniu zbiorczych wyników drugiej. Dane oceny były używane wcześniej. Wzrost na tych danych nie jest niezależnym potwierdzeniem skuteczności ani nowym wynikiem Kaggle. Nie zmieniono domyślnego modelu, opublikowanych plików ani zgłoszenia. Nowa funkcja inferencji jest oddzielnie dostępna w consensus_runtime.py.\n\nHistoria pierwszych dwóch hipotez i definicja cienkiego obiektu:\n\n'
    earlier=(h/'Wyniki-Solar.txt').read_text(encoding='utf-8');backup=h/'Wyniki-Solar-przed-konsensusem.txt'
    if backup.exists():raise FileExistsError('Preserve report history')
    backup.write_text(earlier,encoding='utf-8');(h/'Wyniki-Solar.txt').write_text(text+earlier,encoding='utf-8');print(summary,flush=True)
if __name__=='__main__':main()
