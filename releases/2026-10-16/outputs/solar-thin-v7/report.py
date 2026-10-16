"""Report all hypotheses, rejected tuning trials and the unchanged baseline."""
import precision as p
r=p.r;HERE=p.HERE
def main():
    learned=r.read(HERE/'assessment.json');joint=r.read(HERE/'joint/assessment.json');base=learned['results']['baseline'];model=learned['results']['candidate'];calibrated=joint['results']['candidate']
    assert base==joint['results']['baseline']
    hypotheses=[]
    for folder,name in [(HERE,'learned precision filter'),(HERE/'joint','joint rejection/addition'),(HERE/'contour','contour growth'),(HERE/'shrink','contour erosion'),(HERE/'replacement','mask replacement')]:
        frozen=r.read(folder/'frozen.json');hypotheses.append({'name':name,'trials':len(frozen['trials']),'selected_on_tune':frozen['selected'],'evaluated_on_106_images':(folder/'assessment.json').exists()})
    result={'baseline':base,'learned_filter':model,'joint_calibration':calibrated,'hypotheses':hypotheses,'thin_improvement_without_extra_fp_achieved':False,'recommended_model':'existing rf_filter','assessment_images':106,'assessment_dates':95,'independent_blind':False,'diagnostic':r.read(HERE/'missed-diagnostic.json'),'published_or_submitted':False}
    assert model==base
    r.write(HERE/'summary.json',result)
    text='Solar — dalsze próby cienkich filamentów, 16.10.2026\nAutor: Jacek Adamek\n\nNie uzyskano potwierdzonego wariantu zwiększającego wykrywanie cienkich obiektów bez wzrostu liczby fałszywych detekcji. Zachowano dotychczasowy model jako domyślny.\n\nOcena rozwojowa: 106 zdjęć, 95 dat, maski każdego autora oceniane osobno.\n'
    for name,v in [('Bazowy RF z filtrem',base),('Wyuczony filtr dodatkowych cienkich kandydatów',model),('Wspólna kalibracja odrzucania i dodawania',calibrated)]:
        text+=f"\n{name}: PQ {v['pq']:.6f}; TP {v['tp']}, FP {v['fp']}, FN {v['fn']}; cienkie wykryte {v['thin_tp']}/314 ({100*v['thin_recall']:.2f}%).\n"
    text+='\nFiltr jakości: 22 cechy obrazu, geometrii i prawdopodobieństw RF/U-Net; ExtraTrees 128 drzew, głębokość 8, liść >=2. Uczenie na 36 zdjęciach (3189 masek kandydatów z dwóch progów, 218 dodatnich). Etykieta dodatnia wymaga dopasowania IoU >0.5 w każdym zestawie adnotacji. Dobór progu na osobnych 19 zdjęciach. Na ocenie filtr nie dopuścił żadnego nowego obiektu: uniknął dodatkowych FP, ale nie poprawił czułości.\n'
    text+='\nKalibracja odrzucania/dodawania: wybrano próg bazowego komponentu 0.205 i wsparcie RF 0.6 dla cienkich dodatków. Na strojeniu było więcej cienkich trafień i mniej FP. Na 106 zdjęciach spadła liczba FP, lecz także TP i PQ; liczba cienkich trafień nie wzrosła. Wariant odrzucono.\n'
    text+='\nTrzy kolejne hipotezy nie spełniły kryteriów już na 19 zdjęciach strojenia: poszerzanie cienkich obrysów (27 ustawień), ostrożne zwężanie z ochroną przed usunięciem obiektu (9 ustawień), zastępowanie istniejącej maski lepiej ocenioną maską U-Net (24 ustawienia). Nie uruchamiano oceny na 106 zdjęciach dla niewybranych wariantów.\n'
    text+='\nDiagnostyka: spośród 232 pomijanych cienkich adnotacji U-Net ma poprawne dopasowanie dla 17, lecz pierwotne reguły dodatkowych cienkich i odłączonych kandydatów dopuszczają tylko 1. To uzasadniło próbę zastępowania masek, ale nie stanowi automatycznej poprawy predykcji ani warunku dostępnego podczas inferencji.\n'
    text+='\nDefinicja cienkiego obiektu pozostaje geometryczna: 2*powierzchnia/liczba pikseli wewnętrznej granicy <=8 i dłuższy rozmiar ramki >=40, przy 2048. Użyta wyłącznie do oceny adnotacji. Wszystkie metody inferencji używają wyłącznie obrazu, zapisanych modeli i protokołów.\n'
    text+='\nOgraniczenia: pięć hipotez, wielokrotne użycie danych rozwojowych, projekt zastępowania masek inspirowany diagnostyką na ocenie. Wyniki nie są niezależnym testem, nowym wynikiem Kaggle ani dowodem braku FP w przyszłych obrazach. Nie zmieniono bazowych modeli, CSV ani opublikowanych artefaktów.\n'
    text+='\nKontrola wykonania: testy cech, ochrony masek, erozji, odrzuconych zamian i walidacji odpowiedzi Gemma; osobno dwa porównania rzeczywistej inferencji z obliczeniami badawczymi, przy zabronionym dostępie do adnotacji i pamięci predykcji obrazów.\n'
    (HERE/'Wyniki-Solar.txt').write_text(text,encoding='utf-8');print(result,flush=True)
if __name__=='__main__':main()
