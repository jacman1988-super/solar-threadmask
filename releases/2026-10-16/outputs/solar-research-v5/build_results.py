import sys,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
root=Path.cwd();p=root/'outputs/solar-research-v5';sys.path.insert(0,str(p))
from research import Dataset,probabilities,instances,read,write,OLD,DATA
from pipeline import load_image
from solar_v3 import segment_v3
from solar_v2 import ConfigV2
from organizer_metric import pool
from scipy import ndimage as ndi

r=read(p/'assessment.json')
if (p/'component-assessment.json').exists():
    extra=read(p/'component-assessment.json')
    r['results']['rf_filter']=extra['results'];r['rows']['rf_filter']=extra['rows']
    r['paired_gain_95_date_cluster_interval']['rf_filter']=extra['paired_gain_95_date_cluster_interval']
    r['frozen']['rf_filter']=extra['frozen']
    selected=read(p/'model-selection-followup.json')['method']
else:selected=max(r['frozen'],key=lambda m:r['frozen'][m]['tune']['pq'])
write(p/'combined-assessment.json',r)
v3=r['results']['v3'];new=r['results'][selected]
gain=new['pq']-v3['pq'];ci=r['paired_gain_95_date_cluster_interval'][selected]
labels={'v3':'Dotychczasowy v3','rf':'Las pikselowy','unet300':'U-Net 300','unet600':'U-Net 600','unet900':'U-Net 900','rf_filter':'Las + filtr obiektów'}
lines=['SOLAR: LOKALNE PORÓWNANIE NOWYCH MODELI','Autor: Jacek Adamek','',
       'Materiałów z tego eksperymentu nie opublikowano ani nie wysłano. Zgłoszony plik v3 pozostaje bez zmian.',
       f'Dane: 170 obrazów treningowych, 55 do wyboru ustawień, {r["images"]} do oceny ({r["days"]} dat).',
       'Dodatkowy filtr obiektów: dawne 55 obrazów strojenia podzielone na 36 do uczenia filtra i 19 do wyboru progu.',
       'Ustawienia modeli pikselowych wcześniej wybierano na całych 55 obrazach; dalszy wybór pozostaje rozwojowy.',
       'Daty zbioru oceny były oglądane we wcześniejszych badaniach. Nie jest to nowy ślepy test.',
       'Rozdzielenie dat nie wyklucza korelacji obserwacji z sąsiednich dni.',
       'Metryka: lokalna implementacja publicznego ewaluatora v6, osobno dla każdego zestawu anotacji.',
       'Wyników lokalnych nie należy porównywać liczbowo z publicznym wynikiem Kaggle 0,14.',
       '', 'Metoda | PQ | TP | FP | FN | precyzja | czułość']
for m,t in r['results'].items():
    lines.append(f"{labels[m]} | {t['pq']:.6f} | {t['tp']} | {t['fp']} | {t['fn']} | {t['precision']:.4f} | {t['recall']:.4f}")
lines += ['',f'Wybór na danych rozwojowych, przed wynikiem oceny: {labels[selected]}.',
          f'Zmiana PQ względem v3: {gain:+.6f} ({100*gain/v3["pq"]:+.1f}%).',
          f'95% przedział sparowanego bootstrapu po datach: [{ci[0]:.6f}, {ci[1]:.6f}].',
          'Przedział opisuje zmienność dla tych dat; nie usuwa obciążenia z wcześniejszego rozwoju metody.',
          '', 'Ograniczenia i ocena szans:',
          'Nie uzyskano nowego wyniku Kaggle i nie można przypisać nowego miejsca w rankingu.',
          'Nie ma podstaw do gwarantowania nagrody. Ocenę jakości trzeba oddzielić od samego przyjęcia zgłoszenia.',
          'Sieć jest małym modelem uczonym od zera na CPU. Pozostają rozdzielanie dotykających się obiektów,',
          'zachowanie cienkich struktur oraz zmienność masek między ekspertami.',
          '', 'Odtwarzanie: REPRODUCE.txt; pełne wyniki: combined-assessment.json; testy: tests.txt.',
          'Źródła metody i licencje: SOURCES.txt.']
(p/'Wyniki-Solar.txt').write_text('\n'.join(lines),encoding='utf-8')
write(p/'selected-local-model.json',{'method':selected,'selection':'Highest tune PQ, before consulting assessment','local_assessment_pq':new['pq'],'gain_vs_v3':gain,'paired_gain_interval':ci,'published':False,'submitted':False})

# Static scientific comparison, from measured counts only.
font='C:/Windows/Fonts/arial.ttf';bold='C:/Windows/Fonts/arialbd.ttf'
f=lambda n,b=False:ImageFont.truetype(bold if b else font,n)
im=Image.new('RGB',(1100,750),'white');dr=ImageDraw.Draw(im)
dr.text((36,24),'Solar: lokalna ocena 106 obrazów',font=f(32,True),fill='#14263c')
dr.text((36,72),'Te same daty i metryka dla wszystkich metod; wyników nie wysłano do Kaggle.',font=f(19),fill='#455363')
colors=['#8b98a8','#2c8994','#315cc5','#237849']
for i,(m,t) in enumerate(r['results'].items()):
    y=140+i*120
    dr.text((36,y),labels[m],font=f(24,True),fill='#14263c')
    dr.rounded_rectangle((290,y,290+int(t['pq']*1500),y+33),radius=5,fill=colors[i])
    dr.text((305+int(t['pq']*1500),y),f"PQ {t['pq']:.3f}",font=f(24,True),fill='#14263c')
    dr.text((290,y+45),f"TP {t['tp']}   FP {t['fp']}   FN {t['fn']}   czułość {100*t['recall']:.1f}%",font=f(20),fill='#455363')
dr.text((36,662),'Ocena rozwojowa: te daty były już oglądane. Brak nowego wyniku rankingowego.',font=f(19),fill='#8e421a')
dr.text((36,699),'Jacek Adamek | protokół, modele, logi i testy zapisane lokalnie',font=f(18),fill='#455363')
im.save(p/'porownanie-wynikow.png')

# Fixed first two assessment filenames; no selection by performance.
d=Dataset();oldmodel=read(OLD/'results/v3/frozen-model.json');cfg=ConfigV2(**read(OLD/'results/v2/frozen-config.json'))
frames=[]
for name in d.protocol['split']['assessment'][:2]:
    source=load_image(DATA/'train/train_images'/name);_,target,disk=d.arrays(name)
    old=segment_v3(source,cfg,oldmodel)
    from inference import segment as image_only_segment
    newlabels=image_only_segment(source,selected)
    t=np.asarray(Image.fromarray((target>.25).astype(np.uint8)).resize((2048,2048),Image.Resampling.NEAREST),bool)
    row=Image.new('RGB',(1500,540),'white');rd=ImageDraw.Draw(row)
    for j,(mask,title,color) in enumerate([(t,'Maski ekspertów (>25% głosów)',(45,200,80)),(old>0,'Dotychczasowy v3',(240,130,30)),(newlabels>0,labels[selected],(40,140,250))]):
        rgb=np.repeat(source[...,None],3,axis=-1);boundary=mask&~ndi.binary_erosion(mask,iterations=2)
        rgb[mask]=(.65*rgb[mask]+.35*np.array(color)).astype(np.uint8);rgb[boundary]=color
        row.paste(Image.fromarray(rgb).resize((500,500)),(500*j,40));rd.text((500*j+10,8),title,font=f(20,True),fill='#14263c')
    rd.text((12,517),name,font=f(16),fill='white');frames.append(row)
viz=Image.new('RGB',(1500,1080),'white')
for k,row in enumerate(frames):viz.paste(row,(0,k*540))
viz.save(p/'porownanie-masek.png')
print('\n'.join(lines),flush=True)
